"""
train.py — train a TEM (Transformer Energy Model) on ImageNet in SD-VAE latent space.

Each micro-step:
  1. Encode the real batch with the SD VAE (positives).
  2. Draw negatives from a class-conditional replay buffer (or noise) and run
     `sample_step` Langevin steps on the energy.
  3. Contrastive-divergence loss with an L2 energy regularizer, plus
       - KL term: one differentiable Langevin step through a frozen copy of the
         energy (second-order gradient),
       - repulsion of the negatives from buffer samples,
       - PRA (Positive Representation Alignment): align the block-`pra_layer`
         tokens of the positives with spatially normalized DINOv2-L patch features.
  4. Push the negatives back into the buffer.
Gradients are accumulated over `grad_accum_steps` micro-steps; an EMA copy of the
weights is kept for sampling / FID.

Step counters (checkpoint names, logs, intervals) count micro-steps (data
batches), not optimizer steps.

Usage (4 GPUs):
    torchrun --nproc_per_node=4 --standalone train.py --model_size L --resolution 512
    torchrun --nproc_per_node=4 --standalone train.py --model_size XL --resolution 256 --use_wandb false
Every TrainConfig field is a --flag; run with -h to list them.

Resume_from_step=-1 (default) resumes from the newest intact
checkpoint in the run directory, so a requeued SLURM job continues where it stopped.
"""

import os
os.environ.setdefault("WANDB_CACHE_DIR", "wandbcache")
import glob
import hashlib
import random
import re
import zipfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import utils
from tqdm import tqdm
from diffusers import AutoencoderKL
from transformers import AutoModel
import wandb

from tem import TEM_CONFIGS, build_tem, default_head_layers
from utils import (VAE_SCALE, InceptionFeatures, calculate_fid, decode_latents, feature_stats,
                   imagenet_dataset, inception_features, load_or_compute_real_stats, parse_config)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # --- Model ---
    model_size: Literal['S', 'B', 'L', 'XL'] = 'L'
    patch_size: int = 2
    num_energy_heads: int = 3          # 1: final head only, 2: + middle head, 3: + early head
    head_layers: str = ''              # blocks for early,middle heads, e.g. "8,17"; '' = 1/3, 2/3 of depth

    # --- Data ---
    data_path: str = ''                # ImageNet train root (n######## folders), required
    resolution: int = 256              # 256 or 512 (latents are resolution/8)
    num_classes: int = 1000
    num_workers: int = 0

    # --- Conditioning ---
    cond_dropout_prob: float = 0.0     # >0 adds a null label (= num_classes) and logs unconditional FID
    pos_neg_id_matching: bool = True   # negatives are sampled with the positives' labels

    # --- PRA: Positive Representation Alignment with DINOv2-L ---
    pra: bool = True
    pra_coeff: float = 0.5
    pra_layer: int = -1                # 0-indexed block whose tokens are aligned; -1 = block at 1/3 depth
    pra_gamma: float = 0.6             # spatial normalization of the DINOv2 targets: x - gamma * mean(x)

    # --- Langevin sampling ---
    step_size: float = 500
    sample_step: int = 40
    noise_std: float = 0.05
    clip_norm: float = 0.0             # per-sample L2 norm clip of the latent; 0 = 100 at 256px, 200 at 512px
    resamples: int = 1

    # --- Replay buffer (per GPU) ---
    buffer_size: int = 4000            # split evenly across classes
    replay_probability: float = 0.99

    # --- Optimization ---
    alpha: float = 1.0                 # weight of the energy-magnitude (L2) regularizer
    batch_size: int = 8                # per GPU
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    grad_clip: bool = True             # max grad norm 1.0
    use_amp: bool = True               # bf16 autocast
    kl: bool = True
    kl_coeff: float = 1.0
    repel_im: bool = True
    repel_coeff: float = 0.3
    use_ema: bool = True
    ema_decay: float = 0.9999

    # --- Logging / checkpoints ---
    project_name: str = "TEM-imagenet"
    experiment_name: str = ''          # '' -> TEM-<size>-<patch>_in<resolution>
    output_dir: str = "./checkpoints"  # checkpoints go to <output_dir>/<experiment_name>
    resume_from_step: int = -1         # -1: newest checkpoint, 0: fresh start, >0: that step
    max_steps: int = 0                 # stop after this step (0 = train until killed)
    use_wandb: bool = True
    log_interval: int = 10
    sample_interval: int = 500
    save_interval: int = 1000
    fid_interval: int = 100
    fid_samples: int = 1000
    fid_real_samples: int = 10000
    real_stats: str = ''               # npz with mu/sigma; '' -> <output_dir>/fid_stats_imagenet<res>_train_<N>.npz
    seed: int = 42                     # data-order seed

    def __post_init__(self):
        if not self.data_path:
            raise ValueError("--data_path (ImageNet train folder) is required")
        assert self.resolution % 16 == 0, self.resolution
        self.latent_size = self.resolution // 8
        self.in_channels = 4
        self.head_layers_tuple = tuple(int(v) for v in self.head_layers.split(',')) if self.head_layers else None
        if self.pra_layer < 0:
            self.pra_layer = default_head_layers(TEM_CONFIGS[self.model_size]['depth'])[0]
        if self.clip_norm <= 0:
            self.clip_norm = 100.0 if self.resolution <= 256 else 200.0
        if not self.experiment_name:
            self.experiment_name = f"TEM-{self.model_size}-{self.patch_size}_in{self.resolution}"
        self.output_dir = os.path.abspath(self.output_dir)
        self.save_dir = os.path.join(self.output_dir, self.experiment_name)
        if not self.real_stats:
            self.real_stats = os.path.join(
                self.output_dir, f"fid_stats_imagenet{self.resolution}_train_{self.fid_real_samples}.npz")
        self.tags = ["energy-based-model", "TEM", self.model_size, f"imagenet{self.resolution}"]

        self.resume_from_checkpoint = None
        if self.resume_from_step < 0:
            self.resume_from_step, self.resume_from_checkpoint = find_latest_checkpoint(
                self.save_dir, world_size=int(os.environ.get("WORLD_SIZE", 1)))
        elif self.resume_from_step > 0:
            self.resume_from_checkpoint = os.path.join(self.save_dir, f'model_step_{self.resume_from_step}.pth')


# =============================================================================
# Checkpoint helpers
# =============================================================================

def checkpoint_is_complete(path):
    """torch.save writes a zip archive whose central directory lands last, so a
    checkpoint truncated by preemption fails to open here."""
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
        return True
    except Exception:
        return False


def find_latest_checkpoint(save_dir, world_size=1):
    """Newest resumable step in save_dir as (step, path); (0, None) if there is none.

    A step is resumable only if the rank-0 file and every rank's buffer file are
    present and intact, so a save interrupted by preemption falls back to the
    previous step instead of crashing the requeued job.
    """
    candidates = []
    for path in glob.glob(os.path.join(save_dir, 'model_step_*.pth')):
        match = re.fullmatch(r'model_step_(\d+)\.pth', os.path.basename(path))
        if match:
            candidates.append((int(match.group(1)), path))

    for step, path in sorted(candidates, reverse=True):
        files = [path] + [path.replace('.pth', f'_rank_{r}_buffer.pth') for r in range(1, world_size)]
        missing = [f for f in files if not checkpoint_is_complete(f)]
        if not missing:
            return step, path
        print(f"Skipping step {step}: missing/truncated {[os.path.basename(f) for f in missing]}")
    return 0, None


def atomic_save(obj, path):
    """Write to a temp file then rename, so preemption mid-save can't leave a
    half-written file at `path`."""
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


# =============================================================================
# Replay buffer
# =============================================================================

class ReplayBuffer:
    """Class-conditional ring buffer of latents: buffer_size // num_classes per class."""

    def __init__(self, config):
        self.config = config
        self.num_classes = config.num_classes
        self.latent_shape = (config.in_channels, config.latent_size, config.latent_size)
        self._storage = {i: [] for i in range(self.num_classes)}
        self._maxsize = max(config.buffer_size // self.num_classes, 1)
        self._next_idx = {i: 0 for i in range(self.num_classes)}

    def __len__(self):
        return sum(len(l) for l in self._storage.values())

    def add(self, ims, labels):
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        for im, l in zip(ims, labels):
            l = int(l)
            if l >= self.num_classes:  # null (dropped) label
                continue
            storage = self._storage[l]
            if len(storage) < self._maxsize:
                storage.append(im)
            else:
                storage[self._next_idx[l]] = im
                self._next_idx[l] = (self._next_idx[l] + 1) % self._maxsize

    def sample(self, batch_size, p, device, ids=None):
        """Latents for `ids` (random classes if None): each one comes from that
        class's buffer with probability p (if non-empty), else N(0, I).
        Returns (latents, ids) with label dropout applied to ids."""
        if ids is None:
            ids = torch.randint(0, self.num_classes, (batch_size,), device=device)
        images = torch.randn((ids.shape[0], *self.latent_shape), device=device)
        for i, cid in enumerate(ids.cpu().numpy()):
            buffer_list = self._storage.get(int(cid), [])
            if random.random() < p and len(buffer_list) > 0:
                images[i] = random.choice(buffer_list).to(device)

        ids = ids.clone()
        mask = torch.rand(ids.shape[0], device=device) < self.config.cond_dropout_prob
        ids[mask] = self.config.num_classes
        return images, ids

    def state_dict(self):
        return {'storage': self._storage, 'next_idx': self._next_idx}

    def load_state_dict(self, state):
        self._storage = state['storage']
        self._next_idx = state['next_idx']


# =============================================================================
# Data
# =============================================================================

def get_stateful_distributed_loader(dataset, config, rank, world_size, start_step=0, seed=42):
    """Infinite DDP loader that resumes deterministically at `start_step`."""
    steps_per_epoch = len(DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                             drop_last=True)) // config.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"Batch size {config.batch_size} is too large for dataset partition on rank {rank}.")

    current_epoch = start_step // steps_per_epoch
    steps_to_skip = start_step % steps_per_epoch

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    while True:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(current_epoch)

        # Drop the already-seen indices from this epoch's shuffled order instead of
        # loading and discarding the images, so resuming takes seconds, not hours.
        if steps_to_skip > 0:
            if rank == 0:
                print(f"Resuming: fast-forwarding {steps_to_skip} steps in epoch {current_epoch}...")
            sampler = list(sampler)[steps_to_skip * config.batch_size:]
            steps_to_skip = 0

        g = torch.Generator()
        g.manual_seed(seed + current_epoch)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                            num_workers=config.num_workers, pin_memory=True,
                            worker_init_fn=seed_worker, generator=g, drop_last=True)
        yield from loader
        current_epoch += 1


# =============================================================================
# Langevin sampling
# =============================================================================

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def clip_latent_norm(x, clip_norm):
    flat = x.view(x.size(0), -1)
    scale = torch.clamp(clip_norm / flat.norm(dim=1, keepdim=True), max=1.0)
    return (flat * scale).view_as(x)


def langevin_dynamics(neg_img, neg_id, model, steps: int, step_size: float,
                      noise_std: float, clip_norm: float, use_amp: bool):
    for _ in range(steps):
        neg_img.requires_grad_(True)
        neg_img = neg_img + torch.randn_like(neg_img) * noise_std
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            neg_out = model(neg_img, neg_id)
        im_grad = torch.autograd.grad(neg_out.sum(), neg_img, create_graph=False)[0]
        neg_img = clip_latent_norm(neg_img - step_size * im_grad, clip_norm).detach()
    return neg_img

langevin_dynamics_compiled = torch.compile(langevin_dynamics)


def generate_samples(model, input_id, buffer, n_samples, config, sampling=False):
    """Run Langevin from buffer/noise inits.

    Returns (neg_img, neg_id, neg_img_kl, neg_id_kl). In training mode with
    config.kl, neg_img_kl is the result of one extra Langevin step that keeps the
    graph to the model parameters (used for the KL loss).
    """
    replay_prob = 1.0 if sampling else config.replay_probability
    neg_img, neg_id = buffer.sample(n_samples, p=replay_prob, device=config.device)
    neg_img = neg_img.repeat_interleave(config.resamples, dim=0)
    if input_id is not None:
        neg_id = input_id
    neg_id = neg_id.repeat_interleave(config.resamples)

    requires_grad(model.parameters(), False)
    neg_img = langevin_dynamics_compiled(
        neg_img, neg_id, model, steps=config.sample_step - 1, step_size=config.step_size,
        noise_std=config.noise_std, clip_norm=config.clip_norm, use_amp=config.use_amp)
    if not sampling:
        requires_grad(model.parameters(), True)

    # Last step without noise.
    neg_img.requires_grad = True
    neg_img_kl = neg_img
    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp):
        neg_out = model(neg_img, neg_id)
    im_grad = torch.autograd.grad(neg_out.sum(), neg_img, create_graph=False)[0]
    neg_img = neg_img - config.step_size * im_grad

    neg_id_kl = None
    if not sampling and config.kl:
        neg_id_kl = neg_id
        # The math SDPA kernel supports the double backward the KL term needs.
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                neg_kl_out = model(neg_img_kl, neg_id_kl)
        im_grad = torch.autograd.grad([neg_kl_out.sum()], [neg_img_kl], create_graph=True)[0]
        neg_img_kl = clip_latent_norm(neg_img_kl - config.step_size * im_grad, config.clip_norm)

    neg_img = clip_latent_norm(neg_img, config.clip_norm).detach()
    return neg_img, neg_id, neg_img_kl, neg_id_kl


# =============================================================================
# Training step
# =============================================================================

@torch.no_grad()
def update_ema(model, model_ema, decay):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data.mul_(decay).add_(param.data, alpha=1 - decay)


def train_step(model, model_ema, pos_img, pos_id, buffer, vae, encoder_model, optimizer, config,
               is_update_step=True):
    """One micro-step: forward, loss, backward; optimizer/EMA update if is_update_step."""
    # Positives: images in [0, 1] -> SD-VAE latents.
    pos_img_pixels = pos_img
    with torch.no_grad():
        pos_img = vae.encode(2 * pos_img - 1).latent_dist.sample() * VAE_SCALE

    pos_id = pos_id.clone()
    pos_id[torch.rand(pos_id.shape[0], device=config.device) < config.cond_dropout_prob] = config.num_classes
    input_id = pos_id if config.pos_neg_id_matching else None

    # Negatives. Sampling (and its second-order KL graph) runs on model.module:
    # DDP's reducer can't track it, so gradients are synced by hand below.
    neg_img, neg_id, neg_img_kl, neg_id_kl = generate_samples(
        model.module, input_id, buffer, pos_img.shape[0], config, sampling=False)

    model.train()
    log_dict = {}
    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp):
        neg_out = model.module(neg_img.clone(), neg_id)
        if config.pra:
            pos_out, pra_act = model.module(pos_img, pos_id, pra_layer=config.pra_layer)
        else:
            pos_out = model.module(pos_img, pos_id)

        # Contrastive divergence + L2 regularizer, summed over energy heads.
        pos_loss = config.alpha * (pos_out ** 2).sum(dim=1)
        neg_loss = config.alpha * (neg_out ** 2).sum(dim=1)
        energy_gap = pos_out.sum(dim=1) - neg_out.sum(dim=1)
        loss = (pos_loss + neg_loss + energy_gap).mean()
        log_dict['pos_energy_mean'] = pos_out.sum(dim=1).mean().item()
        log_dict['neg_energy_mean'] = neg_out.sum(dim=1).mean().item()

        if config.kl and neg_img_kl is not None:
            # Energy of the differentiable sample under frozen weights: gradients
            # flow only through the sampling step.
            model.requires_grad_(False)
            loss_kl = model.module(neg_img_kl, neg_id_kl)
            model.requires_grad_(True)
            kl_term = config.kl_coeff * loss_kl.sum(dim=1).mean()

            if config.repel_im and len(buffer) > 1000:
                im_flat = neg_img_kl.view(neg_img_kl.shape[0], -1)
                neg_img_buf, _ = buffer.sample(100, p=1, device=config.device)
                dist_matrix = torch.norm(im_flat[:, None, :] - neg_img_buf.view(100, -1)[None, :, :], p=2, dim=-1)
                loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                loss = loss - config.repel_coeff * loss_repel

            loss = loss + kl_term
            log_dict['kl_loss'] = kl_term.item()

        if config.pra:
            # DINOv2 (patch 14) at 14 * tokens-per-side gives one feature per TEM token.
            dino_res = 14 * (config.latent_size // config.patch_size)
            dino_in = F.interpolate(pos_img_pixels, size=(dino_res, dino_res), mode='bicubic')
            with torch.no_grad():
                dino_features = encoder_model(dino_in.to(torch.bfloat16)).last_hidden_state[:, 1:, :].float()
                dino_features = dino_features - config.pra_gamma * dino_features.mean(dim=1, keepdim=True)
                dino_features = dino_features / (dino_features.std(dim=1, keepdim=True) + 1e-6)
            pra_loss = 1 - F.cosine_similarity(dino_features, pra_act, dim=-1).mean()
            loss = loss + config.pra_coeff * pra_loss
            log_dict['pra_loss'] = pra_loss.item()

    loss = loss / config.grad_accum_steps
    log_dict['loss'] = loss.item() * config.grad_accum_steps
    loss.backward()

    if is_update_step:
        # Zero-weighted pass through the DDP wrapper: fires DDP's all-reduce hooks
        # so the locally accumulated gradients get averaged across ranks.
        dummy_in = torch.zeros((1, config.in_channels, config.latent_size, config.latent_size),
                               device=config.device, requires_grad=True)
        dummy_id = torch.zeros((1,), device=config.device, dtype=torch.long)
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp):
            dummy_out = model(dummy_in, dummy_id)
        (dummy_out.sum() * 0.0).backward()

        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if config.use_ema:
            update_ema(model.module, model_ema, config.ema_decay)

    buffer.add(neg_img.detach().cpu(), neg_id.cpu())
    return log_dict, neg_img.detach(), neg_id.detach()


# =============================================================================
# FID during training
# =============================================================================

def distributed_fid_features(model, buffer, vae, inception, n_local, config, labels=None):
    """Each rank samples n_local images and extracts Inception features; rank 0
    gets the concatenation of all ranks' features (others get None)."""
    feats = []
    for start in range(0, n_local, 32):
        bs = min(32, n_local - start)
        ids = None if labels is None else torch.full((bs,), labels, device=config.device, dtype=torch.long)
        samples, _, _, _ = generate_samples(model, ids, buffer, bs, config, sampling=True)
        feats.append(inception_features(decode_latents(vae, samples), inception, config.device)[0])
    local = torch.from_numpy(np.concatenate(feats)).to(config.device)

    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(local) for _ in range(world_size)] if dist.get_rank() == 0 else None
    dist.gather(local, gather_list=gathered, dst=0)
    return torch.cat(gathered).cpu().numpy() if gathered is not None else None


# =============================================================================
# Main worker
# =============================================================================

def main_worker(rank, world_size, config):
    # Long timeout: rank 0 may spend a while computing the real FID stats on a first run.
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.cuda.set_device(rank)
    config.device = rank

    if rank == 0:
        # Fixed id + resume so a requeued job continues the same wandb run.
        wandb.init(project=config.project_name, name=config.experiment_name,
                   id=hashlib.md5(config.experiment_name.encode()).hexdigest()[:16], resume='allow',
                   config=config.__dict__, tags=config.tags,
                   mode='online' if config.use_wandb else 'disabled')
        print(f"Configuration: {config}")
        os.makedirs(config.save_dir, exist_ok=True)

    # --- Models ---
    num_classes_model = config.num_classes + 1 if config.cond_dropout_prob > 0 else config.num_classes
    model_kwargs = dict(input_size=config.latent_size, in_channels=config.in_channels,
                        num_classes=num_classes_model, patch_size=config.patch_size,
                        num_energy_heads=config.num_energy_heads, head_layers=config.head_layers_tuple,
                        pra_dim=1024)
    model = build_tem(config.model_size, **model_kwargs).to(rank)
    model_ema = build_tem(config.model_size, **model_kwargs).to(rank)
    model_ema.load_state_dict(model.state_dict())
    model_ema.requires_grad_(False)
    if rank == 0:
        print(f"TEM-{config.model_size}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        wandb.watch(model, log_freq=100)

    model = DDP(model, device_ids=[rank])
    model._set_static_graph()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.0, 0.999))

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(rank)
    vae.eval()
    vae.requires_grad_(False)
    vae = torch.compile(vae)

    encoder_model = None
    if config.pra:
        encoder_model = AutoModel.from_pretrained('facebook/dinov2-large').to(rank).half()
        encoder_model.eval()
        encoder_model.requires_grad_(False)
        encoder_model = torch.compile(encoder_model)

    # --- Data ---
    dataset = imagenet_dataset(config.data_path, config.resolution, flip=True)
    assert len(dataset.classes) == config.num_classes, (len(dataset.classes), config.num_classes)
    data_loader = get_stateful_distributed_loader(dataset, config, rank=rank, world_size=world_size,
                                                  start_step=config.resume_from_step, seed=config.seed)
    buffer = ReplayBuffer(config)

    # --- FID: Inception on every rank (features are gathered), real stats on rank 0 ---
    inception = InceptionFeatures().to(rank)
    real_mu = real_sigma = None
    if rank == 0:
        real_mu, real_sigma = load_or_compute_real_stats(
            config.real_stats, lambda: imagenet_dataset(config.data_path, config.resolution, flip=False),
            inception, rank, config.fid_real_samples, num_workers=config.num_workers)
    dist.barrier()

    # --- Resume ---
    start_step = 0
    if config.resume_from_checkpoint is not None:
        if rank == 0:
            print(f"Loading checkpoint from {config.resume_from_checkpoint}")
        # Staged on CPU: load_state_dict copies into the resident GPU params.
        checkpoint = torch.load(config.resume_from_checkpoint, map_location='cpu', weights_only=False)
        # Same parameters, different blocks: a head-layout mismatch would load
        # cleanly but read the wrong blocks, so refuse it.
        saved = checkpoint.get('head_layers')
        if saved is not None and tuple(saved) != model.module.head_layers:
            raise ValueError(f"checkpoint has head_layers={tuple(saved)}, model has {model.module.head_layers}; "
                             f"pass --head_layers {','.join(map(str, saved))}")
        if saved is None and rank == 0:
            print(f"Warning: checkpoint does not record head_layers; assuming {model.module.head_layers}")
        model.module.load_state_dict(checkpoint['model_state_dict'])
        if config.use_ema and checkpoint.get('ema_model_state_dict') is not None:
            model_ema.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)

        if rank == 0:
            buffer.load_state_dict(checkpoint['buffer_state'])
        else:
            buffer_file = config.resume_from_checkpoint.replace('.pth', f'_rank_{rank}_buffer.pth')
            buffer.load_state_dict(torch.load(buffer_file, map_location='cpu', weights_only=False)['buffer_state'])
        if rank == 0:
            print(f"Resumed from step {start_step}; rank-0 buffer has {len(buffer)} samples")
        checkpoint = None
        torch.cuda.empty_cache()

    # --- Training loop ---
    data_iter = enumerate(data_loader)
    if rank == 0:
        data_iter = tqdm(data_iter, desc="Training", initial=start_step)
    optimizer.zero_grad()

    for i, (pos_img, pos_id) in data_iter:
        step = i + start_step
        pos_img, pos_id = pos_img.to(rank), pos_id.to(rank)
        is_update_step = (i + 1) % config.grad_accum_steps == 0

        log_dict, neg_img, neg_id = train_step(model, model_ema, pos_img, pos_id, buffer, vae,
                                               encoder_model, optimizer, config, is_update_step)

        if rank == 0:
            log_dict['step'] = step
            log_dict['buffer_size'] = len(buffer)
            log_dict['energy_gap'] = log_dict['pos_energy_mean'] - log_dict['neg_energy_mean']
            if step % config.log_interval == 0:
                wandb.log(log_dict)
                desc = f'Training - Step {step} - Loss: {log_dict["loss"]:.5f}'
                if 'kl_loss' in log_dict:
                    desc += f' - KL: {log_dict["kl_loss"]:.5f}'
                data_iter.set_description(desc)

            if i % config.sample_interval == 0:
                grid = utils.make_grid(decode_latents(vae, neg_img[:9]).cpu(), nrow=3)
                wandb.log({'generated_samples': wandb.Image(grid, caption=f'Generated samples at step {step}'),
                           'generated_labels': [int(l) for l in neg_id[:9].cpu()],
                           'step': step})

        # --- Checkpoint: rank 0 saves model/optimizer/EMA + its buffer, other ranks their buffer ---
        if step % config.save_interval == 0 and i > 0:
            if rank == 0:
                print(f"\nStep {step}: saving model...")
                atomic_save({
                    'step': step,
                    'model_state_dict': model.module.state_dict(),
                    'ema_model_state_dict': model_ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': log_dict['loss'],
                    'config': config.__dict__,
                    'head_layers': model.module.head_layers,
                    'buffer_state': buffer.state_dict(),
                }, f'{config.save_dir}/model_step_{step}.pth')
            else:
                atomic_save({'buffer_state': buffer.state_dict()},
                            f'{config.save_dir}/model_step_{step}_rank_{rank}_buffer.pth')
            dist.barrier()

        # --- FID of EMA samples (fid_samples split across ranks) ---
        if i % config.fid_interval == 0 and i > 0:
            n_local = config.fid_samples // world_size
            fid_line = None
            feats = distributed_fid_features(model_ema, buffer, vae, inception, n_local, config)
            if rank == 0:
                fid = calculate_fid(real_mu, real_sigma, *feature_stats(feats))
                print(f"FID at step {step}: {fid:.4f}")
                wandb.log({'fid_score': fid, 'step': step})
                fid_line = f"Step {step}: FID = {fid:.4f}"

            if config.cond_dropout_prob > 0:
                feats = distributed_fid_features(model_ema, buffer, vae, inception, n_local, config,
                                                 labels=config.num_classes)
                if rank == 0:
                    uncond_fid = calculate_fid(real_mu, real_sigma, *feature_stats(feats))
                    print(f"Uncond FID at step {step}: {uncond_fid:.4f}")
                    wandb.log({'uncond_fid_score': uncond_fid, 'step': step})
                    fid_line += f", UncondFID = {uncond_fid:.4f}"

            if rank == 0:
                with open(f'{config.save_dir}/fid_scores.txt', 'a') as f:
                    f.write(fid_line + "\n")
            feats = None
            torch.cuda.empty_cache()
            dist.barrier()

        if config.max_steps and step >= config.max_steps:
            if rank == 0:
                print(f"Reached max_steps={config.max_steps}, stopping.")
            break

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    config = parse_config(TrainConfig, description=__doc__)
    if local_rank == 0:
        if config.resume_from_checkpoint is None:
            print("No checkpoint found, starting from step 0")
        else:
            print(f"Resuming from {config.resume_from_checkpoint}")

    torch.backends.cudnn.benchmark = True
    main_worker(local_rank, world_size, config)
