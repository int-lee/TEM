"""
inference.py — sample a trained TEM and save the images and calculated metricsfor FID / sFID / IS / precision / recall.

Persistent-chain sampling (per GPU):
  1. Start with an EMPTY flat replay buffer (buffer_size split across GPUs).
  2. Each batch runs `total_langevin_steps` Langevin steps for random class labels;
     after `mid_step` steps the intermediate latents are pushed to the buffer.
  3. Until the buffer is full, chains start from pure noise; afterwards they start
     from the buffer with probability `replay_probability` (noise otherwise)..

Output: one .npz with
    arr_0 : uint8 [num_samples, H, W, 3]   (NHWC, as in ADM / DiT sample_ddp.py)
    arr_1 : int64 [num_samples]            class labels
plus sample_grid.png next to it.

Usage (4 GPUs):
    torchrun --nproc_per_node=4 --standalone inference.py \\
        --checkpoint /path/to/model_step_N.pth --model_size XL --resolution 256
    python evaluator.py VIRTUAL_imagenet256_labeled.npz samples/<name>.npz   # ADM evaluator
Every SampleConfig field is a --flag; run with -h to list them.
"""

import os
import random
import zipfile
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
from torch.amp import autocast
from torchvision import utils
from diffusers import AutoencoderKL
from tqdm import tqdm

from tem import build_tem
from utils import VAE_SCALE, parse_config


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SampleConfig:
    # --- Checkpoint ---
    checkpoint: str = ''               # path to model_step_N.pth, required
    use_ema: bool = True

    # --- Architecture (must match training) ---
    model_size: Literal['S', 'B', 'L', 'XL'] = 'XL'
    patch_size: int = 2
    num_energy_heads: int = 3
    head_layers: str = ''              # e.g. "8,17"; '' = layout saved in the checkpoint, else 1/3, 2/3 of depth
    resolution: int = 256              # 256 or 512 (latents are resolution/8)
    num_classes: int = 1000

    # --- Sampling ---
    total_langevin_steps: int = 60     # total Langevin steps per sample (last one is noise-free)
    mid_step: int = 40                 # step after which latents are saved to the buffer
    step_size: float = 500
    noise_std: float = 0.05
    clip_norm: float = 0.0             # per-sample L2 norm clip of the latent; 0 = 100 at 256px, 200 at 512px
    use_amp: bool = True

    # --- Buffer ---
    buffer_size: int = 5000            # total across GPUs
    replay_probability: float = 0.95   # P(init from buffer) once the buffer is full

    # --- Output ---
    num_samples: int = 50000
    batch_size: int = 32               # per GPU
    vae_batch_size: int = 8            # decode chunk size
    seed: int = 0
    output: str = ''                   # .npz path; '' -> samples/<run>_<step>_s<steps>_n<num_samples>.npz

    def __post_init__(self):
        if not self.checkpoint:
            raise ValueError("--checkpoint is required")
        assert 0 < self.mid_step < self.total_langevin_steps
        self.latent_size = self.resolution // 8
        self.in_channels = 4
        self.head_layers_tuple = tuple(int(v) for v in self.head_layers.split(',')) if self.head_layers else None
        if self.clip_norm <= 0:
            self.clip_norm = 100.0 if self.resolution <= 256 else 200.0
        if not self.output:
            run = os.path.basename(os.path.dirname(os.path.abspath(self.checkpoint)))
            step = os.path.splitext(os.path.basename(self.checkpoint))[0].replace('model_step_', 'step')
            self.output = os.path.join(
                "samples", f"{run}_{step}_s{self.total_langevin_steps}_n{self.num_samples}.npz")
        if not self.output.endswith('.npz'):
            self.output += '.npz'


# =============================================================================
# Langevin dynamics
# =============================================================================

def _clip_latent_norm(x, clip_norm):
    flat = x.view(x.shape[0], -1)
    scale = torch.clamp(clip_norm / flat.norm(dim=1, keepdim=True), max=1.0)
    return (flat * scale).view_as(x)


def _langevin_phase(neg_img, neg_id, model, steps, step_size, noise_std, clip_norm, use_amp):
    """`steps` noisy Langevin steps."""
    for _ in range(steps):
        neg_img.requires_grad_(True)
        neg_img = neg_img + torch.randn_like(neg_img) * noise_std
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            neg_out = model(neg_img, neg_id)
        im_grad = torch.autograd.grad(neg_out.sum(), neg_img, create_graph=False)[0]
        neg_img = _clip_latent_norm(neg_img - step_size * im_grad, clip_norm).detach()
    return neg_img


_langevin_phase_compiled = torch.compile(_langevin_phase)


def _langevin_final_step(neg_img, neg_id, model, step_size, clip_norm, use_amp):
    """Last Langevin step, without noise."""
    neg_img.requires_grad_(True)
    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
        neg_out = model(neg_img, neg_id)
    im_grad = torch.autograd.grad(neg_out.sum(), neg_img, create_graph=False)[0]
    return _clip_latent_norm(neg_img - step_size * im_grad, clip_norm).detach()


# =============================================================================
# Flat replay buffer
# =============================================================================

class FlatReplayBuffer:
    """Class-agnostic ring buffer of latents (stored on CPU)."""

    def __init__(self, max_size, latent_shape):
        self.max_size = max_size
        self.latent_shape = latent_shape
        self._storage = []
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def is_full(self):
        return len(self._storage) >= self.max_size

    def add_batch(self, latents):
        for item in latents.detach().cpu():
            if len(self._storage) < self.max_size:
                self._storage.append(item)
            else:
                self._storage[self._next_idx] = item
                self._next_idx = (self._next_idx + 1) % self.max_size

    def sample(self, batch_size, p, device):
        """Each item comes from the buffer with probability p, else N(0, I)."""
        imgs = []
        for _ in range(batch_size):
            if random.random() < p and len(self._storage) > 0:
                imgs.append(random.choice(self._storage).to(device))
            else:
                imgs.append(torch.randn(*self.latent_shape, device=device))
        return torch.stack(imgs)


# =============================================================================
# Model loading
# =============================================================================

def load_model(config, rank):
    # mmap on CPU: the .pth also holds optimizer state + replay buffers we don't need.
    checkpoint = torch.load(config.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if config.use_ema and checkpoint.get('ema_model_state_dict') is not None:
        state_dict, which = checkpoint['ema_model_state_dict'], 'EMA'
    else:
        state_dict, which = checkpoint['model_state_dict'], 'raw'
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

    # The label table has num_classes (+1 if trained with label dropout) rows.
    num_classes_model = state_dict['y_embedder.embedding_table.weight'].shape[0]

    # Head blocks: --head_layers if given, else what training recorded, else 1/3, 2/3 of depth.
    head_layers = config.head_layers_tuple
    saved = checkpoint.get('head_layers')
    if head_layers is None and saved is not None:
        head_layers = tuple(saved)
    elif head_layers is not None and saved is not None and tuple(saved) != head_layers:
        raise ValueError(f"--head_layers {head_layers} does not match the checkpoint's {tuple(saved)}")
    model = build_tem(config.model_size, input_size=config.latent_size, in_channels=config.in_channels,
                      num_classes=num_classes_model, patch_size=config.patch_size,
                      num_energy_heads=config.num_energy_heads, head_layers=head_layers)
    model.load_state_dict(state_dict)
    if rank == 0:
        print(f"Loaded {which} weights from {config.checkpoint} (step {checkpoint.get('step', '?')}), "
              f"energy heads after blocks {model.head_layers} + final")
        if saved is None and config.head_layers_tuple is None:
            print("Warning: checkpoint does not record head_layers; using the 1/3, 2/3 default "
                  "(pass --head_layers if it was trained with another layout).")
    return model.to(rank).eval().requires_grad_(False)


# =============================================================================
# Output
# =============================================================================

def write_npz(path, image_shards, label_shards):
    """Write arr_0 (uint8 NHWC images) and arr_1 (labels) to an uncompressed .npz,
    streaming the per-GPU shards so the full array is never held in memory."""
    images = [np.load(f, mmap_mode='r') for f in image_shards]
    labels = np.concatenate([np.load(f) for f in label_shards])
    shape = (sum(a.shape[0] for a in images),) + images[0].shape[1:]
    tmp = f"{path}.tmp{os.getpid()}"
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with zf.open('arr_0.npy', 'w', force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(
                f, {'descr': np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
                    'fortran_order': False, 'shape': shape})
            for a in images:
                for i in range(0, a.shape[0], 1000):
                    f.write(np.ascontiguousarray(a[i:i + 1000]).tobytes())
        with zf.open('arr_1.npy', 'w', force_zip64=True) as f:
            np.lib.format.write_array(f, labels)
    os.replace(tmp, path)
    return shape


# =============================================================================
# Main worker
# =============================================================================

def main_worker(rank, world_size, config):
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)

    out_dir = os.path.dirname(os.path.abspath(config.output))
    shard_dir = config.output + ".shards"
    if rank == 0:
        os.makedirs(shard_dir, exist_ok=True)
        print(f"Configuration: {config}")
    dist.barrier()

    model = load_model(config, rank)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(rank)
    vae.eval()
    vae.requires_grad_(False)
    vae = torch.compile(vae)

    @torch.no_grad()
    def decode_uint8(z):
        """Latents -> uint8 NHWC images (DiT sample_ddp.py quantization)."""
        out = []
        for i in range(0, z.shape[0], config.vae_batch_size):
            x = vae.decode(z[i:i + config.vae_batch_size] / VAE_SCALE).sample
            x = torch.clamp(127.5 * x + 128.0, 0, 255)
            out.append(x.permute(0, 2, 3, 1).to(torch.uint8).cpu())
        return torch.cat(out).numpy()

    latent_shape = (config.in_channels, config.latent_size, config.latent_size)
    buffer = FlatReplayBuffer(config.buffer_size // world_size, latent_shape)
    n_local = config.num_samples // world_size + (1 if rank < config.num_samples % world_size else 0)
    bs = config.batch_size
    phase1_steps = config.mid_step
    phase2_steps = config.total_langevin_steps - config.mid_step - 1  # last step is the noise-free one
    langevin_kw = dict(step_size=config.step_size, noise_std=config.noise_std,
                       clip_norm=config.clip_norm, use_amp=config.use_amp)

    if rank == 0:
        print(f"Sampling {config.num_samples} images ({n_local}/GPU x {world_size} GPUs, batch {bs}), "
              f"{config.total_langevin_steps} Langevin steps, buffer {config.buffer_size} "
              f"(save after step {config.mid_step})")

    images, labels, done = [], [], 0
    pbar = tqdm(total=n_local, desc="Sampling", unit="img", disable=(rank != 0))
    while done < n_local:
        if buffer.is_full:
            neg_img = buffer.sample(bs, p=config.replay_probability, device=rank)
        else:
            neg_img = torch.randn(bs, *latent_shape, device=rank)
        neg_id = torch.randint(0, config.num_classes, (bs,), device=rank)

        neg_img = _langevin_phase_compiled(neg_img, neg_id, model, steps=phase1_steps, **langevin_kw)
        buffer.add_batch(neg_img)
        neg_img = _langevin_phase_compiled(neg_img, neg_id, model, steps=phase2_steps, **langevin_kw)
        neg_img = _langevin_final_step(neg_img, neg_id, model, step_size=config.step_size,
                                       clip_norm=config.clip_norm, use_amp=config.use_amp)

        keep = min(bs, n_local - done)
        images.append(decode_uint8(neg_img[:keep]))
        labels.append(neg_id[:keep].cpu().numpy())
        done += keep
        pbar.update(keep)
    pbar.close()

    np.save(os.path.join(shard_dir, f"images_rank{rank}.npy"), np.concatenate(images))
    np.save(os.path.join(shard_dir, f"labels_rank{rank}.npy"), np.concatenate(labels).astype(np.int64))
    del images, labels
    dist.barrier()

    if rank == 0:
        image_shards = [os.path.join(shard_dir, f"images_rank{r}.npy") for r in range(world_size)]
        label_shards = [os.path.join(shard_dir, f"labels_rank{r}.npy") for r in range(world_size)]
        shape = write_npz(config.output, image_shards, label_shards)
        grid = torch.from_numpy(np.load(image_shards[0], mmap_mode='r')[:64].copy()).permute(0, 3, 1, 2)
        utils.save_image(grid.float() / 255.0, os.path.join(out_dir, os.path.basename(config.output)[:-4] + "_grid.png"),
                         nrow=8)
        for f in image_shards + label_shards:
            os.remove(f)
        os.rmdir(shard_dir)
        print(f"Saved {shape} uint8 samples -> {config.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    config = parse_config(SampleConfig, description=__doc__)
    torch.backends.cudnn.benchmark = True
    main_worker(local_rank, world_size, config)
