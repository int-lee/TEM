"""Shared helpers: ImageNet loading, VAE decoding, Inception/FID (training monitor), config CLI."""

import argparse
import os
from functools import partial
import re
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import datasets, transforms

# Stable Diffusion VAE (stabilityai/sd-vae-ft-mse) latent scale.
VAE_SCALE = 0.18215


#################################################################################
#                                     Data                                      #
#################################################################################

def center_crop_arr(pil_image, image_size):
    """center_crop_arr from guided-diffusion (https://github.com/openai/guided-diffusion),
    also used by ADM, DiT and SiT: box-halve
    while the short side is >= 2*image_size, one BICUBIC resize, center crop."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class WnidImageFolder(datasets.ImageFolder):
    """ImageFolder that only treats n######## (WordNet id) folders as classes, so
    stray folders/tars next to the class folders don't shift the label indices."""
    def find_classes(self, directory):
        classes = sorted(e.name for e in os.scandir(directory) if e.is_dir() and re.fullmatch(r'n\d{8}', e.name))
        return classes, {c: i for i, c in enumerate(classes)}


def imagenet_dataset(root, resolution, flip=True):
    """ImageNet train folder -> (tensor in [0, 1], label), ADM preprocessing."""
    tfs = [transforms.Lambda(partial(center_crop_arr, image_size=resolution))]
    if flip:
        tfs.append(transforms.RandomHorizontalFlip())
    tfs.append(transforms.ToTensor())
    return WnidImageFolder(root=root, transform=transforms.Compose(tfs))


@torch.no_grad()
def decode_latents(vae, z):
    """VAE latents -> images in [0, 1]."""
    return torch.clamp((vae.decode(z / VAE_SCALE).sample + 1.0) / 2.0, 0, 1)


#################################################################################
#                                 FID / Inception                               #
#################################################################################

class InceptionFeatures(nn.Module):
    """torchvision Inception-v3 returning (2048-d pool features, class probs).
    Inputs are images in [0, 1]; they are bilinearly resized to 299."""
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.eval()
        self._features = None
        self.model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        logits = self.model(x)
        return self._features.flatten(1), F.softmax(logits, dim=1)


@torch.no_grad()
def inception_features(images, inception, device, batch_size=50):
    """Images (N, 3, H, W) in [0, 1] -> (features (N, 2048), probs (N, 1000)) as numpy."""
    feats, probs = [], []
    for i in range(0, len(images), batch_size):
        f, p = inception(images[i:i + batch_size].detach().to(device).float())
        feats.append(f.cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(feats), np.concatenate(probs)


def feature_stats(feats):
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


@torch.no_grad()
def real_image_stats(dataset, inception, device, num_samples, batch_size=100, num_workers=8, seed=0):
    """FID mean/cov of `num_samples` random images from `dataset`, streamed
    through Inception so the images are never held in memory at once."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx), batch_size=batch_size,
                                         num_workers=num_workers)
    feats = []
    for images, _ in loader:
        feats.append(inception_features(images, inception, device)[0])
    return feature_stats(np.concatenate(feats))


def load_or_compute_real_stats(path, dataset_fn, inception, device, num_samples, **kw):
    """Cached real stats at `path` (npz with mu, sigma); computed and cached on a miss."""
    if os.path.exists(path):
        data = np.load(path)
        print(f"Real FID stats: loaded {path}")
        return data['mu'], data['sigma']
    print(f"Real FID stats: computing from {num_samples} images (cached to {path})...")
    mu, sigma = real_image_stats(dataset_fn(), inception, device, num_samples, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, mu=mu, sigma=sigma)
    os.replace(tmp, path)  # atomic: a preempted job never leaves a partial cache
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two Gaussians."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


#################################################################################
#                                   Config CLI                                  #
#################################################################################

def _str2bool(s):
    if s.lower() in ('1', 'true', 'yes', 'y'):
        return True
    if s.lower() in ('0', 'false', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {s!r}')


def parse_config(config_cls, description=None):
    """Every dataclass field becomes --<field>; flags left unset keep the default."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    hints = typing.get_type_hints(config_cls)
    for name, field in config_cls.__dataclass_fields__.items():
        if not field.init:
            continue
        t = hints[name]
        kwargs = dict(default=None, help=f'(default: {field.default!r})')
        if typing.get_origin(t) is typing.Literal:
            kwargs['choices'] = typing.get_args(t)
        elif t is bool:
            kwargs['type'] = _str2bool
        elif t in (int, float, str):
            kwargs['type'] = t
        else:
            kwargs['type'] = str
        parser.add_argument(f'--{name}', **kwargs)
    overrides = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    return config_cls(**overrides)
