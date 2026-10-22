# Scaling Transformer Energy-Based Models for High Quality Image Synthesis

**Jungsoo Lee<sup>1</sup>, Alexi Gladstone<sup>2</sup>, Yilun Du<sup>3</sup>**
<sup>1</sup>MIT &nbsp; <sup>2</sup>UIUC &nbsp; <sup>3</sup>Harvard University

[[Paper]](TODO) &nbsp; [[Project page]] https://int-lee.github.io/scalable-ebm/

<!-- <p align="center"><img src="assets/teaser.png" width="90%"></p> -->

This repository contains the PyTorch implementation of the **Transformer Energy-Based Model (TEM)**. TEM is a contrastively trained energy-based model (EBM) that generates high-fidelity ImageNet images using Langevin MCMC without classifier-free guidance.



## Results

Class-conditional ImageNet, FID-50K, **no classifier-free guidance**:

| Model | Params | Resolution | Sampling steps | FID ↓ |
|---|---|---|---|---|
| TEM-L/2  | 415M | 256×256 | 60 | 4.74 |
| TEM-XL/2 | 609M | 256×256 | 60 | 3.82 |
| TEM-XL/2 | 609M | 512×512 | 60 | 8.24 |

Inference-time scaling for TEM-XL/2 at 256×256 (step size 500):

| Sampling steps | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|
| 40 | 7.63 | 106.0 | 0.65 | 0.65 |
| 60 | 3.82 | 176.2 | 0.76 | 0.57 |
| 80 | 4.72 | 221.1 | 0.81 | 0.51 |

Pretrained checkpoints: TODO.

## Installation

```bash
git clone TODO && cd TransformerEBM
conda env create -f environment.yml && conda activate tem
# or, in an existing Python 3.10 environment (install the torch build for your CUDA first):
pip install -r requirements.txt
```

These pretrained networks are downloaded automatically on first use; set `HF_HOME` / `TORCH_HOME` to choose the cache location:

| Network | Used for |
|---|---|
| [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) | the latent space (4×32×32 at 256px, 4×64×64 at 512px) |
| [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large) | the PRA target (training only) |
| torchvision Inception-v3 | FID monitor during training |


## Data

Point `--data_path` at the ImageNet-1k **train** directory: one sub-folder per class (`n01440764/`, ...), as for `torchvision.datasets.ImageFolder`. Only folders named `n########` are treated as classes.

Images are preprocessed as in ADM/DiT: center crop with box-downsampling and a bicubic resize, then a random horizontal flip.

## Training

Training uses 4 GPUs with the paper's defaults: batch 256 (8 per GPU × 8 accumulation steps × 4 GPUs), Adam with learning rate 2e-4 and betas (0, 0.999), 40 Langevin steps (step size 500, noise 0.05), and EMA decay 0.9999.

```bash
# TEM-XL/2, ImageNet 256x256
torchrun --nproc_per_node=4 --standalone train.py \
    --data_path /path/to/imagenet/train --model_size XL --resolution 256

# TEM-XL/2, ImageNet 512x512
torchrun --nproc_per_node=4 --standalone train.py \
    --data_path /path/to/imagenet/train --model_size XL --resolution 512

# other sizes: --model_size S | B | L
```

Every option is a command-line flag (`python train.py -h`):



**Steps.** The step counter (checkpoint names, `--max_steps`, logging intervals) counts micro-batches. With the defaults, one micro-batch is 32 images across 4 GPUs, so the paper's 50K iterations at batch 256 correspond to `--max_steps 400000`.

**Outputs.** Each run writes to `<output_dir>/<experiment_name>/` (default `./checkpoints/TEM-<size>-2_in<res>/`):
- `model_step_N.pth`: model, EMA, optimizer and replay buffer.
- `fid_scores.txt`: a quick FID monitor during training (1k samples). Use `inference.py` and the ADM evaluator for real evaluations.

Checkpoints are written atomically. A restarted job resumes automatically from the newest complete checkpoint, including the data order, so preemptible cluster queues are safe to use.

**Compute.** Training TEM-XL/2 at 256×256 for 50K iterations takes about 6 days on 4 H200 GPUs. Training at 512×512 is about 3–4× slower.

## Sampling and evaluation

`inference.py` follows the paper's persistent-chain protocol:
1. The replay buffer starts empty, with a capacity of 5,000.
2. While it fills, chains start from 𝒩(0, I); afterwards they start from buffered states.
3. Each chain runs `--total_langevin_steps` steps (default 60). Its state after step 40 goes into the buffer, so the buffer stays matched to the training horizon.
4. Every chain gets an independently sampled class label.


```bash
# 1. Generate 50K samples -> samples/<run>_<step>_s60_n50000.npz
torchrun --nproc_per_node=4 --standalone inference.py \
    --checkpoint /path/to/model_step_N.pth --model_size XL --resolution 256
```

**Inference-time scaling.** No retraining is needed: increase `--total_langevin_steps` (e.g. 40 → 60 → 80) or `--step_size` (e.g. 500 → 1000) to move along the fidelity–diversity curve.


## Citation

```bibtex
@article{lee2026tem,
  title   = {Scaling Transformer Energy-Based Models for High Quality Image Synthesis},
  author  = {Lee, Jungsoo and Gladstone, Alexi and Du, Yilun},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```
