"""
TEM: Transformer Energy Model.

A class-conditional ViT that maps a VAE latent x (B, 4, H, W) and a class label
y (B,) to a vector of scalar energies (B, num_energy_heads), one per energy head.
The total energy is energies.sum(dim=1).

Backbone (per block): RMSNorm, attention with QK-RMSNorm and 2D RoPE, SwiGLU MLP,
and adaLN conditioning on the class embedding (scale + gate, no shift). A learned
position embedding (initialized to 2D sin-cos) is added to the patchified latents.

Energy heads: mean-pool over tokens -> LayerNorm -> MLP -> scalar, placed at 1/3,
2/3 and 3/3 of the depth (see default_head_layers). One head always follows the
last block; num_energy_heads=2 adds the 2/3 head, 3 also the 1/3 head.

PRA (Positive Representation Alignment): forward(x, y, pra_layer=k) also returns
the tokens after block k passed through `projector` (a 3x3 conv over the token
grid), to be aligned with DINOv2 patch features.

References for the components below:
  - DiT (Peebles & Xie, 2023), https://github.com/facebookresearch/DiT:
    patch embedding, label embedding, adaLN-Zero conditioning, sin-cos position
    embedding and weight initialization.
  - LightningDiT (Yao et al., 2025), https://github.com/hustvl/LightningDiT:
    the block layout (RMSNorm, QK-norm, 2D RoPE, SwiGLU, shift-free adaLN).
  - EVA-02 (Fang et al., 2023), https://github.com/baaivision/EVA, based on
    lucidrains/rotary-embedding-torch: 2D rotary position embedding.
  - Singh et al. (2025), "What matters for representation alignment",
    https://arxiv.org/abs/2512.10794: convolutional PRA projector.
"""

import numpy as np
import torch
import torch._functorch.config
import torch.nn as nn
import torch.nn.functional as F

# The sampler differentiates through a torch.compile'd Langevin loop and the KL
# term takes a second-order gradient; donated buffers break that, so turn them off.
torch._functorch.config.donated_buffer = False


TEM_CONFIGS = {
    'S':  dict(depth=12, hidden_size=384,  num_heads=6),
    'B':  dict(depth=12, hidden_size=768,  num_heads=12),
    'L':  dict(depth=24, hidden_size=1024, num_heads=16),
    'XL': dict(depth=28, hidden_size=1152, num_heads=16),
}


def default_head_layers(depth):
    """0-indexed blocks after which the early / middle energy heads read the
    tokens: the ends of the first and second thirds of the network
    (S/B: 3, 7  L: 7, 15  XL: 8, 17). The final head follows block depth-1."""
    return tuple((j * depth) // 3 - 1 for j in (1, 2))


#################################################################################
#                                  Building blocks                              #
#################################################################################

def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, N, D) with a strided conv."""
    def __init__(self, input_size, patch_size, in_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


def rotate_half(x):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbedding(nn.Module):
    """2D RoPE (EVA-02 style): half of each head's channels rotate with the row
    index, the other half with the column index."""
    def __init__(self, dim, seq_len, theta=10000):
        super().__init__()
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(seq_len).float()
        freqs = torch.einsum('i,f->if', t, freqs).repeat_interleave(2, dim=-1)  # (L, dim)
        freqs = torch.cat([freqs[:, None, :].expand(seq_len, seq_len, dim),
                           freqs[None, :, :].expand(seq_len, seq_len, dim)], dim=-1)
        freqs = freqs.reshape(-1, 2 * dim)
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TEMBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * int(hidden_size * mlp_ratio)))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True),
        )

    def forward(self, x, c, rope):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), scale_msa), rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp))
        return x


class EnergyHead(nn.Module):
    """Scalar energy from the mean of all patch tokens."""
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size // 2, bias=True)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size // 2, 1, bias=True)

    def forward(self, x):
        x = self.norm_final(x).mean(dim=1)
        return self.linear2(self.act(self.linear(x)))


#################################################################################
#                                       TEM                                     #
#################################################################################

class TEM(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4,
                 hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0,
                 num_classes=1000, num_energy_heads=3, head_layers=None,
                 pra_dim=1024):
        super().__init__()
        assert 1 <= num_energy_heads <= 3, num_energy_heads
        if head_layers is None:
            head_layers = default_head_layers(depth)
        assert all(0 <= l < depth for l in head_layers), (head_layers, depth)
        self.in_channels = in_channels
        self.depth = depth
        self.num_energy_heads = num_energy_heads
        self.early_head_layer = head_layers[0] if num_energy_heads >= 3 else None
        self.middle_head_layer = head_layers[1] if num_energy_heads >= 2 else None
        self.head_layers = tuple(head_layers)

        # Sin-cos initialized, then learned.
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.feat_rope = VisionRotaryEmbedding(dim=hidden_size // num_heads // 2,
                                               seq_len=input_size // patch_size)
        self.blocks = nn.ModuleList([TEMBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.initialize_weights()

        # Created after initialize_weights, so they keep PyTorch's default init.
        # PRA projector: a 3x3 conv over the token grid.
        self.projector = nn.Conv2d(hidden_size, pra_dim, kernel_size=3, padding=1)
        self.final_layer = EnergyHead(hidden_size)
        if self.middle_head_layer is not None:
            self.middle_layer = EnergyHead(hidden_size)
        if self.early_head_layer is not None:
            self.early_layer = EnergyHead(hidden_size)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y, pra_layer=None):
        """Returns energies (B, num_energy_heads); with pra_layer=k also returns
        the projected tokens after block k, shape (B, N, pra_dim)."""
        x = self.x_embedder(x) + self.pos_embed
        c = self.y_embedder(y)

        energies = []
        pra_act = None
        for i, block in enumerate(self.blocks):
            x = block(x, c, self.feat_rope)

            if i == self.early_head_layer:
                energies.append(self.early_layer(x))
            if i == self.middle_head_layer:
                energies.append(self.middle_layer(x))

            if i == pra_layer:
                B, N, C = x.shape
                H = W = int(N ** 0.5)
                grid = x.view(B, H, W, C).permute(0, 3, 1, 2)
                pra_act = self.projector(grid).permute(0, 2, 3, 1).reshape(B, N, -1)

        energies.append(self.final_layer(x))
        energy = torch.cat(energies, dim=-1)

        if pra_layer is not None:
            return energy, pra_act
        return energy


def build_tem(model_size, input_size, in_channels=4, num_classes=1000, patch_size=2,
              num_energy_heads=3, head_layers=None, pra_dim=1024):
    """TEM-S/B/L/XL. head_layers=None places the heads at 1/3 and 2/3 depth."""
    return TEM(input_size=input_size, patch_size=patch_size, in_channels=in_channels,
               num_classes=num_classes, num_energy_heads=num_energy_heads,
               head_layers=None if head_layers is None else tuple(head_layers),
               pra_dim=pra_dim, **TEM_CONFIGS[model_size])


#################################################################################
#                           Sine/Cosine Positional Embedding                    #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Returns (grid_size*grid_size, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
