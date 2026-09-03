# MOSS-TTS-V15-ComfyUI patch: inference-only common DiT pieces extracted from
# upstream diffsynth/models/wan_video_dit.py (the 750-line video WanModel and
# its civitai/diffusers hash tables are NOT vendored). einops.rearrange calls
# were rewritten as plain torch view/transpose so the package works without
# einops installed.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import RMSNorm

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    # MOSS-TTS-V15-ComfyUI patch: rearrange("b s (n d) -> ...") rewritten as
    # view/transpose chains to drop the einops dependency. Per-tensor sequence
    # lengths are kept separate — cross-attention mixes q from the latent (s)
    # with k/v from the text context (s_k), so each tensor views its own shape.
    b, s_q = q.shape[0], q.shape[1]
    s_kv = k.shape[1]
    d = q.shape[-1] // num_heads
    if compatibility_mode:
        q = q.view(b, s_q, num_heads, d).transpose(1, 2)
        k = k.view(b, s_kv, num_heads, d).transpose(1, 2)
        v = v.view(b, s_kv, num_heads, d).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, s_q, num_heads * d)
    elif FLASH_ATTN_3_AVAILABLE:
        q = q.view(b, s_q, num_heads, d)
        k = k.view(b, s_kv, num_heads, d)
        v = v.view(b, s_kv, num_heads, d)

        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = x.reshape(b, s_q, num_heads * d)
    elif FLASH_ATTN_2_AVAILABLE:
        q = q.view(b, s_q, num_heads, d)
        k = k.view(b, s_kv, num_heads, d)
        v = v.view(b, s_kv, num_heads, d)
        x = flash_attn.flash_attn_func(q, k, v)
        x = x.reshape(b, s_q, num_heads * d)
    elif SAGE_ATTN_AVAILABLE:
        q = q.view(b, s_q, num_heads, d).transpose(1, 2)
        k = k.view(b, s_kv, num_heads, d).transpose(1, 2)
        v = v.view(b, s_kv, num_heads, d).transpose(1, 2)
        x = sageattn(q, k, v)
        x = x.transpose(1, 2).reshape(b, s_q, num_heads * d)
    else:
        q = q.view(b, s_q, num_heads, d).transpose(1, 2)
        k = k.view(b, s_kv, num_heads, d).transpose(1, 2)
        v = v.view(b, s_kv, num_heads, d).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, s_q, num_heads * d)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    # Real-valued RoPE: mathematically equivalent to the complex64 / float64
    # formulation, but avoids torch.view_as_complex and float64 casts so the
    # whole forward stays inside a single CUDA Graph under torch.compile.
    out_dtype = x.dtype
    b, s, _ = x.shape
    # MOSS-TTS-V15-ComfyUI patch: einops rearrange -> torch view.
    x = x.view(b, s, num_heads, -1)                       # [b, s, n, d]
    x = x.float().reshape(*x.shape[:-1], -1, 2)           # [b, s, n, d/2, 2]
    x_even, x_odd = x[..., 0], x[..., 1]
    cos = freqs.real.to(x_even.dtype)
    sin = freqs.imag.to(x_even.dtype)
    x_out = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    ).flatten(2)                                            # [b, s, n*d]
    return x_out.to(out_dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x
