# MOSS-TTS-V15-ComfyUI patch notes (vs upstream diffsynth/models/wan_audio_dit.py):
#   * ModelMixin/ConfigMixin dependency on diffusers removed — WanAudioModel is
#     a plain torch.nn.Module and @register_to_config is dropped (the pipeline
#     constructs the model from explicit kwargs read out of transformer/config.json).
#   * DiTBlock and sinusoidal_embedding_1d come from the sibling dit_block.py
#     (extracted from upstream wan_video_dit.py, which is not vendored).
#   * WanModelStateDictConverter + civitai/diffusers hash tables removed; HF
#     (diffusers-named) checkpoints are remapped in pipelines/wan_audio.py
#     (_convert_hf_dit_state_dict) instead.
#   * SimpleAdapter / wan_video_camera_controller import removed — camera
#     control adapters are not part of MOSS-SoundEffect; add_control_adapter=True
#     now raises.
#   * einops.rearrange rewritten as permute/reshape so einops is not required.
#   * Unused upstream leftovers dropped: local modulate(), precompute_freqs_cis_3d.
import torch
import torch.nn as nn
import math
from typing import Literal, Tuple, Optional
from .dit_block import DiTBlock, sinusoidal_embedding_1d


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, s: float = 1.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def legacy_precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0, base_tps=4.0, target_tps=44100/2048):
    s = float(base_tps) / float(target_tps)
    # 1d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
    # No positional encoding applied to the remaining dimensions.
    no_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, s)
    no_freqs_cis = torch.ones_like(no_freqs_cis)
    return f_freqs_cis, no_freqs_cis, no_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            # t_mod is originally [B, C]; broadcasting works for B=1 but not for
            # B>1 against [1, 2, C], so reshape explicitly here.
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanAudioModel(nn.Module):

    # MOSS-TTS-V15-ComfyUI patch: constructor signature is upstream-verbatim
    # (minus @register_to_config); the pipeline passes every field from
    # transformer/config.json explicitly.
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.vae_type = vae_type
        self.patch_embedding = nn.Conv1d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        if vae_type == "oobleck":
            freqs = legacy_precompute_freqs_cis_1d(head_dim, base_tps=4.0, target_tps=44100/2048)
        elif vae_type == "dac":
            freqs = precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")
        # Register RoPE freqs as buffers so model.to(device) / accelerate move
        # them with the model, and so forward / torch.compile graphs do not
        # mutate Python attributes mid-trace.
        self.register_buffer("freqs_cis_0", freqs[0], persistent=False)
        self.register_buffer("freqs_cis_1", freqs[1], persistent=False)
        self.register_buffer("freqs_cis_2", freqs[2], persistent=False)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            # MOSS-TTS-V15-ComfyUI patch: SimpleAdapter (wan_video_camera_controller)
            # is not vendored; the sound-effect model never uses it.
            raise NotImplementedError(
                "add_control_adapter requires the camera-control adapter, "
                "which is not vendored in this inference-only package."
            )
        self.control_adapter = None

    @property
    def freqs(self):
        # Backwards-compatible accessor: external code can still use self.freqs[i].
        return (self.freqs_cis_0, self.freqs_cis_1, self.freqs_cis_2)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        # MOSS-TTS-V15-ComfyUI patch: control-camera adapter branch removed
        # (control_adapter is always None here).
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = x.permute(0, 2, 1).contiguous()  # 'b c f -> b f c'
        return x, grid_size  # x, grid_size: (f)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        # 'b f (p c) -> b c (f p)' without einops.
        b, f, _ = x.shape
        p = self.patch_size[0]
        x = x.view(b, f, p, -1).permute(0, 3, 1, 2).reshape(b, -1, f * p)
        return x

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, ) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, -1).expand(f, -1),
            self.freqs[1][:f].view(f, -1).expand(f, -1),
            self.freqs[2][:f].view(f, -1).expand(f, -1),
        ], dim=-1).reshape(f, 1, -1)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, ))
        return x
