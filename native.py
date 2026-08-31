"""Native model construction and weight streaming for MOSS-TTS v1.5.

No trust_remote_code, no HF transformers_modules cache: the remote-code files
are vendored under ``assets/`` (with small compatibility patches, marked
``# MOSS-TTS-V15-ComfyUI patch:``) and imported as synthetic packages.
Models are built on ``torch.device("meta")`` and weights are streamed from
safetensors — loads stay off the GPU until the ComfyUI patcher (or the
fallback path) moves them.

Supports both v1.5 variants:
  * "local"  — MossTTSLocalModel (Qwen3-4B backbone + nano-GPT2 local
    transformer), MOSS-Audio-Tokenizer-v2, 48 kHz stereo, n_vq=12.
  * "delay"  — MossTTSDelayModel (Qwen3 8B, delay-pattern streams),
    MOSS-Audio-Tokenizer (v1), 24 kHz mono, n_vq=32.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from . import compat

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantSpec:
    key: str                         # "local" | "delay"
    label: str                       # UI label
    repo_id: str                     # HF repo holding the TTS weights
    codec_repo_id: str               # HF repo holding the audio codec
    asset_pkg: str                   # dir under assets/
    codec_asset_pkg: str             # dir under assets/
    sample_rate: int
    stereo: bool                     # native decode output channel count
    n_vq: int
    frames_per_second: float = 12.5  # MOSS acoustic frame rate (both variants)


VARIANTS: dict[str, VariantSpec] = {
    "local": VariantSpec(
        key="local",
        label="MOSS-TTS-Local-Transformer-v1.5 (4B, 48kHz stereo)",
        repo_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        codec_repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        asset_pkg="moss_tts_local",
        codec_asset_pkg="moss_audio_tokenizer_v2",
        sample_rate=48000,
        stereo=True,
        n_vq=12,
    ),
    "delay": VariantSpec(
        key="delay",
        label="MOSS-TTS-v1.5 (8B, 24kHz)",
        repo_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        codec_repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        asset_pkg="moss_tts_delay",
        codec_asset_pkg="moss_audio_tokenizer_v1",
        sample_rate=24000,
        stereo=False,
        n_vq=32,
    ),
}

DEFAULT_VARIANT = "local"


# ---------------------------------------------------------------------------
# Vendored asset package registration
# ---------------------------------------------------------------------------

def _assets_root() -> Path:
    return Path(__file__).resolve().parent / "assets"


_SYNTHETIC_PREFIX = "_mossttsv15_"


def _register_asset_package(pkg_dir_name: str) -> None:
    """Expose an assets/<pkg_dir_name> directory as an importable package.

    The vendored files use relative imports (``from .configuration_...``), so
    they must execute as package members. Registered under a synthetic module
    name to avoid colliding with any other custom node's copies.
    """
    pkg_name = _SYNTHETIC_PREFIX + pkg_dir_name
    if pkg_name in sys.modules:
        return
    pkg_dir = _assets_root() / pkg_dir_name
    init_file = pkg_dir / "__init__.py"
    if not pkg_dir.is_dir() or not init_file.is_file():
        raise FileNotFoundError(
            f"Bundled asset package missing: {pkg_dir}. Reinstall or update this node pack."
        )
    spec = importlib.util.spec_from_file_location(
        pkg_name, str(init_file), submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)


def tts_classes(spec: VariantSpec):
    """Return (Config, Model, Processor) classes for the TTS model of `spec`."""
    _register_asset_package(spec.asset_pkg)
    if spec.key == "local":
        from _mossttsv15_moss_tts_local.configuration_moss_tts import MossTTSLocalConfig
        from _mossttsv15_moss_tts_local.modeling_moss_tts import MossTTSLocalModel
        from _mossttsv15_moss_tts_local.processing_moss_tts import MossTTSLocalProcessor

        return MossTTSLocalConfig, MossTTSLocalModel, MossTTSLocalProcessor
    if spec.key == "delay":
        from _mossttsv15_moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig
        from _mossttsv15_moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
        from _mossttsv15_moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor

        return MossTTSDelayConfig, MossTTSDelayModel, MossTTSDelayProcessor
    raise ValueError(f"unknown variant: {spec.key}")


def codec_classes(spec: VariantSpec):
    """Return (Config, Model) classes for the audio codec of `spec`."""
    _register_asset_package(spec.codec_asset_pkg)
    mod_cfg = f"{_SYNTHETIC_PREFIX}{spec.codec_asset_pkg}.configuration_moss_audio_tokenizer"
    mod_mdl = f"{_SYNTHETIC_PREFIX}{spec.codec_asset_pkg}.modeling_moss_audio_tokenizer"
    import importlib

    cfg_mod = importlib.import_module(mod_cfg)
    mdl_mod = importlib.import_module(mod_mdl)
    return cfg_mod.MossAudioTokenizerConfig, mdl_mod.MossAudioTokenizerModel


# ---------------------------------------------------------------------------
# Config reading (no AutoConfig — direct class construction)
# ---------------------------------------------------------------------------

def read_tts_config(spec: VariantSpec, model_dir: Path):
    ConfigCls, _, _ = tts_classes(spec)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    for meta_key in ("auto_map", "architectures"):
        raw.pop(meta_key, None)
    return ConfigCls(**raw)


def read_codec_config(spec: VariantSpec, codec_dir: Path):
    ConfigCls, _ = codec_classes(spec)
    raw = json.loads((codec_dir / "config.json").read_text(encoding="utf-8"))
    for meta_key in ("auto_map", "architectures", "model_type"):
        raw.pop(meta_key, None)
    return ConfigCls(**raw)


# ---------------------------------------------------------------------------
# Meta-device construction
# ---------------------------------------------------------------------------

def build_tts_model(spec: VariantSpec, config, attn_implementation: str):
    _, ModelCls, _ = tts_classes(spec)
    if spec.key == "local":
        config.attn_implementation = attn_implementation
        if hasattr(config, "qwen3_config"):
            config.qwen3_config._attn_implementation = attn_implementation
        # The tiny local transformer does not support "eager" — fall to sdpa.
        config.local_transformer_attn_implementation = (
            "sdpa" if attn_implementation == "eager" else attn_implementation
        )
    else:  # delay — plain Qwen3Model from transformers reads _attn_implementation
        config.language_config._attn_implementation = attn_implementation
    with torch.device("meta"):
        model = ModelCls(config)
    return model


def build_codec_model(
    spec: VariantSpec,
    config,
    runtime_dtype: torch.dtype,
    attn_implementation: str,
):
    _, CodecCls = codec_classes(spec)
    half = runtime_dtype in (torch.bfloat16, torch.float16)
    # The quantizer always runs fp32 (codebook numerics); encoder/decoder halves.
    if hasattr(config, "codec_weight_dtype"):
        config.codec_weight_dtype = "bf16" if half else "fp32"
    if hasattr(config, "compute_dtype"):
        config.compute_dtype = "bf16" if half else "fp32"
    if hasattr(config, "attention_implementation"):
        config.attention_implementation = (
            "sdpa" if attn_implementation == "eager" else attn_implementation
        )
    with torch.device("meta"):
        model = CodecCls(config)
    return model


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _weight_norm_remap(name: str) -> str:
    """Legacy checkpoints store weight_norm as weight_g/weight_v; current
    ``nn.utils.parametrizations.weight_norm`` wants parametrizations.weight.originalN."""
    return name.replace(
        ".weight_g", ".parametrizations.weight.original0"
    ).replace(".weight_v", ".parametrizations.weight.original1")


def _set_parameter(model: nn.Module, name: str, tensor: torch.Tensor,
                   dtype: torch.dtype, device: torch.device) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    current = getattr(parent, leaf)
    if tuple(current.shape) != tuple(tensor.shape):
        tensor = tensor.reshape(current.shape)
    if tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    setattr(parent, leaf, nn.Parameter(tensor.to(device=device).contiguous(), requires_grad=False))


def _materialize_non_persistent_buffers(model: nn.Module, device: torch.device) -> None:
    """Re-create non-persistent buffers left on meta after meta-device construction.

    transformers' Qwen3RotaryEmbedding stores inv_freq AND original_inv_freq as
    nn.Buffer(persistent=False) — both must hang real values, not zeros: a
    zeroed rotary table corrupts attention and the model diverges into NaN.
    Unknown leftover buffers get zeros but are logged loudly so the failure
    mode is diagnosable instead of silently-degraded audio.
    """
    for module in model.modules():
        metas = [(name, buf) for name, buf in module.named_buffers(recurse=False)
                 if buf.device.type == "meta"]
        if not metas:
            continue
        cls_name = type(module).__name__
        if "RotaryEmbedding" in cls_name and hasattr(module, "compute_default_rope_parameters"):
            with torch.no_grad():
                inv_freq, attention_scaling = module.compute_default_rope_parameters(
                    module.config, device
                )
            for name, buf in metas:
                if name == "inv_freq":
                    module.register_buffer(name, inv_freq.to(dtype=buf.dtype), persistent=False)
                elif name == "original_inv_freq":
                    module.register_buffer(name, inv_freq.clone().to(dtype=buf.dtype), persistent=False)
            if hasattr(module, "attention_scaling"):
                module.attention_scaling = attention_scaling
            continue
        for name, buf in metas:
            recomputed = False
            if name == "inv_freq" and hasattr(module, "_compute_inv_freq"):
                try:
                    new_buf = module._compute_inv_freq(device=device)
                    module.register_buffer(name, new_buf.to(device=device, dtype=buf.dtype), persistent=False)
                    recomputed = True
                except Exception:
                    pass
            if not recomputed:
                logger.warning(
                    "Non-persistent buffer %s.%s (class %s) had no recompute "
                    "path; zero-filled. If generation quality regresses, "
                    "inspect this buffer.", cls_name, name, cls_name,
                )
                module.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype, device=device), persistent=False)


def _ensure_no_meta_tensors(model: nn.Module, label: str) -> None:
    meta = [n for n, t in model.state_dict().items() if t.device.type == "meta"]
    meta += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta:
        raise RuntimeError(f"{label} load left meta tensors: {meta[:8]}")


def _iter_safetensors_entries(directory: Path) -> list[tuple[Path, list[str]]]:
    """(shard path, keys) pairs — single file or sharded via index.json."""
    single = directory / "model.safetensors"
    if single.is_file():
        return [(single, [])]  # keys read lazily from the file itself
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json under {directory}"
        )
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    return [(directory / shard, sorted(keys)) for shard, keys in sorted(shards.items())]


def _is_quantizer_key(key: str) -> bool:
    return key.startswith(("quantizer.", "quantizer_"))


def _load_state_stream(
    model: nn.Module,
    entries: list[tuple[Path, list[str]]],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
    progress_callback: Optional[Callable[[int, int], None]],
    skip_keys: set[str] | None = None,
    quantizer_fp32: bool = False,
) -> None:
    """Stream weights key-by-key into a meta-built model."""
    skip_keys = skip_keys or set()
    model_keys = set(model.state_dict().keys())
    unit = "tensor"
    # flatten (path, keys) with lazy key read for single-file case
    tasks: list[tuple[Path, str]] = []
    for path, keys in entries:
        if keys:
            tasks.extend((path, k) for k in keys)
        else:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                tasks.extend((path, k) for k in sorted(handle.keys()))
    pbar = tqdm(total=len(tasks), desc=f"Loading {label} weights", unit=unit,
                dynamic_ncols=True, leave=False) if tqdm is not None else None
    done = 0
    try:
        for path, name in tasks:
            done += 1
            # Legacy checkpoints may store weight_norm as weight_g/weight_v.
            if name not in model_keys:
                name = _weight_norm_remap(name)
            if name in skip_keys or name not in model_keys:
                if pbar is not None:
                    pbar.update(1)
                if progress_callback is not None:
                    progress_callback(done, len(tasks))
                continue
            target_dtype = torch.float32 if (quantizer_fp32 and _is_quantizer_key(name)) else dtype
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
            _set_parameter(model, name, tensor, target_dtype, device)
            if pbar is not None:
                pbar.update(1)
            if progress_callback is not None:
                progress_callback(done, len(tasks))
    finally:
        if pbar is not None:
            pbar.close()


def load_tts_weights(spec: VariantSpec, model: nn.Module, model_dir: Path,
                     dtype: torch.dtype, weight_device: torch.device,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
    entries = _iter_safetensors_entries(model_dir)
    if spec.key == "local":
        # text_lm_head.weight + audio_lm_heads.{i}.weight are tied to the
        # embedding tables; skip loading them, then re-alias via tie_weights()
        # (which only relinks — it never re-initializes the trained
        # local_text_lm_head that the checkpoint provides).
        tied = {"text_lm_head.weight"}
        tied.update(f"audio_lm_heads.{i}.weight" for i in range(int(model.config.n_vq)))
        _load_state_stream(model, entries, dtype, weight_device, "MOSS-TTS",
                           progress_callback, skip_keys=tied)
        model.tie_weights()
    else:  # delay — plain key set, nothing tied
        _load_state_stream(model, entries, dtype, weight_device, "MOSS-TTS",
                           progress_callback)
    _materialize_non_persistent_buffers(model, weight_device)
    _ensure_no_meta_tensors(model, "MOSS-TTS")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_codec_weights(spec: VariantSpec, codec: nn.Module, codec_dir: Path,
                       dtype: torch.dtype, weight_device: torch.device,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
    entries = _iter_safetensors_entries(codec_dir)
    # The v1 codec (delay variant) is an all-fp32 checkpoint with no
    # codec_weight_dtype knob in its code — keep every tensor fp32, exactly as
    # the official from_pretrained path does. The v2 codec (local) is designed
    # for bf16 weights with an fp32 quantizer.
    codec_dtype = torch.float32 if spec.codec_asset_pkg == "moss_audio_tokenizer_v1" else dtype
    _load_state_stream(codec, entries, codec_dtype, weight_device, "MOSS-codec",
                       progress_callback, quantizer_fp32=True)
    _materialize_non_persistent_buffers(codec, weight_device)
    _ensure_no_meta_tensors(codec, "MOSS-Audio-Tokenizer")
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)


# ---------------------------------------------------------------------------
# Tokenizer + Processor construction
# ---------------------------------------------------------------------------

def build_tokenizer(model_dir: Path):
    """Load the Qwen-based tokenizer straight from the (local) model dir —
    also picks up the bundled chat_template.jinja, required by the delay path."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


def build_processor(spec: VariantSpec, tokenizer, codec, config):
    _, _, ProcessorCls = tts_classes(spec)
    return ProcessorCls(tokenizer=tokenizer, audio_tokenizer=codec, model_config=config)


# ---------------------------------------------------------------------------
# Comfy-castable module conversion
# ---------------------------------------------------------------------------

def estimate_module_bytes(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for value in list(module.parameters()) + list(module.buffers()):
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        total += value.numel() * value.element_size()
    return total


def set_runtime_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Tag float params/buffers with ``<name>_comfy_model_dtype`` so ComfyUI's
    lowvram/AIMDO cast machinery knows the compute dtype (quantizer stays fp32)."""
    for module_name, sub in module.named_modules():
        target = torch.float32 if module_name == "quantizer" or module_name.startswith("quantizer.") else dtype
        for name, value in sub.named_parameters(recurse=False):
            if value.is_floating_point():
                setattr(sub, f"{name}_comfy_model_dtype", target)
        for name, value in sub.named_buffers(recurse=False):
            if value.is_floating_point():
                setattr(sub, f"{name}_comfy_model_dtype", target)


class _CastWeightAttrs:
    """comfy.ops.cast_bias_weight reads s.weight_function / s.bias_function /
    s.bias directly — mirror what comfy's CastWeightBiasOp base provides."""

    comfy_cast_weights = True
    weight_function: list = []
    bias_function: list = []


class _ComfyLinear(nn.Linear, _CastWeightAttrs):
    def forward(self, x):
        with compat.cast_weight_context(self, x) as (weight, bias, _stream):
            return torch.nn.functional.linear(x, weight, bias)


class _ComfyEmbedding(nn.Embedding, _CastWeightAttrs):
    @property
    def bias(self):  # nn.Embedding has no bias; cast_bias_weight reads it
        return None

    def forward(self, x):
        # Mirror comfy.ops Embedding: the ids are integers — never let the cast
        # derive dtype from them. Pass device explicitly and keep the table's
        # own dtype.
        with compat.cast_weight_context(self, device=x.device, dtype=None) as (weight, _bias, _stream):
            return torch.nn.functional.embedding(
                x, weight,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
            )


class _ComfyLayerNorm(nn.LayerNorm, _CastWeightAttrs):
    def forward(self, x):
        with compat.cast_weight_context(self, x) as (weight, bias, _stream):
            return torch.nn.functional.layer_norm(
                x, self.normalized_shape, weight, bias, self.eps
            )


class _ComfyConv1d(nn.Conv1d, _CastWeightAttrs):
    def forward(self, x):
        with compat.cast_weight_context(self, x) as (weight, bias, _stream):
            return self._conv_forward(x, weight, bias)


class _ComfyConvTranspose1d(nn.ConvTranspose1d, _CastWeightAttrs):
    # Mirror comfy.ops ConvTranspose1d.forward_comfy_cast_weights: compute
    # output_padding the same way torch's own forward does, then run the
    # functional op against the cast weight/bias.
    def forward(self, x, output_size=None):
        num_spatial_dims = 1
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,
            num_spatial_dims, self.dilation,
        )
        with compat.cast_weight_context(self, x) as (weight, bias, _stream):
            return torch.nn.functional.conv_transpose1d(
                x, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation,
            )


_CASTABLE: tuple[tuple[type, type], ...] = (
    (nn.Linear, _ComfyLinear),
    (nn.Embedding, _ComfyEmbedding),
    (nn.LayerNorm, _ComfyLayerNorm),
    (nn.Conv1d, _ComfyConv1d),
    (nn.ConvTranspose1d, _ComfyConvTranspose1d),
)


def convert_modules_for_comfy(model: nn.Module) -> None:
    """Swap nn module classes in place for ComfyUI weight-cast integration.

    ModelPatcher(Dynamic) only streams weights through VBAR/cast machinery for
    modules exposing ``comfy_cast_weights``. Skip conversion silently when the
    comfy.ops API is entirely absent (plain-library mode).
    """
    if compat._try_import("comfy.ops") is None:
        return
    for module in model.modules():
        for base, castable in _CASTABLE:
            if type(module) is base:
                module.__class__ = castable
                break
