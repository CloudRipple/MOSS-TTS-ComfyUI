"""MOSS-TTS bundle lifecycle: model discovery, loading, and ComfyUI/AIMDO
memory-management integration.

Deep integration is *additive*: every ComfyUI touchpoint goes through
compat.py feature detection. Without ComfyUI (unit tests, plain scripts) the
bundle simply loads to the target device directly.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from . import compat, native
from .native import VARIANTS, VariantSpec

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

DTYPE_OPTIONS = ["auto", "bf16", "fp16", "fp32"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention_2", "eager"]
MODEL_FOLDER_NAME = "mosstts"

_ACTIVE_BUNDLE: Optional["MossTTSBundle"] = None
_ACTIVE_LOAD_KEY: Optional[tuple] = None
_BUNDLE_GENERATION = 0


@dataclass
class MossTTSBundle:
    spec: VariantSpec
    model: Any = field(repr=False)
    processor: Any = field(repr=False)
    codec: Any = field(repr=False)
    model_dir: Path
    codec_dir: Path
    device: torch.device
    torch_dtype: torch.dtype
    dtype_name: str
    attn_implementation: str
    patchers: list[Any] = field(default_factory=list, repr=False)

    @property
    def sample_rate(self) -> int:
        return self.spec.sample_rate


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------

def _models_root_candidates() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("MOSS_TTS_MODELS_DIR")
    if env_root:
        roots.append(Path(env_root))
    comfy_root = compat.comfy_models_dir()
    if comfy_root:
        roots.append(Path(comfy_root) / MODEL_FOLDER_NAME)
    return roots


def _resolve_dir_for(repo_id: str, download_if_missing: bool) -> Path:
    """Resolve a HF repo to a local directory without copying anything.

    Order: $MOSS_TTS_MODELS_DIR/<Repo-Name> → ComfyUI models/mosstts/<Repo-Name>
    → HF hub cache snapshot (respecting HF_HOME) → (optionally) download.
    """
    repo_name = repo_id.split("/")[-1]
    for root in _models_root_candidates():
        candidate = root / repo_name
        if (candidate / "config.json").is_file():
            return candidate
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, local_files_only=True)
        return Path(path)
    except Exception:
        pass
    if not download_if_missing:
        raise FileNotFoundError(
            f"{repo_id} not found locally. Searched: "
            f"{[str(r / repo_name) for r in _models_root_candidates()]} + HF cache. "
            "Enable download_if_missing or place the model files accordingly."
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id))


def resolve_model_dirs(spec: VariantSpec, download_if_missing: bool) -> tuple[Path, Path]:
    return (
        _resolve_dir_for(spec.repo_id, download_if_missing),
        _resolve_dir_for(spec.codec_repo_id, download_if_missing),
    )


# ---------------------------------------------------------------------------
# dtype / attention resolution
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        dtype = torch.bfloat16  # MOSS training precision on both variants
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    if device.type == "cpu":
        if dtype in (torch.float16, torch.bfloat16):
            logger.warning("Half precision on CPU is poorly supported; using fp32.")
        return torch.float32
    return dtype


def _flash_attn_usable(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return False
    import importlib.util

    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 8
    except Exception:
        return True


def resolve_attention(attention: str, device: torch.device, dtype: torch.dtype) -> str:
    """The vendored code defaults to flash_attention_2, which hard-crashes
    without the flash_attn package — 'auto' never does that."""
    if attention == "auto":
        if _flash_attn_usable(device, dtype):
            return "flash_attention_2"
        return "sdpa" if device.type == "cuda" else "eager"
    if attention == "flash_attention_2" and not _flash_attn_usable(device, dtype):
        logger.warning("flash_attention_2 requested but unusable; using sdpa instead.")
        return "sdpa"
    return attention


def _configure_torch_attention(device: torch.device) -> None:
    """cuDNN SDPA is broken for some CUDA/torch combos (per MOSS README);
    keep flash/mem-efficient/math as fallbacks."""
    if torch.device(device).type != "cuda":
        return
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ComfyUI patcher registration
# ---------------------------------------------------------------------------

def _ensure_writable_device_property(module: torch.nn.Module) -> None:
    """HF PreTrainedModel exposes a read-only `device` property; ComfyUI
    machinery occasionally assigns to it. Install a writable override via an
    instance-local subclass."""
    cls = module.__class__
    prop = getattr(cls, "device", None)
    if not isinstance(prop, property) or prop.fset is not None:
        return
    if getattr(module, "_mosstts_writable_device_property", False):
        return

    def _get(self):
        return self.__dict__.get("_mosstts_runtime_device") or prop.fget.__get__(self, cls)()

    def _set(self, value):
        self.__dict__["_mosstts_runtime_device"] = torch.device(value)

    module.__class__ = type(
        f"{cls.__name__}ComfyWritableDevice",
        (cls,),
        {"device": property(_get, _set)},
    )
    module._mosstts_writable_device_property = True


def _register_with_comfy(module: torch.nn.Module, device: torch.device,
                         dynamic: bool, *, load: bool = True) -> Optional[Any]:
    """Register module under a ModelPatcher and onload it through ComfyUI.
    Returns the patcher, or None when ComfyUI integration is unavailable."""
    _ensure_writable_device_property(module)
    patcher_cls = compat.model_patcher_class(dynamic=dynamic)
    if patcher_cls is None or device.type == "cpu":
        module.to(device)
        return None
    size = native.estimate_module_bytes(module)
    module.model_loaded_weight_memory = 0
    patcher = patcher_cls(
        module,
        load_device=device,
        offload_device=compat.get_offload_device(),
        size=size,
    )
    if not patcher.is_dynamic():
        try:
            module.device = device
        except Exception:
            pass
    # TTS and Codec can exceed the available VRAM when resident together.
    # Register a patcher without loading it so runtime operations can resume
    # only the component they currently need.
    if not load:
        return patcher
    if compat.load_models_gpu([patcher]):
        logger.info(
            "Registered %s with ComfyUI%s memory management.",
            module.__class__.__name__, "/AIMDO" if patcher.is_dynamic() else "",
        )
    else:
        module.to(device)
        return None
    return patcher


# ---------------------------------------------------------------------------
# Bundle load / unload
# ---------------------------------------------------------------------------

def _bundle_load_key(spec: VariantSpec, model_dir: Path, codec_dir: Path,
                     device: torch.device, dtype: torch.dtype, attn: str) -> tuple:
    files = native._iter_safetensors_entries(model_dir) + native._iter_safetensors_entries(codec_dir)
    stat_bits: list[Any] = []
    for path, _ in files:
        stat = path.stat()
        stat_bits.extend([str(path), stat.st_size, stat.st_mtime_ns])
    return (spec.key, str(stat_bits), str(device), str(dtype), attn)


def load_mosstts_bundle(
    variant: str,
    dtype_name: str = "auto",
    attention: str = "auto",
    download_if_missing: bool = True,
    load_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MossTTSBundle:
    """Load (or reuse) a MOSS-TTS bundle."""
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    model_dir, codec_dir = resolve_model_dirs(spec, download_if_missing)
    device = compat.normalize_device(compat.get_torch_device())
    dtype = resolve_dtype(dtype_name, device)
    attn = resolve_attention(attention, device, dtype)
    _configure_torch_attention(device)

    load_key = _bundle_load_key(spec, model_dir, codec_dir, device, dtype, attn)
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        # Runtime operations resume only the component they need. Eagerly
        # restoring both here would defeat the one-component VRAM policy.
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_mosstts_bundle(_ACTIVE_BUNDLE, reason="variant/dtype/attention/weights changed")

    logger.info(
        "Loading %s from %s (codec %s) on %s dtype=%s attention=%s",
        spec.repo_id, model_dir, codec_dir, device, dtype, attn,
    )

    # Weights stream to CPU; the patcher owns GPU residency afterwards.
    weight_device = torch.device("cpu") if device.type != "cpu" else device
    tts_config = native.read_tts_config(spec, model_dir)
    codec_config = native.read_codec_config(spec, codec_dir)
    model = native.build_tts_model(spec, tts_config, attn)
    codec = native.build_codec_model(spec, codec_config, dtype, attn)
    codec_dtype = torch.float32 if spec.codec_asset_pkg == "moss_audio_tokenizer_v1" else dtype
    native.load_tts_weights(spec, model, model_dir, dtype, weight_device, load_progress_callback)
    native.load_codec_weights(spec, codec, codec_dir, dtype, weight_device, load_progress_callback)
    tokenizer = native.build_tokenizer(model_dir)
    processor = native.build_processor(spec, tokenizer, codec, tts_config)

    # Note: we intentionally run through a plain (static) ModelPatcher even
    # when AIMDO/DynamicVRAM is active. VBAR-style weight paging conflicts with
    # the vendored codec's device sniffing (`next(parameters()).device` reads
    # "cpu" while VBAR pages weights to GPU), producing cpu-vs-cuda mismatches
    # inside the conv stack. Static patching stays correct; weights persist on
    # the GPU and ComfyUI can still unload them through the patcher.
    dynamic = False
    native.convert_modules_for_comfy(model)
    native.convert_modules_for_comfy(codec)
    native.set_runtime_dtype(model, dtype)
    native.set_runtime_dtype(codec, codec_dtype)

    patchers: list[Any] = []
    tts_patcher = _register_with_comfy(model, device, dynamic)
    if tts_patcher is not None:
        patchers.append(tts_patcher)
    # Keep Codec off GPU until an encode/decode operation needs it.
    codec_patcher = _register_with_comfy(codec, device, dynamic, load=False)
    if codec_patcher is not None:
        patchers.append(codec_patcher)

    bundle = MossTTSBundle(
        spec=spec, model=model, processor=processor, codec=codec,
        model_dir=model_dir, codec_dir=codec_dir,
        device=device, torch_dtype=dtype, dtype_name=dtype_name,
        attn_implementation=attn, patchers=patchers,
    )
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    compat.install_unload_hook(_on_comfy_unload)
    return bundle


def _resume_module(bundle: MossTTSBundle, module: torch.nn.Module) -> None:
    """Load one bundle component, allowing ComfyUI to offload other patchers."""
    for patcher in bundle.patchers:
        if getattr(patcher, "model", None) is module:
            compat.load_models_gpu([patcher])
            return
    module.to(bundle.device)


def resume_model(bundle: MossTTSBundle) -> None:
    _resume_module(bundle, bundle.model)


def resume_codec(bundle: MossTTSBundle) -> None:
    _resume_module(bundle, bundle.codec)


def resume_bundle(bundle: MossTTSBundle) -> None:
    """Compatibility helper: resume both components when explicitly requested."""
    resume_model(bundle)
    resume_codec(bundle)


def unload_mosstts_bundle(bundle: Optional[MossTTSBundle], reason: str = "requested") -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading MOSS-TTS bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        try:
            patcher.detach()
        except Exception:
            pass
    bundle.patchers.clear()
    for module in (bundle.model, bundle.codec):
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            if hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
            else:
                module.to("cpu")
        except Exception:
            pass
    bundle.model = None
    bundle.codec = None
    if getattr(bundle.processor, "audio_tokenizer", None) is not None:
        bundle.processor.audio_tokenizer = None
    gc.collect()
    compat.soft_empty_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def _on_comfy_unload(reason: str) -> None:
    global _BUNDLE_GENERATION
    _BUNDLE_GENERATION += 1
    unload_mosstts_bundle(_ACTIVE_BUNDLE, reason=reason)


def bundle_generation() -> int:
    return _BUNDLE_GENERATION


def active_bundle() -> Optional[MossTTSBundle]:
    return _ACTIVE_BUNDLE
