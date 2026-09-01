"""Feature-detected shims over ComfyUI internals and transformers versions.

Every helper here degrades gracefully: nothing in this module may crash when
ComfyUI is absent (plain-library use / unit tests) or when a ComfyUI internal
API moved. Deep integration is attempted only when the exact symbols we use
are present.

Why the paranoia: the two community MOSS-TTS packs broke on new ComfyUI
releases precisely because they hard-called moving internals
(`comfy.ops.cast_bias_weight` positional args, a `CoreModelPatcher` that never
existed on master, `comfy.logging` renamed to `comfy.internal_logging`).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")


# ---------------------------------------------------------------------------
# transformers version
# ---------------------------------------------------------------------------

def transformers_version() -> tuple[int, int]:
    import transformers

    try:
        major, minor, *_ = (int(p) for p in transformers.__version__.split("."))
        return (major, minor)
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# comfy.core imports — all optional
# ---------------------------------------------------------------------------

def _try_import(module: str):
    try:
        import importlib

        return importlib.import_module(module)
    except Exception:
        return None


def comfy_available() -> bool:
    return _try_import("comfy.model_management") is not None


def get_torch_device() -> torch.device:
    mm = _try_import("comfy.model_management")
    if mm is not None and hasattr(mm, "get_torch_device"):
        try:
            return torch.device(mm.get_torch_device())
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_offload_device() -> torch.device:
    mm = _try_import("comfy.model_management")
    if mm is not None and hasattr(mm, "unet_offload_device"):
        try:
            return torch.device(mm.unet_offload_device())
        except Exception:
            pass
    return torch.device("cpu")


def normalize_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(f"cuda:{index}")
    return device


# ---------------------------------------------------------------------------
# memory management integration
# ---------------------------------------------------------------------------

def aimdo_active(device: torch.device) -> bool:
    """True when ComfyUI's runtime has AIMDO/DynamicVRAM actually enabled."""
    if torch.device(device).type == "cpu":
        return False
    mm = _try_import("comfy.memory_management")
    if mm is None or not bool(getattr(mm, "aimdo_enabled", False)):
        return False
    # The dynamic machinery lives in the comfy_aimdo package; require it too.
    for sub in ("control", "host_buffer", "model_vbar"):
        if _try_import(f"comfy_aimdo.{sub}") is None:
            return False
    return True


def model_patcher_class(dynamic: bool):
    """Return the best available ModelPatcher class, or None outside ComfyUI."""
    mp = _try_import("comfy.model_patcher")
    if mp is None or not hasattr(mp, "ModelPatcher"):
        return None
    if dynamic and hasattr(mp, "ModelPatcherDynamic"):
        return mp.ModelPatcherDynamic
    return mp.ModelPatcher


def load_models_gpu(patchers: list[Any]) -> bool:
    """Push new or partially loaded patchers onto the GPU through ComfyUI."""
    mm = _try_import("comfy.model_management")
    if mm is None or not hasattr(mm, "load_models_gpu"):
        return False
    current = getattr(mm, "current_loaded_models", [])
    todo = []
    for patcher in patchers:
        loaded = next((entry for entry in current
                       if getattr(entry, "model", None) is patcher), None)
        if loaded is None:
            todo.append(patcher)
            continue
        try:
            if loaded.model_loaded_memory() < loaded.model_memory():
                todo.append(patcher)
        except Exception:
            todo.append(patcher)
    if todo:
        mm.load_models_gpu(todo)
    return True


def soft_empty_cache() -> None:
    mm = _try_import("comfy.model_management")
    if mm is not None and hasattr(mm, "soft_empty_cache"):
        mm.soft_empty_cache()
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def install_unload_hook(unload_callback: Callable[[str, Any], None]) -> None:
    """Wrap ComfyUI's native unload entry points so our bundle releases too.

    Idempotent. Note current_loaded_models entries hold weakrefs since
    ComfyUI mid-2026 — never keep the loaded objects alive here.
    """
    mm = _try_import("comfy.model_management")
    if mm is None:
        return
    if getattr(mm, "_mosstts_v15_unload_hook_installed", False):
        return

    original_all = getattr(mm, "unload_all_models", None)
    if callable(original_all):

        def unload_all_models_hook(*args, __f=original_all, **kwargs):
            try:
                return __f(*args, **kwargs)
            finally:
                unload_callback("ComfyUI unload_all_models", None)

        mm.unload_all_models = unload_all_models_hook

    original_clones = getattr(mm, "unload_model_and_clones", None)
    if callable(original_clones):

        def unload_model_and_clones_hook(model, *args, __f=original_clones, **kwargs):
            try:
                return __f(model, *args, **kwargs)
            finally:
                unload_callback("ComfyUI unload_model_and_clones", model)

        mm.unload_model_and_clones = unload_model_and_clones_hook

    mm._mosstts_v15_unload_hook_installed = True


# ---------------------------------------------------------------------------
# weight casting for custom nn.Modules (comfy.ops)
# ---------------------------------------------------------------------------

def cast_weight_context(module: torch.nn.Module, input: Any = None, *,
                        device: Any = None, dtype: Any = None, offloadable: bool = True):
    """Return a context manager yielding ``(weight, bias, offload_stream)``.

    Call convention mirrors comfy's own ops: float ops (Linear/Conv/LayerNorm)
    pass ``input`` positionally so the cast derives dtype/device from it;
    integer-input ops (Embedding) must instead pass explicit ``device=`` with
    ``dtype=None`` — casting an embedding table to the ids' int64 dtype would
    corrupt it (this choice decides correctness, not performance).

    Prefers `comfy.ops.CastBiasWeightContext` (present since 2026-08; natively
    yields a 2-tuple, so we adapt it). Falls back to cast_bias_weight/
    uncast_bias_weight with keyword args only. Outside ComfyUI, degrades to a
    plain ``(weight, bias)`` pass-through.
    """
    ops = _try_import("comfy.ops")
    if ops is not None and hasattr(ops, "CastBiasWeightContext"):
        return _ModernCastContext(ops, module, input, device=device, dtype=dtype,
                                  offloadable=offloadable)
    return _LegacyCastContext(ops, module, input, offloadable, device=device, dtype=dtype)


class _ModernCastContext:
    """Adapt comfy.ops.CastBiasWeightContext (yields 2-tuple) to 3-tuple shape."""

    def __init__(self, ops, module, input, *, device=None, dtype=None, offloadable=True):
        kwargs = {"offloadable": offloadable}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None or input is None:
            kwargs["dtype"] = dtype
        self._ctx = ops.CastBiasWeightContext(module, *([input] if input is not None else []),
                                              **kwargs)

    def __enter__(self):
        weight, bias = self._ctx.__enter__()
        return weight, bias, None

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


class _LegacyCastContext:
    def __init__(self, ops, module, input, offloadable, device=None, dtype=None):
        self._ops = ops
        self._m = module
        self._input = input
        self._offloadable = offloadable
        self._device = device
        self._dtype = dtype
        self._weight = None
        self._bias = None
        self._stream = None

    def __enter__(self):
        m = self._m
        if self._ops is None:
            return m.weight, getattr(m, "bias", None), None
        kwargs = {"input": self._input, "offloadable": self._offloadable}
        if self._device is not None:
            kwargs["device"] = self._device
        if self._dtype is not None or self._input is None:
            kwargs["dtype"] = self._dtype
        result = self._ops.cast_bias_weight(m, **kwargs)
        if len(result) == 3:
            self._weight, self._bias, self._stream = result
        else:
            self._weight, self._bias = result
            self._stream = None
        return self._weight, self._bias, self._stream

    def __exit__(self, *exc):
        if self._ops is not None:
            self._ops.uncast_bias_weight(self._m, self._weight, self._bias, self._stream)
        return False


# ---------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Wrap comfy.utils.ProgressBar when importable; silent otherwise."""

    def __init__(self, total: int):
        total = max(1, int(total))
        utils = _try_import("comfy.utils")
        self._bar = utils.ProgressBar(total) if utils is not None and hasattr(utils, "ProgressBar") else None
        self._total = total

    def __call__(self, current: int, total: int) -> None:
        if self._bar is not None:
            self._bar.update_absolute(max(0, int(current)), max(1, int(total)))


# ---------------------------------------------------------------------------
# model folder discovery
# ---------------------------------------------------------------------------

def comfy_models_dir() -> Optional[str]:
    folder_paths = _try_import("folder_paths")
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        try:
            return str(folder_paths.models_dir)
        except Exception:
            pass
    return None


def register_model_folder(name: str, path: str) -> None:
    folder_paths = _try_import("folder_paths")
    if folder_paths is None:
        return
    try:
        table = folder_paths.folder_names_and_paths
        if name in table:
            paths, exts = table[name]
            if path not in paths:
                paths.append(path)
        else:
            table[name] = ([path], {".safetensors", ".json", ".txt", ".jinja"})
    except Exception as exc:  # pragma: no cover - purely additive nicety
        logger.debug("register_model_folder(%s) skipped: %s", name, exc)
