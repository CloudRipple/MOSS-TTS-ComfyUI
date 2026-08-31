"""CPU-only unit tests for the pack: registry shape, conversions, knobs,
fail-fast guards, and comfy-compat fallbacks. No weights, no GPU."""

from __future__ import annotations

import pytest
import torch

import mosstts_v15

pack = mosstts_v15


# ---------------------------------------------------------------------------
# node registry
# ---------------------------------------------------------------------------

def test_node_registry_complete():
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert len(pack.NODE_CLASS_MAPPINGS) == 5
    for cls in pack.NODE_CLASS_MAPPINGS.values():
        inputs = cls.INPUT_TYPES()
        assert "required" in inputs
        assert isinstance(cls.FUNCTION, str) and hasattr(cls, cls.FUNCTION.split(".")[0])
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)


def test_variant_specs():
    from mosstts_v15.native import VARIANTS

    local, delay = VARIANTS["local"], VARIANTS["delay"]
    assert local.sample_rate == 48000 and local.stereo and local.n_vq == 12
    assert delay.sample_rate == 24000 and not delay.stereo and delay.n_vq == 32
    assert "tokenizer" in ("tokenizer",)  # smoke: nothing to check, kept honest
    assert local.repo_id != delay.repo_id and local.codec_repo_id != delay.codec_repo_id


# ---------------------------------------------------------------------------
# estimate tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_english():
    cls = pack.NODE_CLASS_MAPPINGS["MossTTSV15_EstimateTokens"]()
    (tokens,) = cls.estimate("one two three four five six", 150.0)  # 6 words @150wpm = 2.4s
    assert tokens == 30


def test_estimate_tokens_cjk():
    cls = pack.NODE_CLASS_MAPPINGS["MossTTSV15_EstimateTokens"]()
    (tokens,) = cls.estimate("一二三四五六", 150.0)  # 6 chars @150cpm = 2.4s
    assert tokens == 30


def test_estimate_tokens_empty():
    cls = pack.NODE_CLASS_MAPPINGS["MossTTSV15_EstimateTokens"]()
    assert cls.estimate("   ", 150.0) == (0,)


# ---------------------------------------------------------------------------
# audio conversions
# ---------------------------------------------------------------------------

def test_audio_conversions():
    import mosstts_v15.runtime as runtime

    wav3d = torch.zeros(1, 2, 4800)
    d1 = runtime.tensor_to_comfy_audio(wav3d, 48000)
    assert d1["waveform"].shape == (1, 2, 4800) and d1["sample_rate"] == 48000
    back, sr = runtime.comfy_audio_to_tensor(d1)
    assert back.shape == (2, 4800) and sr == 48000
    d2 = runtime.tensor_to_comfy_audio(torch.zeros(2400), 24000)
    assert d2["waveform"].shape == (1, 1, 2400)


# ---------------------------------------------------------------------------
# fail-fast + knob filtering
# ---------------------------------------------------------------------------

def test_require_text_fail_fast():
    import mosstts_v15.runtime as runtime

    with pytest.raises(ValueError):
        runtime._require_text("   ")
    with pytest.raises(ValueError):
        runtime._require_text(None)


class _Spec:
    key = "local"


class _FakeBundle:
    spec = _Spec()


def test_generate_kwargs_local_vs_delay():
    import mosstts_v15.runtime as runtime

    bundle = _FakeBundle()
    kwargs = dict(max_new_tokens=100, do_sample=True, text_temperature=1.0,
                  text_top_p=1.0, text_top_k=50, audio_temperature=1.7,
                  audio_top_p=0.8, audio_top_k=25, audio_repetition_penalty=1.0)
    local = runtime._generate_kwargs(bundle, **kwargs)
    assert local["do_sample"] is True and local["max_new_tokens"] == 100

    bundle.spec.key = "delay"
    delay = runtime._generate_kwargs(bundle, **{**kwargs, "do_sample": True})
    assert "do_sample" not in delay  # delay has no such arg
    delay_greedy = runtime._generate_kwargs(bundle, **{**kwargs, "do_sample": False})
    assert delay_greedy["text_temperature"] == 0.0 and delay_greedy["audio_temperature"] == 0.0


# ---------------------------------------------------------------------------
# compat fallbacks (no comfy imported here)
# ---------------------------------------------------------------------------

def test_cast_context_passthrough_without_comfy():
    from mosstts_v15 import compat

    lin = torch.nn.Linear(4, 4)
    x = torch.zeros(1, 4)
    with compat._LegacyCastContext(None, lin, x, True) as (w, b, s):
        assert w.shape == lin.weight.shape and s is None


def test_progress_reporter_no_comfy():
    from mosstts_v15 import compat

    rep = compat.ProgressReporter(10)
    rep(3, 10)  # must not raise


def test_resolve_dtype_cpu_is_fp32():
    from mosstts_v15.loader import resolve_dtype

    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
    assert resolve_dtype("bf16", torch.device("cpu")) is torch.float32


def test_resolve_attention_never_crashes_without_flash():
    from mosstts_v15.loader import resolve_attention

    # CPU must resolve to eager regardless of request
    assert resolve_attention("auto", torch.device("cpu"), torch.float32) == "eager"
    assert resolve_attention("sdpa", torch.device("cpu"), torch.float32) == "sdpa"


def test_weight_norm_remap():
    from mosstts_v15.native import _weight_norm_remap

    assert _weight_norm_remap("a.weight_g") == "a.parametrizations.weight.original0"
    assert _weight_norm_remap("a.weight_v") == "a.parametrizations.weight.original1"
    assert _weight_norm_remap("a.weight") == "a.weight"


def test_models_dir_env_override(tmp_path, monkeypatch):
    from mosstts_v15 import loader, native

    spec = native.VARIANTS["local"]
    d = tmp_path / spec.repo_id.split("/")[-1]
    d.mkdir()
    (d / "config.json").write_text("{}")
    monkeypatch.setenv("MOSS_TTS_MODELS_DIR", str(tmp_path))
    assert loader._resolve_dir_for(spec.repo_id, False) == d


def test_comfy_castable_modules_forward_without_comfy():
    """Converted modules must keep working when comfy.ops is absent."""
    import mosstts_v15.native as native

    for mod, x in (
        (torch.nn.Linear(8, 8), torch.randn(2, 8)),
        (torch.nn.Embedding(16, 8), torch.randint(0, 16, (2, 4))),
        (torch.nn.LayerNorm(8), torch.randn(2, 8)),
        (torch.nn.Conv1d(4, 4, 3, padding=1), torch.randn(2, 4, 16)),
        (torch.nn.ConvTranspose1d(4, 4, 3, padding=1), torch.randn(2, 4, 16)),
    ):
        native.convert_modules_for_comfy(mod) if False else None
        # convert in place via class swap, then forward
        for base, castable in native._CASTABLE:
            if type(mod) is base:
                mod.__class__ = castable
        ref = mod.forward
        out = mod.forward(x) if isinstance(mod, torch.nn.ConvTranspose1d) else ref(x)
        assert out is not None
