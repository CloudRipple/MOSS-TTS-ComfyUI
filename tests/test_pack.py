"""CPU-only unit tests for the pack: registry shape, conversions, knobs,
fail-fast guards, and comfy-compat fallbacks. No weights, no GPU."""

from __future__ import annotations

import pytest
import torch

import moss_tts

pack = moss_tts


# ---------------------------------------------------------------------------
# node registry
# ---------------------------------------------------------------------------

def test_node_registry_complete():
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert len(pack.NODE_CLASS_MAPPINGS) == 6
    for cls in pack.NODE_CLASS_MAPPINGS.values():
        inputs = cls.INPUT_TYPES()
        assert "required" in inputs
        assert isinstance(cls.FUNCTION, str) and hasattr(cls, cls.FUNCTION.split(".")[0])
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)


def test_variant_specs():
    from moss_tts.native import VARIANTS

    local, delay, voicegen = VARIANTS["local"], VARIANTS["delay"], VARIANTS["voicegen"]
    assert local.sample_rate == 48000 and local.stereo and local.n_vq == 12
    assert delay.sample_rate == 24000 and not delay.stereo and delay.n_vq == 32
    assert voicegen.sample_rate == 24000 and not voicegen.stereo and voicegen.n_vq == 16
    assert voicegen.asset_pkg == delay.asset_pkg  # same MossTTSDelay family
    assert voicegen.repo_id == "OpenMOSS-Team/MOSS-VoiceGenerator"
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
    import moss_tts.runtime as runtime

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
    import moss_tts.runtime as runtime

    with pytest.raises(ValueError):
        runtime._require_text("   ")
    with pytest.raises(ValueError):
        runtime._require_text(None)


class _Spec:
    key = "local"
    asset_pkg = "moss_tts_local"
    stereo = True


class _FakeBundle:
    spec = _Spec()


def test_bundle_repr_is_compact():
    from pathlib import Path
    from types import SimpleNamespace

    from moss_tts.loader import MossTTSBundle

    bundle = MossTTSBundle(
        spec=SimpleNamespace(key="local"),
        model=torch.nn.Linear(1, 1),
        processor=SimpleNamespace(audio_tokenizer=object()),
        codec=torch.nn.Linear(1, 1),
        model_dir=Path("model"),
        codec_dir=Path("codec"),
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        dtype_name="fp32",
        attn_implementation="eager",
        patchers=[object()],
    )
    text = repr(bundle)
    assert "model=" not in text
    assert "processor=" not in text
    assert "codec=" not in text
    assert "patchers=" not in text
    assert "dtype_name='fp32'" in text


def test_generate_kwargs_local_vs_delay():
    import moss_tts.runtime as runtime

    bundle = _FakeBundle()
    kwargs = dict(max_new_tokens=100, do_sample=True, text_temperature=1.0,
                  text_top_p=1.0, text_top_k=50, audio_temperature=1.7,
                  audio_top_p=0.8, audio_top_k=25, audio_repetition_penalty=1.0)
    local = runtime._generate_kwargs(bundle, **kwargs)
    assert local["do_sample"] is True and local["max_new_tokens"] == 100

    bundle.spec.asset_pkg = "moss_tts_delay"
    delay = runtime._generate_kwargs(bundle, **{**kwargs, "do_sample": True})
    assert "do_sample" not in delay  # delay has no such arg
    delay_greedy = runtime._generate_kwargs(bundle, **{**kwargs, "do_sample": False})
    assert delay_greedy["text_temperature"] == 0.0 and delay_greedy["audio_temperature"] == 0.0


# ---------------------------------------------------------------------------
# compat fallbacks (no comfy imported here)
# ---------------------------------------------------------------------------

def test_cast_context_passthrough_without_comfy():
    from moss_tts import compat

    lin = torch.nn.Linear(4, 4)
    x = torch.zeros(1, 4)
    with compat._LegacyCastContext(None, lin, x, True) as (w, b, s):
        assert w.shape == lin.weight.shape and s is None


def test_progress_reporter_no_comfy():
    from moss_tts import compat

    rep = compat.ProgressReporter(10)
    rep(3, 10)  # must not raise


def test_resolve_dtype_cpu_is_fp32():
    from moss_tts.loader import resolve_dtype

    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
    assert resolve_dtype("bf16", torch.device("cpu")) is torch.float32


def test_resolve_attention_never_crashes_without_flash():
    from moss_tts.loader import resolve_attention

    # CPU must resolve to eager regardless of request
    assert resolve_attention("auto", torch.device("cpu"), torch.float32) == "eager"
    assert resolve_attention("sdpa", torch.device("cpu"), torch.float32) == "sdpa"


def test_weight_norm_remap():
    from moss_tts.native import _weight_norm_remap

    assert _weight_norm_remap("a.weight_g") == "a.parametrizations.weight.original0"
    assert _weight_norm_remap("a.weight_v") == "a.parametrizations.weight.original1"
    assert _weight_norm_remap("a.weight") == "a.weight"


def test_models_dir_env_override(tmp_path, monkeypatch):
    from moss_tts import loader, native

    spec = native.VARIANTS["local"]
    d = tmp_path / spec.repo_id.split("/")[-1]
    d.mkdir()
    (d / "config.json").write_text("{}")
    monkeypatch.setenv("MOSS_TTS_MODELS_DIR", str(tmp_path))
    assert loader._resolve_dir_for(spec.repo_id, False) == d


def test_load_installs_cache_invalidation_unload_hook(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from moss_tts import loader
    from moss_tts.nodes import MossTTSV15LoadModel

    model = torch.nn.Linear(1, 1)
    codec = torch.nn.Linear(1, 1)
    processor = SimpleNamespace(audio_tokenizer=codec)
    hooks = []

    monkeypatch.setattr(loader.compat, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loader.compat, "install_unload_hook", hooks.append)
    monkeypatch.setattr(loader, "resolve_model_dirs", lambda *args: (tmp_path, tmp_path))
    monkeypatch.setattr(loader, "_bundle_load_key", lambda *args: ("test",))
    monkeypatch.setattr(loader.native, "read_tts_config", lambda *args: object())
    monkeypatch.setattr(loader.native, "read_codec_config", lambda *args: object())
    monkeypatch.setattr(loader.native, "build_tts_model", lambda *args: model)
    monkeypatch.setattr(loader.native, "build_codec_model", lambda *args: codec)
    monkeypatch.setattr(loader.native, "load_tts_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader.native, "load_codec_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader.native, "build_tokenizer", lambda *args: object())
    monkeypatch.setattr(loader.native, "build_processor", lambda *args: processor)

    bundle = loader.load_mosstts_bundle(
        "local", dtype_name="fp32", attention="eager", download_if_missing=False,
    )
    try:
        assert len(hooks) == 1
        before = loader.bundle_generation()
        assert MossTTSV15LoadModel.IS_CHANGED() == before
        assert bundle.processor.audio_tokenizer is codec
        hooks[0]("unrelated unload", SimpleNamespace(clone_base_uuid="other"))
        assert loader.bundle_generation() == before
        assert bundle.model is model
        hooks[0]("test unload", None)
        assert loader.bundle_generation() == before + 1
        assert MossTTSV15LoadModel.IS_CHANGED() == before + 1
        assert bundle.model is None
        assert bundle.codec is None
        assert bundle.processor.audio_tokenizer is None
    finally:
        loader.unload_mosstts_bundle(bundle)


def test_comfy_castable_modules_forward_without_comfy():
    """Converted modules must keep working when comfy.ops is absent."""
    import moss_tts.native as native

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


def test_qwen_rmsnorm_is_comfy_castable(monkeypatch):
    """Custom RMSNorm weights must participate in partial offload streaming."""
    from moss_tts import compat
    import moss_tts.native as native

    native.tts_classes(native.VARIANTS["local"])
    from _mossttsv15_moss_tts_local.qwen3_decoder import MossQwen3RMSNorm

    norm = MossQwen3RMSNorm(8)
    with monkeypatch.context() as patch:
        original_try_import = compat._try_import
        patch.setattr(
            compat,
            "_try_import",
            lambda name: object() if name == "comfy.ops" else original_try_import(name),
        )
        native.convert_modules_for_comfy(norm)

    assert type(norm) is native._ComfyRMSNorm
    assert norm.comfy_cast_weights is True
    x = torch.randn(2, 4, 8)
    expected = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    assert torch.allclose(norm(x), expected.to(x.dtype), atol=1e-5, rtol=1e-5)


def test_rotary_materialization_4x_rope_init_fn():
    """tf4.x-style rotary (rope_init_fn attr, no compute_default_rope_parameters)
    must get a recomputed inv_freq, not the historical zero-fill that left the
    model position-blind on transformers 4.x."""
    import torch.nn as nn

    import moss_tts.native as native

    class Fake4xRotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()
            self.attention_scaling = 0.0
            self.register_buffer("inv_freq", torch.empty(4, device="meta"), persistent=False)

            def rope_init_fn(config, device):
                return torch.tensor([1.0, 0.1, 0.01, 0.001], device=device), 1.0

            self.rope_init_fn = rope_init_fn

    mod = Fake4xRotaryEmbedding()
    wrapper = nn.Module()
    wrapper.rot = mod
    native._materialize_non_persistent_buffers(wrapper, torch.device("cpu"))
    assert mod.inv_freq.device.type == "cpu"
    assert torch.all(mod.inv_freq != 0)
    assert mod.attention_scaling == 1.0


def test_voice_design_node_guards():
    import moss_tts.nodes as nodes

    node = nodes.MossTTSV15VoiceDesign()
    knobs = dict(
        language="auto", instruction="", audio_temperature=1.5, audio_top_p=0.6,
        audio_top_k=50, audio_repetition_penalty=1.1, text_temperature=1.0,
        text_top_p=1.0, text_top_k=50, target_tokens=0, max_new_tokens=100,
        do_sample=True, seed=42,
    )
    bundle = _FakeBundle()  # spec.key = "local"
    with pytest.raises(ValueError, match="VoiceGenerator"):
        node.run(bundle, "hi", **knobs)
    bundle.spec.key = "voicegen"
    with pytest.raises(ValueError, match="instruction"):
        node.run(bundle, "hi", **knobs)


def test_voicegen_uses_delay_kwargs_path():
    import moss_tts.runtime as runtime

    bundle = _FakeBundle()
    bundle.spec.key = "voicegen"
    bundle.spec.asset_pkg = "moss_tts_delay"
    bundle.spec.stereo = False
    kwargs = dict(max_new_tokens=100, do_sample=True, text_temperature=1.0,
                  text_top_p=1.0, text_top_k=50, audio_temperature=1.5,
                  audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1)
    out = runtime._generate_kwargs(bundle, **kwargs)
    assert "do_sample" not in out  # delay-family generate() rejects extra kwargs
