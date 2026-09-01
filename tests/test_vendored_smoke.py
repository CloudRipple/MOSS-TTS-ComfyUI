"""Smoke tests for the vendored remote-code packages, runnable under both
transformers 4.57.x and 5.x venvs. Models are built on the meta device so no
weights are needed and RAM stays flat."""

from __future__ import annotations

import torch

import moss_tts.native as native
from moss_tts.native import VARIANTS


def _real_tokenizer(spec):
    """Real Qwen tokenizer from the local HF cache (weights excluded); skip when
    offline so the rest of the suite still runs in generic CI."""
    pytest_skip = __import__("pytest").skip
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            spec.repo_id,
            local_files_only=True,
            allow_patterns=["*.json", "*.txt", "*.jinja"],
        )
    except Exception as e:
        pytest_skip(f"local HF cache for {spec.repo_id} not available: {e}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snap)


def _real_codec(spec):
    """Build the real vendored codec on the meta device — exactly the class
    ProcessorMixin must accept in production."""
    cfg = native.read_codec_config(spec, native._assets_root() / spec.codec_asset_pkg)
    return native.build_codec_model(spec, cfg, torch.bfloat16, "sdpa")


def test_local_config_model_meta():
    spec = VARIANTS["local"]
    cfg = native.read_tts_config(spec, native._assets_root() / spec.asset_pkg)
    model = native.build_tts_model(spec, cfg, "sdpa")
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 1e9  # Qwen3-4B-class backbone on meta device


def test_delay_config_model_meta():
    spec = VARIANTS["delay"]
    cfg = native.read_tts_config(spec, native._assets_root() / spec.asset_pkg)
    model = native.build_tts_model(spec, cfg, "sdpa")
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 1e9


def test_codec_configs_instantiate():
    for spec in VARIANTS.values():
        codec = _real_codec(spec)
        total = sum(p.numel() for p in codec.parameters())
        assert total > 1e8  # ~300M codec on meta


def test_processor_constructs_both_variants():
    """ProcessorMixin must accept the (tokenizer, audio_tokenizer, model_config)
    triple on every supported transformers version."""
    for spec in VARIANTS.values():
        cfg = native.read_tts_config(spec, native._assets_root() / spec.asset_pkg)
        processor = native.build_processor(spec, _real_tokenizer(spec), _real_codec(spec), cfg)
        assert processor.audio_tokenizer is not None
        assert int(processor.model_config.sampling_rate) == spec.sample_rate


def test_local_build_user_message_fields():
    spec = VARIANTS["local"]
    cfg = native.read_tts_config(spec, native._assets_root() / spec.asset_pkg)
    processor = native.build_processor(spec, _real_tokenizer(spec), _real_codec(spec), cfg)
    msg = processor.build_user_message(text="hello", language="English", tokens=25)
    assert msg["role"] == "user"


def test_delay_progress_callback_patch_present():
    """The vendored delay generate() must accept progress_callback (our patch)."""
    import inspect

    from _mossttsv15_moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

    params = inspect.signature(MossTTSDelayModel.generate).parameters
    assert "progress_callback" in params
    assert "show_progress" in params
    # tqdm must be opt-in for pack usage, not hardcoded on
    assert params["show_progress"].default is False
