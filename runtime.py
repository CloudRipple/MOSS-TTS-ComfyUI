"""Generation runtime: AUDIO-dict conversion, seeding, conversation assembly,
and per-variant generate/decode dispatch.

Variant differences handled here (see assets/ for the definitive APIs):
  * local: processor.decode(outputs, return_stereo=True) -> fp32 CPU [2, T];
    generate() supports do_sample + top-level temperature fallbacks and takes
    a progress_callback(current_frame, total_frames).
  * delay: processor.decode(outputs) -> mono [T]; generate() has no do_sample
    (temperature <= 0 selects greedy) and, upstream, no progress callback — our
    vendored copy adds one (see "MOSS-TTS-V15-ComfyUI patch" sites in
    assets/moss_tts_delay/modeling_moss_tts.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from .loader import MossTTSBundle, resume_bundle

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

LANGUAGES = [
    "auto", "Chinese", "Cantonese", "English", "Arabic", "Czech", "Danish",
    "Dutch", "Finnish", "French", "German", "Greek", "Hebrew", "Hindi",
    "Hungarian", "Italian", "Japanese", "Korean", "Macedonian", "Malay",
    "Persian (Farsi)", "Polish", "Portuguese", "Romanian", "Russian",
    "Spanish", "Swahili", "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese",
]


# ---------------------------------------------------------------------------
# AUDIO dict <-> tensor
# ---------------------------------------------------------------------------

def comfy_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    """ComfyUI AUDIO {waveform [B,C,T], sample_rate} -> ([C,T] float32 CPU, sr)."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 2:
        wav = wav[:2]
    return wav.contiguous().clone(), sample_rate  # clone: input may be an inference tensor


def tensor_to_comfy_audio(wav: torch.Tensor, sample_rate: int) -> dict:
    """[T] / [C,T] / [B,C,T] -> ComfyUI AUDIO dict; inference tensors cloned out."""
    w = wav.detach().float().cpu().clone()
    if w.ndim == 1:
        w = w.unsqueeze(0)
    if w.ndim == 2:
        w = w.unsqueeze(0)
    return {"waveform": w.contiguous(), "sample_rate": int(sample_rate)}


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _require_text(text: str, field: str = "text") -> str:
    """Fail fast on empty prompts — MOSS never emits EOS with nothing to say
    and would generate until max_new_tokens, looking exactly like a hang."""
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError(
            f"'{field}' is empty. MOSS-TTS cannot start from an empty prompt "
            "(it never stops on its own in that case)."
        )
    return stripped


def _build_user_kwargs(
    *,
    text: str,
    language: str,
    instruction: str,
    target_tokens: int,
    reference: Optional[list] = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"text": text}
    if reference:
        kwargs["reference"] = reference
    if language and language != "auto":
        kwargs["language"] = language
    if int(target_tokens) > 0:
        kwargs["tokens"] = int(target_tokens)
    if instruction and instruction.strip():
        kwargs["instruction"] = instruction.strip()
    return kwargs


def _generate_kwargs(bundle: MossTTSBundle, *, max_new_tokens: int, do_sample: bool,
                     text_temperature: float, text_top_p: float, text_top_k: int,
                     audio_temperature: float, audio_top_p: float, audio_top_k: int,
                     audio_repetition_penalty: float) -> dict[str, Any]:
    """Filter/downsample the common knob set to what each variant's
    generate() actually accepts (delay hard-rejects unknown kwargs)."""
    base = {
        "max_new_tokens": int(max_new_tokens),
        "text_temperature": float(text_temperature),
        "text_top_p": float(text_top_p),
        "text_top_k": int(text_top_k),
        "audio_temperature": float(audio_temperature),
        "audio_top_p": float(audio_top_p),
        "audio_top_k": int(audio_top_k),
        "audio_repetition_penalty": float(audio_repetition_penalty),
        "show_progress": False,
    }
    if bundle.spec.key == "delay":
        # Delay has no do_sample flag; temperature <= 0 means greedy.
        if not do_sample:
            base["text_temperature"] = 0.0
            base["audio_temperature"] = 0.0
        return base
    # local
    base["do_sample"] = bool(do_sample)
    return base


def _extract_waveform(bundle: MossTTSBundle, outputs) -> torch.Tensor:
    """processor.decode() -> waveform tensor [C, T] (或 None-safe 报错)."""
    decode_kwargs = {}
    if bundle.spec.key == "local":
        decode_kwargs["return_stereo"] = True
    messages = bundle.processor.decode(outputs, **decode_kwargs)
    for message in messages:
        if message is not None and getattr(message, "audio_codes_list", None):
            wav = message.audio_codes_list[0]
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            return wav
    raise RuntimeError(
        "MOSS-TTS returned no decodable audio. Try a different seed, longer "
        "text, or check the input for illegal characters."
    )


def _encode_reference(bundle: MossTTSBundle, audio: dict) -> torch.Tensor:
    """ComfyUI AUDIO -> codec codes [T, n_vq] following the variant's channel
    convention (local: mono duplicated to stereo; delay: down-mixed to mono)."""
    resume_bundle(bundle)
    wav, sample_rate = comfy_audio_to_tensor(audio)
    if bundle.spec.key == "local" and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    codes_list = bundle.processor.encode_audios_from_wav([wav], sample_rate)
    return codes_list[0]


def _run_generate(bundle: MossTTSBundle, conversation, *, mode: str,
                  seed: int, progress_callback=None, **kwargs) -> torch.Tensor:
    resume_bundle(bundle)
    seed_everything(seed)
    batch = bundle.processor([conversation], mode=mode)
    input_ids = batch["input_ids"].to(bundle.device)
    attention_mask = batch["attention_mask"].to(bundle.device)
    with torch.inference_mode():
        outputs = bundle.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            progress_callback=progress_callback,
            **kwargs,
        )
    return _extract_waveform(bundle, outputs)


# ---------------------------------------------------------------------------
# Public node-facing operations
# ---------------------------------------------------------------------------

def generate_speech(bundle: MossTTSBundle, *, text: str, language: str,
                    instruction: str, target_tokens: int, seed: int,
                    progress_callback=None, **gen) -> tuple[torch.Tensor, int]:
    """Reference-free TTS. Returns (waveform [C,T], tokens_generated)."""
    clean = _require_text(text)
    user = bundle.processor.build_user_message(
        **_build_user_kwargs(text=clean, language=language, instruction=instruction,
                             target_tokens=target_tokens)
    )
    kwargs = _generate_kwargs(bundle, **gen)
    logger.info("[MOSS-TTS] speak variant=%s chars=%d target_tokens=%s",
                bundle.spec.key, len(clean), target_tokens or "auto")
    wav = _run_generate(bundle, [user], mode="generation", seed=seed,
                        progress_callback=progress_callback, **kwargs)
    return wav, _frames_of(bundle, wav)


def voice_clone(bundle: MossTTSBundle, *, reference_audio: dict, text: str,
                language: str, instruction: str, target_tokens: int, seed: int,
                progress_callback=None, **gen) -> tuple[torch.Tensor, int]:
    """Zero-shot clone. Returns (waveform [C,T], tokens_generated)."""
    clean = _require_text(text)
    codes = _encode_reference(bundle, reference_audio)
    user = bundle.processor.build_user_message(
        **_build_user_kwargs(text=clean, language=language, instruction=instruction,
                             target_tokens=target_tokens, reference=[codes])
    )
    kwargs = _generate_kwargs(bundle, **gen)
    logger.info("[MOSS-TTS] clone variant=%s chars=%d target_tokens=%s",
                bundle.spec.key, len(clean), target_tokens or "auto")
    wav = _run_generate(bundle, [user], mode="generation", seed=seed,
                        progress_callback=progress_callback, **kwargs)
    return wav, _frames_of(bundle, wav)


def continue_speech(bundle: MossTTSBundle, *, previous_audio: dict,
                    previous_text: str, text: str, language: str,
                    instruction: str, target_tokens: int, seed: int,
                    previous_tokens: int = 0, head_trim_frames: int = 1,
                    progress_callback=None, **gen
                    ) -> tuple[torch.Tensor, int, torch.Tensor, int]:
    """Prefix continuation. Returns (new_wav, new_frames, full_wav, full_frames).

    MOSS is a prefix-continuation model: build_assistant_message carries the
    prior audio, and the user message carries the full script
    (previous_text + new text). The 'tokens' hint is TOTAL (prefix + new).
    """
    new_text = _require_text(text)
    prev_text = (previous_text or "").strip()
    full_text = (prev_text + " " + new_text).strip() if prev_text else new_text

    _, prev_sr = comfy_audio_to_tensor(previous_audio)
    prev_seconds = previous_audio["waveform"].shape[-1] / float(prev_sr)
    prefix_frames = int(previous_tokens) if previous_tokens else round(prev_seconds * bundle.spec.frames_per_second)

    total_hint = 0
    if int(target_tokens) > 0:
        total_hint = prefix_frames + int(target_tokens)

    # Encode the prefix through the codec ourselves (in-memory tensors; no
    # torchaudio.load anywhere — that call hard-depends on torchcodec in new
    # torchaudio). Both variants accept pre-encoded codes [T, n_vq].
    prev_wav, _ = comfy_audio_to_tensor(previous_audio)
    prev_codes = _encode_reference(bundle, previous_audio)

    user = bundle.processor.build_user_message(
        **_build_user_kwargs(text=full_text, language=language,
                             instruction=instruction, target_tokens=total_hint)
    )
    assistant = bundle.processor.build_assistant_message(audio_codes_list=[prev_codes])
    kwargs = _generate_kwargs(bundle, **gen)
    logger.info(
        "[MOSS-TTS] continue variant=%s prev_chars=%d new_chars=%d prefix_frames=%d target_new=%s total_hint=%s",
        bundle.spec.key, len(prev_text), len(new_text), prefix_frames,
        target_tokens or "auto", total_hint or "auto",
    )
    wav = _run_generate(bundle, [user, assistant], mode="continuation",
                        seed=seed, progress_callback=progress_callback, **kwargs)

    # The conv codec's receptive field leaks the last prefix frame into the
    # decoded continuation head — trim ~1 frame (80ms) by default.
    trim = max(0, int(head_trim_frames))
    if trim > 0 and wav.numel() > 0:
        trim_samples = int(round(trim * bundle.spec.sample_rate / bundle.spec.frames_per_second))
        trim_samples = min(trim_samples, wav.shape[-1] - 1)
        if trim_samples > 0:
            wav = wav[..., trim_samples:]

    new_frames = _frames_of(bundle, wav)

    # full_wav = prefix + new, at the model's native sample rate
    import torchaudio

    full_prev = prev_wav
    if prev_sr != bundle.spec.sample_rate:
        full_prev = torchaudio.functional.resample(prev_wav, prev_sr, bundle.spec.sample_rate)
    if full_prev.shape[0] != wav.shape[0]:
        if wav.shape[0] == 1 and full_prev.shape[0] > 1:
            full_prev = full_prev.mean(dim=0, keepdim=True)
        elif wav.shape[0] > 1 and full_prev.shape[0] == 1:
            full_prev = full_prev.repeat(wav.shape[0], 1)
    full_wav = torch.cat([full_prev, wav.cpu()], dim=-1)
    return wav, new_frames, full_wav, prefix_frames + new_frames


def _frames_of(bundle: MossTTSBundle, wav: torch.Tensor) -> int:
    seconds = wav.shape[-1] / float(bundle.spec.sample_rate)
    return int(round(seconds * bundle.spec.frames_per_second))
