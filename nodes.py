"""ComfyUI node definitions for MOSS-TTS v1.5 (Local-Transformer 4B + Delay 8B).

V1 node protocol (NODE_CLASS_MAPPINGS / INPUT_TYPES / ...): the stable,
AST-parseable surface — exactly what the ComfyUI registry crawler expects.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from . import runtime
from .compat import ProgressReporter
from .loader import (
    ATTENTION_OPTIONS,
    DTYPE_OPTIONS,
    bundle_generation,
    load_mosstts_bundle,
)
from .native import DEFAULT_VARIANT, VARIANTS

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

MODEL_TYPE = "MOSSTTS_V15_MODEL"

_VARIANT_LABELS = [VARIANTS[k].label for k in VARIANTS]
_LABEL_TO_KEY = {VARIANTS[k].label: k for k in VARIANTS}

_FRAMES_TOOLTIP = (
    "Duration hint in audio frames (12.5 frames/s): 125 ≈ 10 s, 375 ≈ 30 s. "
    "0 = model decides via EOS. Wire the Estimate Tokens node to compute it."
)

_AUDIO_KNOBS = {
    "audio_temperature": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 3.0, "step": 0.05,
                                    "tooltip": "Acoustic sampling temperature (MOSS default 1.7)."}),
    "audio_top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Acoustic nucleus sampling."}),
    "audio_top_k": ("INT", {"default": 25, "min": 0, "max": 1024, "step": 1,
                            "tooltip": "Acoustic top-k."}),
    "audio_repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01,
                                           "tooltip": "1.0 = off. Mild values (1.05-1.15) suppress droning / tempo freeze."}),
}

_TEXT_KNOBS = {
    "text_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                   "tooltip": "Text-stream (alignment/pacing) temperature."}),
    "text_top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                             "tooltip": "Text-stream nucleus sampling."}),
    "text_top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1,
                           "tooltip": "Text-stream top-k."}),
}

_BUDGET_KNOBS = {
    "target_tokens": ("INT", {"default": 0, "min": 0, "max": 45000, "step": 1, "tooltip": _FRAMES_TOOLTIP}),
    "max_new_tokens": ("INT", {"default": 4096, "min": 16, "max": 45000, "step": 1,
                               "tooltip": "Hard generation budget in frames (12.5 fps): 4096 ≈ 5.5 min cap."}),
    "do_sample": ("BOOLEAN", {"default": True,
                              "tooltip": "Stochastic sampling; off = greedy decode (delay variant maps this to temperature=0)."}),
    "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1,
                     "tooltip": "Same seed + same inputs → identical output."}),
}

_CONDITIONING = {
    "language": (runtime.LANGUAGES, {"default": "auto",
                                     "tooltip": "Language hint. v1.5 performs best when it is set explicitly."}),
    "instruction": ("STRING", {"multiline": True, "default": "",
                               "tooltip": "Free-form style instruction, e.g. 'male, warm, elderly narrator'."}),
}


def _all_knobs() -> dict:
    merged: dict = {}
    merged.update(_CONDITIONING)
    merged.update(_AUDIO_KNOBS)
    merged.update(_TEXT_KNOBS)
    merged.update(_BUDGET_KNOBS)
    return merged


class MossTTSV15LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_VARIANT_LABELS, {"default": VARIANTS[DEFAULT_VARIANT].label,
                                            "tooltip": "Local-Transformer (4B, 48kHz stereo, Qwen3-4B backbone) or Delay (8B, 24kHz)."}),
                "dtype": (DTYPE_OPTIONS, {"default": "auto", "tooltip": "auto = bf16 on CUDA, fp32 on CPU."}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto",
                                                  "tooltip": "auto = flash_attention_2 when the flash_attn package exists, else sdpa."}),
                "download_if_missing": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Resolve models from $MOSS_TTS_MODELS_DIR / HF cache / models/mosstts first; download only when missing."}),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("mosstts_model",)
    FUNCTION = "load_model"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Load a MOSS-TTS v1.5 variant (weights + codec), integrated with ComfyUI memory management."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return bundle_generation()

    def load_model(self, model: str, dtype: str, attention: str, download_if_missing: bool):
        bundle = load_mosstts_bundle(
            variant=_LABEL_TO_KEY[model],
            dtype_name=dtype,
            attention=attention,
            download_if_missing=bool(download_if_missing),
        )
        return (bundle,)


class MossTTSV15GenerateSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {}),
            "text": ("STRING", {"multiline": True, "default": "Hello! This is MOSS-TTS v1.5 running inside ComfyUI.",
                                "tooltip": "Text to synthesize. Supports [pause 3.2s] markers, Pinyin and IPA (delay variant)."}),
        }
        required.update(_all_knobs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    FUNCTION = "generate"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Reference-free text-to-speech; the voice is steered by language + instruction."

    def generate(self, mosstts_model, text, language, instruction, audio_temperature,
                 audio_top_p, audio_top_k, audio_repetition_penalty, text_temperature,
                 text_top_p, text_top_k, target_tokens, max_new_tokens, do_sample, seed):
        wav, frames = runtime.generate_speech(
            mosstts_model, text=text, language=language, instruction=instruction,
            target_tokens=target_tokens, max_new_tokens=max_new_tokens, seed=seed,
            do_sample=do_sample, text_temperature=text_temperature, text_top_p=text_top_p,
            text_top_k=text_top_k, audio_temperature=audio_temperature,
            audio_top_p=audio_top_p, audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            progress_callback=ProgressReporter(max_new_tokens),
        )
        return (runtime.tensor_to_comfy_audio(wav, mosstts_model.spec.sample_rate), frames)


class MossTTSV15VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {}),
            "reference_audio": ("AUDIO", {"tooltip": "Voice reference (5-15 s recommended)."}),
            "text": ("STRING", {"multiline": True, "default": "This line will be spoken in the reference voice."}),
        }
        required.update(_all_knobs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    FUNCTION = "clone"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Zero-shot voice cloning from a reference AUDIO input."

    def clone(self, mosstts_model, reference_audio, text, language, instruction,
              audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty,
              text_temperature, text_top_p, text_top_k, target_tokens, max_new_tokens,
              do_sample, seed):
        wav, frames = runtime.voice_clone(
            mosstts_model, reference_audio=reference_audio, text=text, language=language,
            instruction=instruction, target_tokens=target_tokens,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            text_temperature=text_temperature, text_top_p=text_top_p, text_top_k=text_top_k,
            audio_temperature=audio_temperature, audio_top_p=audio_top_p,
            audio_top_k=audio_top_k, audio_repetition_penalty=audio_repetition_penalty,
            seed=seed, progress_callback=ProgressReporter(max_new_tokens),
        )
        return (runtime.tensor_to_comfy_audio(wav, mosstts_model.spec.sample_rate), frames)


class MossTTSV15ContinueSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {}),
            "previous_audio": ("AUDIO", {"tooltip": "Prior clip to continue from."}),
            "previous_text": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "Exact text that produced previous_audio (word-for-word matters)."}),
            "text": ("STRING", {"multiline": True, "default": "",
                                "tooltip": "Follow-up text to speak next."}),
            "previous_tokens": ("INT", {"default": 0, "min": 0, "max": 45000, "step": 1,
                                        "tooltip": "Frame count of previous_audio; wire tokens_generated from the upstream node. 0 = measure from audio duration."}),
            "head_trim_frames": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1,
                                         "tooltip": "Frames trimmed from the start of the NEW segment (codec receptive-field bleed ≈ 80 ms/frame)."}),
        }
        required.update(_all_knobs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "INT", "AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated", "full_audio", "full_tokens")
    FUNCTION = "continue_speech"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Prefix continuation: extend a generated clip in the same voice."

    def continue_speech(self, mosstts_model, previous_audio, previous_text, text,
                        previous_tokens, head_trim_frames, language, instruction,
                        audio_temperature, audio_top_p, audio_top_k,
                        audio_repetition_penalty, text_temperature, text_top_p, text_top_k,
                        target_tokens, max_new_tokens, do_sample, seed):
        wav, frames, full_wav, full_frames = runtime.continue_speech(
            mosstts_model, previous_audio=previous_audio, previous_text=previous_text,
            text=text, language=language, instruction=instruction,
            target_tokens=target_tokens, previous_tokens=previous_tokens,
            head_trim_frames=head_trim_frames, max_new_tokens=max_new_tokens,
            do_sample=do_sample, text_temperature=text_temperature, text_top_p=text_top_p,
            text_top_k=text_top_k, audio_temperature=audio_temperature,
            audio_top_p=audio_top_p, audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty, seed=seed,
            progress_callback=ProgressReporter(max_new_tokens),
        )
        sr = mosstts_model.spec.sample_rate
        return (
            runtime.tensor_to_comfy_audio(wav, sr), frames,
            runtime.tensor_to_comfy_audio(full_wav, sr), full_frames,
        )


class MossTTSV15EstimateTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "words_per_minute": ("FLOAT", {"default": 150.0, "min": 60.0, "max": 400.0, "step": 5.0,
                                               "tooltip": "Speaking-rate assumption. For CJK this is characters-per-minute."}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("target_tokens",)
    FUNCTION = "estimate"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Heuristic text → target_tokens estimate for duration control."

    def estimate(self, text: str, words_per_minute: float):
        text = (text or "").strip()
        if not text:
            return (0,)
        if _is_cjk(text):
            units = sum(1 for ch in text if not ch.isspace())
        else:
            units = len(text.split())
        seconds = units / max(1e-3, float(words_per_minute) / 60.0)
        return (int(math.ceil(seconds * 12.5)),)


def _is_cjk(text: str) -> bool:
    for ch in text[:200]:
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x309F
                or 0x30A0 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF):
            return True
    return False


NODE_CLASS_MAPPINGS = {
    "MossTTSV15_LoadModel": MossTTSV15LoadModel,
    "MossTTSV15_GenerateSpeech": MossTTSV15GenerateSpeech,
    "MossTTSV15_VoiceClone": MossTTSV15VoiceClone,
    "MossTTSV15_ContinueSpeech": MossTTSV15ContinueSpeech,
    "MossTTSV15_EstimateTokens": MossTTSV15EstimateTokens,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSV15_LoadModel": "MOSS-TTS v1.5 Load Model",
    "MossTTSV15_GenerateSpeech": "MOSS-TTS v1.5 Generate Speech",
    "MossTTSV15_VoiceClone": "MOSS-TTS v1.5 Voice Clone",
    "MossTTSV15_ContinueSpeech": "MOSS-TTS v1.5 Continue Speech",
    "MossTTSV15_EstimateTokens": "MOSS-TTS v1.5 Estimate Tokens",
}
