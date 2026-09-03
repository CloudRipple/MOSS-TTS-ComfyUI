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
    load_soundeffect_bundle,
    sfx_bundle_generation,
)
from .native import DEFAULT_VARIANT, VARIANTS

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

MODEL_TYPE = "MOSSTTS_V15_MODEL"
SFX_MODEL_TYPE = "MOSS_SFX_V2_MODEL"

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
                                            "tooltip": "Local-Transformer (4B, 48kHz stereo, Qwen3-4B backbone) or Delay (8B, 24kHz) or VoiceGenerator (1.7B, 24kHz, voice design)."}),
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


class MossTTSV15VoiceDesign:
    """MOSS-VoiceGenerator（MossTTSDelay 家族，1.7B）：用文字描述音色并直接
    发声，不需要参考音频。官方对该模型推荐的采样默认值与 delay 不同
    （1.5 / 0.6 / 50 / 1.1），故本节点单独给出默认值。"""

    @classmethod
    def INPUT_TYPES(cls):
        knobs = _all_knobs()
        instruction = knobs.pop("instruction")
        instruction = ("STRING", {**instruction[1], "default": "A warm, friendly, young female voice.",
                                  "tooltip": "音色描述（必填）：性别/年龄/情绪/语速/口音等，中英文均可。产出音频可直接当 Voice Clone 的参考。"})
        # MOSS-VoiceGenerator recommended decoding defaults.
        for name, default in (("audio_temperature", 1.5), ("audio_top_p", 0.6),
                              ("audio_top_k", 50), ("audio_repetition_penalty", 1.1)):
            typ, kw = knobs[name]
            knobs[name] = (typ, {**kw, "default": default})
        return {"required": {
            "mosstts_model": (MODEL_TYPE, {}),
            "instruction": instruction,
            "text": ("STRING", {"multiline": True, "default": "Hello! This voice was designed from a text description."}),
            **knobs,
        }}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    FUNCTION = "run"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "声音设计：用文本描述音色并直接发声（MOSS-VoiceGenerator），无需参考音频。"

    def run(self, mosstts_model, text, language, instruction, audio_temperature,
            audio_top_p, audio_top_k, audio_repetition_penalty, text_temperature,
            text_top_p, text_top_k, target_tokens, max_new_tokens, do_sample, seed):
        if mosstts_model.spec.key != "voicegen":
            raise ValueError("Voice Design 需要在 Load Model 里选择 MOSS-VoiceGenerator。")
        if not (instruction or "").strip():
            raise ValueError("Voice Design 必须提供音色描述（instruction）。")
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


class MossTTSV15SoundEffectLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (DTYPE_OPTIONS, {"default": "auto", "tooltip": "auto = bf16 on CUDA, fp32 on CPU."}),
                "download_if_missing": ("BOOLEAN", {"default": True,
                                                    "tooltip": "从 $MOSS_TTS_MODELS_DIR / models/mosstts / HF cache 解析，缺失时才下载。"}),
            }
        }

    RETURN_TYPES = (SFX_MODEL_TYPE,)
    RETURN_NAMES = ("soundeffect_model",)
    FUNCTION = "load_model"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "加载 MOSS-SoundEffect-v2.0（DiT 1.3B + DAC，48 kHz mono，最长 30 秒）。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return sfx_bundle_generation()

    def load_model(self, dtype: str, download_if_missing: bool):
        return (load_soundeffect_bundle(dtype_name=dtype,
                                        download_if_missing=bool(download_if_missing)),)


class MossTTSV15SoundEffectGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soundeffect_model": (SFX_MODEL_TYPE, {}),
                "prompt": ("STRING", {"multiline": True, "default": "Rain falling gently on leaves, occasional distant thunder.",
                                      "tooltip": "音效描述，中英文均可；支持环境声/城市/动物/动作声等"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "不想要的内容（CFG 负提示词），一般留空"}),
                "seconds": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 30.0, "step": 0.1,
                                      "tooltip": "时长（秒），上限 30；实际按最长 latent 生成后裁剪"}),
                "steps": ("INT", {"default": 100, "min": 10, "max": 500, "step": 5,
                                  "tooltip": "去噪步数；官方默认 100"}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 15.0, "step": 0.1,
                                        "tooltip": "CFG 引导强度；官方默认 4.0"}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1,
                                          "tooltip": "flow-match shift；官方默认 5.0"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "文本生成音效/环境声（MOSS-SoundEffect-v2.0，48 kHz mono）。"

    def generate(self, soundeffect_model, prompt, negative_prompt, seconds, steps,
                 cfg_scale, sigma_shift, seed):
        wav = runtime.sound_effect(
            soundeffect_model, prompt=prompt, negative_prompt=negative_prompt,
            seconds=seconds, steps=steps, cfg_scale=cfg_scale,
            sigma_shift=sigma_shift, seed=seed,
            progress_reporter=ProgressReporter(steps),
        )
        return (runtime.tensor_to_comfy_audio(wav, soundeffect_model.sample_rate),)


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
    "MossTTSV15_VoiceDesign": MossTTSV15VoiceDesign,
    "MossTTSV15_ContinueSpeech": MossTTSV15ContinueSpeech,
    "MossTTSV15_EstimateTokens": MossTTSV15EstimateTokens,
    "MossTTSV15_SoundEffectLoad": MossTTSV15SoundEffectLoad,
    "MossTTSV15_SoundEffectGenerate": MossTTSV15SoundEffectGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSV15_LoadModel": "MOSS-TTS v1.5 Load Model",
    "MossTTSV15_GenerateSpeech": "MOSS-TTS v1.5 Generate Speech",
    "MossTTSV15_VoiceClone": "MOSS-TTS v1.5 Voice Clone",
    "MossTTSV15_VoiceDesign": "MOSS-TTS v1.5 Voice Design",
    "MossTTSV15_ContinueSpeech": "MOSS-TTS v1.5 Continue Speech",
    "MossTTSV15_EstimateTokens": "MOSS-TTS v1.5 Estimate Tokens",
    "MossTTSV15_SoundEffectLoad": "MOSS-TTS v1.5 Sound Effect Load",
    "MossTTSV15_SoundEffectGenerate": "MOSS-TTS v1.5 Sound Effect Generate",
}
