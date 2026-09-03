# MOSS-SoundEffect-v2.0 — vendored inference code

Vendored copy of the official OpenMOSS MOSS-SoundEffect-v2.0 inference code,
trimmed to inference-only and adapted for the ComfyUI-MOSS-TTS-v15 runtime.

## Provenance

- Source: `OpenMOSS-Team/moss-tts-official` (a.k.a. `MOSS-TTS` official repo),
  subdirectory `moss_soundeffect_v2/`
- Commit: `c0880299e8b8d0f7119efab17e4e776fffe7b8fa`
- Weights consumed: HF `OpenMOSS-Team/MOSS-SoundEffect-v2.0`
  (diffusers layout: `model_index.json`, `scheduler/`, `transformer/`,
  `text_encoder/`, `tokenizer/`, `vae/vae_128d_48k.pth`)

## Patch convention

Every place that deviates from upstream is marked
`# MOSS-TTS-V15-ComfyUI patch:` (same convention as the other packages under
`assets/`). Files without that marker are byte-identical to upstream.

## Layout

- `__init__.py` — package exports (`MossSoundEffectPipeline`, output dataclass). Upstream-verbatim.
- `pipeline_moss_soundeffect.py` — diffusers-style wrapper pipeline (`from_pretrained(model_index.json)`, `__call__(prompt, seconds, ...)`). Upstream-verbatim.
- `diffsynth/utils/__init__.py` — `BasePipeline` (audio-relevant bits) + `PipelineUnit`/`PipelineUnitRunner`; image/video + training helpers removed.
- `diffsynth/pipelines/wan_audio.py` — `WanAudioPipeline`: HF loader incl. the diffusers→Wan DiT state-dict remap (`_HF_DIT_BLOCK_RENAME` / `_HF_DIT_GLOBAL_RENAME` / `_convert_hf_dit_state_dict`), 4 audio PipelineUnits, CFG denoise loop, batched DAC decode.
- `diffsynth/models/dit_block.py` — DiT block pieces extracted from upstream `wan_video_dit.py` (`DiTBlock`, `SelfAttention`, `CrossAttention`, `GateModule`, `RMSNorm`, `modulate`, `sinusoidal_embedding_1d`, `precompute_freqs_cis`, `rope_apply`, optional flash-attn 2/3 / sageattention backend selection with SDPA fallback).
- `diffsynth/models/wan_audio_dit.py` — `WanAudioModel` (Wan-2.1-style 1D audio DiT) as plain `torch.nn.Module`.
- `diffsynth/models/qwen3_text_encoder.py` — `Qwen3TextEncoder` (Qwen3-1.7B last-hidden-state encoder).
- `diffsynth/models/dac_vae.py` — continuous-only DAC VAE (128-d latent, 48 kHz, hop 960) + own `DAC.load` for the packaged `.pth`.
- `diffsynth/schedulers/flow_match.py` — `FlowMatchScheduler` (`set_timesteps`/`step`/`add_noise`; training weighting removed).
- `diffsynth/prompters/` — `BasePrompter` + `WanPrompter` (Qwen tokenizer wrapper).

## Trimmed (NOT vendored)

- Upstream `diffsynth/trainers/`, `finetuning/`, `hf_export.py`,
  `infer_from_pipeline.py/.sh` — training / export / CLI shells.
- `models/wan_video_dit.py` (video `WanModel` + civitai hash tables),
  `models/wan_video_camera_controller.py`, `models/utils.py` (state-dict io /
  hash helpers) — unused by the inference path; `DiTBlock` etc. live in
  `models/dit_block.py` instead.
- `WanModelStateDictConverter` inside `wan_audio_dit.py`; the HF diffusers-named
  checkpoint is remapped by `_convert_hf_dit_state_dict` in
  `pipelines/wan_audio.py`.
- From `pipelines/wan_audio.py`: TeaCache, `dit2` + DiT-boundary switching,
  UnifiedSequenceParallel/xfuser, TemporalTiler + sliding window, VACE,
  motion_controller, `training_loss`, CfgMerger (cfg_merge), the
  `torch.compiler.cudagraph_mark_step_begin()` call and the `@torch.compile`
  decorator on `model_fn_wan_video`, plus `_WAN_AUDIO_DIT_PRESETS` /
  `_resolve_wan_audio_dit_preset` (dead upstream code).
- From `models/dac_vae.py`: `DACFile`, `compress`/`decompress`, the discrete
  RVQ quantizer (`VectorQuantize`, `ResidualVectorQuantize`), `forward()`/KL
  paths, `audiotools` imports.

## Compatibility adaptations

- No `diffusers` `ModelMixin`/`ConfigMixin`: `WanAudioModel` subclasses
  `torch.nn.Module`; the pipeline constructs it from explicit
  `transformer/config.json` kwargs.
- No `descript-audiotools`: `DAC.load` reimplements the packaged
  `{'state_dict', 'metadata': {'kwargs'}}` `torch.load` format.
- `einops` removed (rewritten as `view`/`transpose`/`permute`/`reshape`).
- `ftfy` optional in `wan_prompter.py` (falls back to html+whitespace cleaning).
- `Qwen3TextEncoder` passes `dtype=` (works on transformers 4.57 and 5.x;
  `torch_dtype=` is deprecated).

## Runtime dependencies

`torch` (>= 2.9), `transformers` (4.57 or 5.x), `safetensors`, `tqdm`, `regex`,
`huggingface-hub`. flash-attn 2/3 and sageattention are optional attention
accelerators; SDPA is the fallback.
