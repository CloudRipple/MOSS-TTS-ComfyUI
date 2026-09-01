# MOSS-TTS v1.5 for ComfyUI

[中文](README.md)

ComfyUI custom nodes for [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
and [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5):

- **Local-Transformer** — Qwen3-**4B**-class backbone + nano-GPT2 local transformer, MOSS-Audio-Tokenizer-v2, **48 kHz stereo**, n_vq=12. (Note: the "1.7B" floating around in other packs' READMEs is wrong; the backbone config is Qwen3-4B-shaped, checkpoint ≈ 9.1 GB in bf16.)
- **Delay** — 8B delay-pattern model, MOSS-Audio-Tokenizer (v1), **24 kHz**, n_vq=32.

Both are loaded through the same nodes.

## Highlights

- Reference-free TTS, zero-shot voice cloning, continuation, hard duration control (`target_tokens`, 12.5 frames/s), 31 languages with explicit language tags, `[pause 3.2s]` markers.
- No `trust_remote_code`: the model code is vendored under `assets/` (with minimal, clearly marked compatibility patches), so the pack does not depend on whatever happens to be in your HF module cache.
- Works on both transformers 4.x and 5.x for both variants (patched vendored code + feature detection); E2E-validated on both 5.16 and 4.57 across two machines.
- Deep ComfyUI memory-management integration: weights register through `ModelPatcher` / `ModelPatcherDynamic` (AIMDO DynamicVRAM aware) and participate in normal unloads; every integration point is feature-detected and degrades gracefully on older ComfyUI.
- Weights are resolved, in order: `$MOSS_TTS_MODELS_DIR/<Repo-Name>` → `ComfyUI/models/mosstts/<Repo-Name>` → the HF hub cache (respecting `HF_HOME`) → auto-download (when `download_if_missing`).

## Install

**ComfyUI-Manager**: Manager → Custom Nodes Manager → search `moss-tts` → Install. Manager fetches the package and installs dependencies (`install.py`) automatically; restart ComfyUI afterwards.

**Git clone**:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CloudRipple/MOSS-TTS-ComfyUI.git
# deps (only missing ones get installed):
python install.py   # or: pip install -r requirements.txt
```

Restart ComfyUI. First use resolves weights from your local caches; if absent
and `download_if_missing` is on, they download from Hugging Face
(~9.1 GB + ~8 GB codec for Local; ~17 GB + ~6.7 GB codec for Delay).

## Nodes (category `MOSS-TTS v1.5`)

| Node | Purpose |
|---|---|
| MOSS-TTS v1.5 Load Model | Load variant (`local` / `delay`), dtype (auto/bf16/fp16/fp32), attention (auto/sdpa/flash_attention_2/eager), download toggle. |
| MOSS-TTS v1.5 Generate Speech | Reference-free TTS. |
| MOSS-TTS v1.5 Voice Clone | Clone from a reference `AUDIO` input. |
| MOSS-TTS v1.5 Continue Speech | Prefix continuation: extend a clip in the same voice. Outputs both the new segment and the stitched full audio (+ exact frame counts for chaining). |
| MOSS-TTS v1.5 Estimate Tokens | Text → `target_tokens` estimate for duration control. |

All generator nodes output `tokens_generated` (audio frames; seconds = frames / 12.5) so continuation chains can hand exact prefix lengths forward.

## Remote API nodes (category `MOSS-TTS v1.5/Remote`)

Instead of loading models locally, you can call the hosted Moss platform (mosi.cn) API:

| Node | Purpose |
|---|---|
| Remote Connect | Connection settings: base_url / API key (or env `MOSS_API_KEY`) / model (moss-tts-1.5-flash, moss-tts-1.0-pro) / async toggle. |
| Remote Speech | Hosted single-speaker TTS; requires a `voice_id`; the 1.5-flash model supports inline `[pause 1.5s]` markers. |
| Remote Voice Clone | Clone a voice from a reference audio on the platform; returns the `voice_id` for Speech. |
| Remote Voice List | List available voices (name + voice id) on your account. |

Not available remotely: continuation, token-level duration control, sampling knobs — those need the local models.

Get an API key from the [Moss platform](https://platform.mosi.cn) console ("API 密钥" page). Ready-made voices can be copied from the [Mossland voice library](https://mossland.studio/voice/library) (card → copy voice id) and pasted into Remote Speech.

## VRAM

- Local (4B): ~12 GB bf16 active.
- Delay (8B): ~22 GB bf16 active.

Keep voice-clone references short (5–15 s); prefix/PKV memory grows linearly with prefix duration.

## Troubleshooting

- **Nothing happens / model can't be found**: the loader prints the exact search paths. Set `$MOSS_TTS_MODELS_DIR` to a directory containing `MOSS-TTS-Local-Transformer-v1.5/` etc., or rely on `HF_HOME` pointing at your HF cache.
- **flash-attn errors**: set attention to `auto` or `sdpa`. flash-attn is optional; sdpa quality is identical.
- **VRAM pressure alongside big ComfyUI workflows**: the pack registers with ComfyUI's memory management, so `Free memory` / unloading works as usual.

## Updating the vendored model code

`assets/` mirrors the remote code of the four HF repos. Patches are marked
`# MOSS-TTS-V15-ComfyUI patch:` and limited to: transformers 4.x/5.x import
compat, an optional progress callback for the delay variant, and Path-accepting
`encode_audios_from_path`. To sync with upstream, recopy the four repos' code
files and re-apply the grep-able patch set.

## License

MIT (this pack). Model weights and upstream model code are Apache-2.0 by OpenMOSS-Team.
