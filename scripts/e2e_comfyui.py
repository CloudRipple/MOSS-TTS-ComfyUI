#!/usr/bin/env python3
"""End-to-end driver: run MOSS-TTS v1.5 workflows against a live ComfyUI
server over the HTTP API and validate the produced audio files.

Usage examples:
  python scripts/e2e_comfyui.py --server http://127.0.0.1:8188 --variant local --mode speak
  python scripts/e2e_comfyui.py --server http://127.0.0.1:8188 --variant delay --mode clone --ref ref_zh.wav
  python scripts/e2e_comfyui.py --server http://127.0.0.1:8188 --variant local --mode continue --ref ref_zh.wav

Modes: speak | clone | continue (a two-stage chain: clone segment 1, then
continue with segment 2 using the generated clip as the prefix).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path


VARIANT_LABEL = {
    "local": "MOSS-TTS-Local-Transformer-v1.5 (4B, 48kHz stereo)",
    "delay": "MOSS-TTS-v1.5 (8B, 24kHz)",
    "voicegen": "MOSS-VoiceGenerator (1.7B, 24kHz, voice design)",
}
EXPECT_SR = {"local": 48000, "delay": 24000, "voicegen": 24000}

VG_INSTRUCTION = "疲惫沙哑的老年声音缓慢抱怨，带有轻微呻吟。"
# MOSS-VoiceGenerator recommended decoding defaults (from its model card).
VG_SAMPLING = {"audio_temperature": 1.5, "audio_top_p": 0.6,
               "audio_top_k": 50, "audio_repetition_penalty": 1.1}


def prompt_voice_design(seed: int, max_new_tokens: int) -> dict:
    sampling = {**_common_sampling(seed, max_new_tokens), **VG_SAMPLING}
    sampling.pop("instruction")  # VoiceDesign takes the voice description here
    return {"prompt": {
        "1": {"class_type": "MossTTSV15_LoadModel", "inputs": {
            "model": VARIANT_LABEL["voicegen"], "dtype": "auto", "attention": "auto",
            "download_if_missing": False}},
        "2": {"class_type": "MossTTSV15_VoiceDesign", "inputs": {
            "mosstts_model": ["1", 0], "instruction": VG_INSTRUCTION,
            "text": TEXT_A, **sampling}},
        "9": {"class_type": "SaveAudio", "inputs": {
            "audio": ["2", 0], "filename_prefix": "e2e_voicegen_design"}},
    }}

TEXT_A = "大家好，这里是 MOSS-TTS v1.5 的端到端验证。如果你听到了这段话，说明生成工作正常。"
TEXT_B = "接下来这句是续写段落，用来验证同一说话人的无缝衔接。"


def post_json(server: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        server + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def get_json(server: str, path: str) -> dict:
    with urllib.request.urlopen(server + path, timeout=120) as resp:
        return json.loads(resp.read())


def upload_audio(server: str, path_or_bytes, name: str) -> str:
    boundary = f"----e2e{random.getrandbits(64):x}"
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else Path(path_or_bytes).read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + bytes(data) + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        server + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["name"]


def _common_sampling(seed: int, max_new_tokens: int) -> dict:
    return {
        "language": "Chinese", "instruction": "", "target_tokens": 0,
        "max_new_tokens": max_new_tokens, "do_sample": True,
        "audio_temperature": 1.7, "audio_top_p": 0.8, "audio_top_k": 25,
        "text_temperature": 1.0, "text_top_p": 1.0, "text_top_k": 50,
        "audio_repetition_penalty": 1.0, "seed": seed,
    }


def prompt_speak(variant: str, seed: int, max_new_tokens: int) -> dict:
    return {"prompt": {
        "1": {"class_type": "MossTTSV15_LoadModel", "inputs": {
            "model": VARIANT_LABEL[variant], "dtype": "auto", "attention": "auto",
            "download_if_missing": False}},
        "2": {"class_type": "MossTTSV15_GenerateSpeech", "inputs": {
            "mosstts_model": ["1", 0], "text": TEXT_A, **_common_sampling(seed, max_new_tokens)}},
        "9": {"class_type": "SaveAudio", "inputs": {
            "audio": ["2", 0], "filename_prefix": f"e2e_{variant}_speak"}},
    }}


def prompt_clone(variant: str, ref_name: str, seed: int, max_new_tokens: int,
                 prefix: str) -> dict:
    return {"prompt": {
        "1": {"class_type": "MossTTSV15_LoadModel", "inputs": {
            "model": VARIANT_LABEL[variant], "dtype": "auto", "attention": "auto",
            "download_if_missing": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": ref_name, "videoUI": ""}},
        "3": {"class_type": "MossTTSV15_VoiceClone", "inputs": {
            "mosstts_model": ["1", 0], "reference_audio": ["2", 0], "text": TEXT_A,
            **_common_sampling(seed, max_new_tokens)}},
        "9": {"class_type": "SaveAudio", "inputs": {
            "audio": ["3", 0], "filename_prefix": prefix}},
    }}


def prompt_continue(variant: str, prev_name: str, seed: int, max_new_tokens: int,
                    prefix: str) -> dict:
    return {"prompt": {
        "1": {"class_type": "MossTTSV15_LoadModel", "inputs": {
            "model": VARIANT_LABEL[variant], "dtype": "auto", "attention": "auto",
            "download_if_missing": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": prev_name, "videoUI": ""}},
        "4": {"class_type": "MossTTSV15_ContinueSpeech", "inputs": {
            "mosstts_model": ["1", 0], "previous_audio": ["2", 0],
            "previous_text": TEXT_A, "text": TEXT_B,
            "previous_tokens": 0, "head_trim_frames": 1,
            **_common_sampling(seed, max_new_tokens)}},
        "9": {"class_type": "SaveAudio", "inputs": {
            "audio": ["4", 2], "filename_prefix": prefix}},  # full_audio
    }}


def run_prompt(server: str, prompt: dict, label: str, timeout_s: int = 1800) -> dict:
    prompt_id = post_json(server, "/prompt", prompt)["prompt_id"]
    print(f"[e2e] queued {label} prompt_id={prompt_id}", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(2)
        history = get_json(server, f"/history/{prompt_id}")
        if prompt_id not in history:
            continue
        result = history[prompt_id]
        status = result.get("status", {})
        if not status.get("completed") or status.get("status_str") != "success":
            print(json.dumps(result, ensure_ascii=False)[:3000])
            raise SystemExit(f"[e2e] FAIL {label}: {status}")
        saved = []
        for node_out in result.get("outputs", {}).values():
            for entry in node_out.get("audio", []):
                saved.append((entry["filename"], entry.get("subfolder", "")))
        if not saved:
            raise SystemExit(f"[e2e] FAIL {label}: no audio output")
        return {"saved": saved}
    raise SystemExit(f"[e2e] FAIL {label}: timeout after {timeout_s}s")


def fetch_output(server: str, filename: str, subfolder: str) -> bytes:
    url = f"{server}/view?filename={filename}&subfolder={subfolder}&type=output"
    return urllib.request.urlopen(url, timeout=120).read()


def validate_wav(path: Path, expect_sr: int, min_seconds: float = 0.5) -> None:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True)  # [T, C]
    rms = float(np.sqrt(np.square(data).mean()))
    seconds = data.shape[0] / sr
    assert sr == expect_sr, f"{path}: sample rate {sr} != expected {expect_sr}"
    assert seconds >= min_seconds, f"{path}: too short ({seconds:.2f}s)"
    assert rms > 1e-4, f"{path}: suspiciously silent (rms={rms:.2e})"
    print(f"[e2e] OK {path.name}: {sr}Hz x{data.shape[1]}ch {seconds:.2f}s rms={rms:.4f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--variant", choices=["local", "delay", "voicegen"], required=True)
    ap.add_argument("--mode", choices=["speak", "clone", "continue", "voicedesign"], required=True)
    ap.add_argument("--ref", type=Path, help="reference wav (clone/continue)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=Path("e2e_out"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "voicedesign":
        assert args.variant == "voicegen", "voicedesign mode requires --variant voicegen"
        result = run_prompt(args.server, prompt_voice_design(args.seed, args.max_new_tokens),
                            "voicegen/voicedesign")
        names = result["saved"]
    elif args.mode == "speak":
        result = run_prompt(args.server, prompt_speak(args.variant, args.seed, args.max_new_tokens),
                            f"{args.variant}/speak")
        names = result["saved"]
    elif args.mode == "clone":
        assert args.ref, "--ref is required for clone mode"
        ref_name = upload_audio(args.server, args.ref, "e2e_ref.wav")
        print(f"[e2e] uploaded ref -> {ref_name}")
        result = run_prompt(args.server, prompt_clone(args.variant, ref_name, args.seed,
                                                      args.max_new_tokens, f"e2e_{args.variant}_clone"),
                            f"{args.variant}/clone")
        names = result["saved"]
    else:  # continue = clone stage + continue stage
        assert args.ref, "--ref is required for continue mode"
        ref_name = upload_audio(args.server, args.ref, "e2e_ref.wav")
        stage1 = run_prompt(args.server, prompt_clone(args.variant, ref_name, args.seed,
                                                      args.max_new_tokens, f"e2e_{args.variant}_seg1"),
                            f"{args.variant}/continue:seg1")
        seg1_filename, seg1_sub = stage1["saved"][0]
        seg1_bytes = fetch_output(args.server, seg1_filename, seg1_sub)
        prev_name = upload_audio(args.server, seg1_bytes, f"e2e_{args.variant}_prev.wav")
        stage2 = run_prompt(args.server, prompt_continue(args.variant, prev_name, args.seed,
                                                         args.max_new_tokens, f"e2e_{args.variant}_cont"),
                            f"{args.variant}/continue:seg2")
        names = stage1["saved"] + stage2["saved"]

    for filename, subfolder in names:
        out = args.out_dir / filename
        out.write_bytes(fetch_output(args.server, filename, subfolder))
        validate_wav(out, EXPECT_SR[args.variant])
    print(f"[e2e] PASS {args.variant}/{args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
