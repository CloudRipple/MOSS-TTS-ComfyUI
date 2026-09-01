"""Remote (Moss platform / mosi.cn) API nodes for MOSS-TTS.

These nodes talk to the hosted /v1/audio API instead of loading local
weights. They cover what the platform actually offers:

- speech        -> POST /v1/audio/speech        (text + voice_id -> audio)
- voice clone   -> POST /v1/audio/voices        (multipart audio_sample -> voice id)
- voice list    -> GET  /v1/audio/voices        (discover voice ids)

Not available remotely (local-model only): prefix continuation, token-level
duration control (target_tokens), sampling knobs (temperature/top-p/top-k).
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Any, Optional

import requests
import torch

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

REMOTE_CONFIG_TYPE = "MOSS_TTS_REMOTE_CONFIG"

_DEFAULT_BASE_URL = "https://api.mosi.cn"
_MODELS = ["moss-tts-1.5-flash", "moss-tts-1.0-pro"]
_FORMATS = ["mp3", "wav"]

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT_SYNC = 300
_POLL_TIMEOUT = 600


class RemoteError(RuntimeError):
    """API call failed; message carries the server-provided reason when present."""


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _check(resp: requests.Response, what: str) -> None:
    if resp.ok:
        return
    detail = ""
    try:
        detail = resp.json().get("error") or resp.json().get("message") or ""
    except Exception:
        detail = resp.text[:500]
    raise RemoteError(f"{what} failed: HTTP {resp.status_code}: {detail}")


def _poll_task(base_url: str, api_key: str, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + _POLL_TIMEOUT
    delay = 2.0
    while True:
        resp = requests.get(
            f"{base_url}/v1/audio/tasks/{task_id}",
            headers=_headers(api_key),
            timeout=_CONNECT_TIMEOUT,
        )
        _check(resp, "task query")
        data = resp.json()
        status = str(data.get("status", "")).upper()
        if status == "SUCCESS":
            return data
        if status in ("FAILED", "FAILURE", "ERROR", "CANCELLED"):
            raise RemoteError(f"remote task {task_id} ended with status {status}: {data}")
        if time.monotonic() > deadline:
            raise RemoteError(f"remote task {task_id} did not finish within {_POLL_TIMEOUT}s")
        time.sleep(float(data.get("retry_after") or delay))
        delay = min(delay * 1.5, 15.0)


def _decode_audio(payload: bytes, ext: str = "") -> tuple[torch.Tensor, int]:
    """Audio bytes -> (waveform [1, C, T] float32, sample_rate)."""
    import soundfile as sf

    if ext and ext.lower() not in ("mp3", "wav", "flac", "ogg", "m4a"):
        raise RemoteError(f"unsupported audio format from server: {ext}")
    data, samplerate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [C, T]
    return wav.unsqueeze(0), int(samplerate)


class MossTTSRemoteConfig:
    """Parsed connection settings carried between remote nodes."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 response_format: str, use_async: bool):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("MOSS_API_KEY", "")
        self.model = model
        self.response_format = response_format
        self.use_async = use_async
        if not self.api_key:
            raise ValueError(
                "api_key 为空：请在节点里填写，或设置环境变量 MOSS_API_KEY。"
            )

    # --- API calls -----------------------------------------------------

    def speech(self, text: str, voice_id: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "input": text,
            "voice_id": voice_id,
            "response_format": self.response_format,
        }
        if self.use_async:
            body["async"] = True
            body["delivery_method"] = "url"
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech", json=body,
                headers=_headers(self.api_key),
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SYNC),
            )
            _check(resp, "speech submit")
            task = resp.json()
            return _poll_task(self.base_url, self.api_key,
                              task.get("task_id") or task["id"])
        # sync: ask for the audio binary directly
        body["delivery_method"] = "audio"
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech", json=body,
            headers=_headers(self.api_key),
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SYNC),
        )
        _check(resp, "speech")
        return {"__audio_bytes__": resp.content,
                "__format__": self.response_format}

    def clone_voice(self, wav_bytes: bytes, name: str, description: str) -> str:
        files = {"audio_sample": ("sample.wav", wav_bytes, "audio/wav")}
        data: dict[str, str] = {}
        if name.strip():
            data["name"] = name.strip()
        if description.strip():
            data["description"] = description.strip()
        resp = requests.post(
            f"{self.base_url}/v1/audio/voices", files=files, data=data,
            headers=_headers(self.api_key),
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SYNC),
        )
        _check(resp, "voice clone")
        return resp.json()["id"]

    def list_voices(self, limit: int = 50) -> list[dict[str, Any]]:
        resp = requests.get(
            f"{self.base_url}/v1/audio/voices",
            params={"limit": limit, "status": "ready"},
            headers=_headers(self.api_key),
            timeout=_CONNECT_TIMEOUT,
        )
        _check(resp, "voice list")
        return resp.json().get("data", [])


# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------

class MossTTSRemoteConnect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": _DEFAULT_BASE_URL,
                                        "tooltip": "API 根地址，一般保持默认；兼容端点则改成对应地址"}),
                "api_key": ("STRING", {"default": "",
                                       "tooltip": "平台控制台「API 密钥」页生成。留空则读环境变量 MOSS_API_KEY。注意：填写后会随工作流 JSON 保存，分享工作流前请清空。"}),
                "model": (_MODELS, {"default": _MODELS[0],
                                    "tooltip": "1.5-flash 支持 [pause X.Ys] 内联停顿标记"}),
                "response_format": (_FORMATS, {"default": "wav",
                                               "tooltip": "wav 无损；mp3 体积小"}),
                "async_mode": ("BOOLEAN", {"default": False,
                                           "tooltip": "长文本开异步：先建任务后轮询结果，避免同步超时"}),
            }
        }

    RETURN_TYPES = (REMOTE_CONFIG_TYPE,)
    RETURN_NAMES = ("remote",)
    FUNCTION = "connect"
    CATEGORY = "MOSS-TTS v1.5/Remote"
    DESCRIPTION = "Moss 平台（mosi.cn）远程 API 连接配置。"

    def connect(self, base_url, api_key, model, response_format, async_mode):
        return (MossTTSRemoteConfig(base_url, api_key, model,
                                    response_format, bool(async_mode)),)


class MossTTSRemoteSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote": (REMOTE_CONFIG_TYPE, {}),
                "text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "要合成的话；1.5-flash 支持 [pause 1.5s] 停顿标记"}),
                "voice_id": ("STRING", {"default": "",
                                        "tooltip": "音色 ID（只接受 id 字符串）。用 Voice List 节点查，或 Voice Clone 节点产出"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "MOSS-TTS v1.5/Remote"
    DESCRIPTION = "远程单人 TTS（/v1/audio/speech）。续写/时长控制/采样参数为本地模型专属能力。"

    def run(self, remote: MossTTSRemoteConfig, text: str, voice_id: str):
        text = (text or "").strip()
        voice_id = (voice_id or "").strip()
        if not text:
            raise ValueError("text 为空")
        if not voice_id:
            raise ValueError("voice_id 为空：远程接口必须指定音色 id")
        result = remote.speech(text, voice_id)
        if "__audio_bytes__" in result:
            wav, sr = _decode_audio(result["__audio_bytes__"], result["__format__"])
        else:
            url = result.get("url")
            if not url:
                raise RemoteError(f"任务完成但响应里没有 url: {result}")
            resp = requests.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SYNC))
            resp.raise_for_status()
            wav, sr = _decode_audio(resp.content, str(result.get("response_format", "")))
        return ({"waveform": wav, "sample_rate": sr},)


class MossTTSRemoteVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote": (REMOTE_CONFIG_TYPE, {}),
                "reference_audio": ("AUDIO", {"tooltip": "参考音频，建议 5-15 秒清晰人声"}),
                "name": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("voice_id", "preview_reference")
    FUNCTION = "run"
    CATEGORY = "MOSS-TTS v1.5/Remote"
    DESCRIPTION = "用参考音频在平台上创建克隆音色，返回 voice_id（连到 Speech 节点）。"

    def run(self, remote: MossTTSRemoteConfig, reference_audio: dict,
            name: str, description: str):
        wav = reference_audio["waveform"]
        sr = int(reference_audio["sample_rate"])
        import soundfile as sf

        buf = io.BytesIO()
        data = wav[0].T.detach().cpu().numpy()  # [T, C]
        sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
        voice_id = remote.clone_voice(buf.getvalue(), name, description)
        logger.info("[MOSS-TTS] remote voice cloned: %s", voice_id)
        return (voice_id, reference_audio)


class MossTTSRemoteVoiceList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"remote": (REMOTE_CONFIG_TYPE, {}),
                             "limit": ("INT", {"default": 50, "min": 1, "max": 150})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voices",)
    FUNCTION = "run"
    CATEGORY = "MOSS-TTS v1.5/Remote"
    DESCRIPTION = "列出当前账号可用音色（name + voice id），供 Speech 节点挑 voice_id。"

    def run(self, remote: MossTTSRemoteConfig, limit: int):
        voices = remote.list_voices(limit)
        lines = [f"{v.get('name') or '(未命名)'}\t{v['id']}" for v in voices]
        return ("\n".join(lines) or "(无音色)",)


REMOTE_NODE_CLASS_MAPPINGS = {
    "MossTTSV15_RemoteConnect": MossTTSRemoteConnect,
    "MossTTSV15_RemoteSpeech": MossTTSRemoteSpeech,
    "MossTTSV15_RemoteVoiceClone": MossTTSRemoteVoiceClone,
    "MossTTSV15_RemoteVoiceList": MossTTSRemoteVoiceList,
}

REMOTE_NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSV15_RemoteConnect": "MOSS-TTS Remote Connect",
    "MossTTSV15_RemoteSpeech": "MOSS-TTS Remote Speech",
    "MossTTSV15_RemoteVoiceClone": "MOSS-TTS Remote Voice Clone",
    "MossTTSV15_RemoteVoiceList": "MOSS-TTS Remote Voice List",
}
