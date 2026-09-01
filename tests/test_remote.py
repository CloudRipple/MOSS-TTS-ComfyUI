"""Remote (mosi.cn) node tests — no network: requests is monkeypatched."""

import io
import json

import pytest
import torch

import mosstts_v15.remote as remote


class _FakeResp:
    def __init__(self, status=200, payload=None, json_data=None, content_type="application/json"):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload
        self._json = json_data
        self.text = payload.decode("utf8", "replace") if isinstance(payload, bytes) else ""

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json

    @property
    def content(self):
        return self._payload or b""

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


def _wav_bytes(sr=44100, seconds=0.5):
    import soundfile as sf
    import numpy as np

    buf = io.BytesIO()
    sf.write(buf, np.zeros((int(sr * seconds), 1), dtype="float32"), sr,
             format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _mk_cfg(monkeypatch):
    monkeypatch.delenv("MOSS_API_KEY", raising=False)
    return remote.MossTTSRemoteConfig("https://api.test.cn/", "k3y",
                                      "moss-tts-1.5-flash", "wav", False)


def test_connect_requires_key(monkeypatch):
    monkeypatch.delenv("MOSS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        remote.MossTTSRemoteConfig("https://api.test.cn", "",
                                   "moss-tts-1.5-flash", "wav", False)


def test_connect_env_fallback(monkeypatch):
    monkeypatch.setenv("MOSS_API_KEY", "env-key")
    cfg = remote.MossTTSRemoteConfig("https://api.test.cn", "",
                                     "moss-tts-1.5-flash", "wav", False)
    assert cfg.api_key == "env-key"


def test_speech_sync_payload_and_audio(monkeypatch):
    cfg = _mk_cfg(monkeypatch)
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        seen.update(url=url, json=json, headers=headers)
        return _FakeResp(payload=_wav_bytes(), content_type="audio/wav")

    monkeypatch.setattr(remote.requests, "post", fake_post)
    node = remote.MossTTSRemoteSpeech()
    (audio,) = node.run(cfg, "hello", "voice-1")
    assert audio["sample_rate"] == 44100
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[-1] == 22050
    assert seen["url"] == "https://api.test.cn/v1/audio/speech"
    assert seen["json"]["model"] == "moss-tts-1.5-flash"
    assert seen["json"]["delivery_method"] == "audio"
    assert seen["headers"]["Authorization"] == "Bearer k3y"


def test_speech_async_polls_task(monkeypatch):
    cfg = _mk_cfg(monkeypatch)
    cfg.use_async = True
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        assert json["async"] is True and json["delivery_method"] == "url"
        return _FakeResp(json_data={"id": "t1", "task_id": "t1", "status": "PENDING"})

    def fake_get(url, headers=None, params=None, timeout=None, **kw):
        calls.append(url)
        if url.endswith("/v1/audio/tasks/t1"):
            return _FakeResp(json_data={
                "status": "SUCCESS", "url": "https://cdn.test/o.wav",
                "response_format": "wav"})
        if url == "https://cdn.test/o.wav":
            return _FakeResp(payload=_wav_bytes())
        raise AssertionError(url)

    monkeypatch.setattr(remote.requests, "post", fake_post)
    monkeypatch.setattr(remote.requests, "get", fake_get)
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    node = remote.MossTTSRemoteSpeech()
    (audio,) = node.run(cfg, "hello", "voice-1")
    assert audio["waveform"].shape[-1] == 22050
    assert any("tasks/t1" in c for c in calls)


def test_speech_rejects_empty_inputs(monkeypatch):
    cfg = _mk_cfg(monkeypatch)
    node = remote.MossTTSRemoteSpeech()
    with pytest.raises(ValueError, match="text"):
        node.run(cfg, "  ", "voice-1")
    with pytest.raises(ValueError, match="voice_id"):
        node.run(cfg, "hello", "")


def test_clone_voice_multipart(monkeypatch):
    cfg = _mk_cfg(monkeypatch)
    seen = {}

    def fake_post(url, files=None, data=None, json=None, headers=None, timeout=None):
        seen.update(url=url, files=files, data=data)
        return _FakeResp(json_data={"id": "voice-9", "object": "audio.voice"})

    monkeypatch.setattr(remote.requests, "post", fake_post)
    node = remote.MossTTSRemoteVoiceClone()
    ref = {"waveform": torch.zeros(1, 1, 8000), "sample_rate": 16000}
    voice_id, preview = node.run(cfg, ref, "Alice", "")
    assert voice_id == "voice-9"
    assert preview is ref
    assert seen["url"].endswith("/v1/audio/voices")
    fname, blob, ctype = seen["files"]["audio_sample"]
    assert fname.endswith(".wav") and ctype == "audio/wav"
    # uploaded payload is a real wav readable by soundfile
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    assert sr == 16000 and len(data) == 8000


def test_voice_list_formatting(monkeypatch):
    cfg = _mk_cfg(monkeypatch)

    def fake_get(url, headers=None, params=None, timeout=None):
        assert params["status"] == "ready"
        return _FakeResp(json_data={"data": [
            {"id": "v1", "name": "Alice"}, {"id": "v2", "name": ""}]})

    monkeypatch.setattr(remote.requests, "get", fake_get)
    node = remote.MossTTSRemoteVoiceList()
    (out,) = node.run(cfg, 50)
    assert out.splitlines() == ["Alice\tv1", "(未命名)\tv2"]


def test_error_mapping_surfaces_server_message(monkeypatch):
    cfg = _mk_cfg(monkeypatch)
    monkeypatch.setattr(remote.requests, "post",
                        lambda *a, **kw: _FakeResp(status=401, json_data={"error": "bad key"}))
    node = remote.MossTTSRemoteSpeech()
    with pytest.raises(remote.RemoteError, match="bad key"):
        node.run(cfg, "hello", "voice-1")
