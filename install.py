"""install.py — ComfyUI-Manager PreInstall hook: install only missing deps."""

import importlib.util
import subprocess
import sys

# pip name -> import name
DEPS = {
    "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors",
    "numpy": "numpy",
    "tqdm": "tqdm",
}

missing = [pip for pip, mod in DEPS.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"[ComfyUI-MOSS-TTS-v15] installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[ComfyUI-MOSS-TTS-v15] all dependencies already present.")
