"""Test configuration. Run pytest from the *parent* of a validly-named symlink
(or install dir) of the pack, e.g.:

    ln -s /path/to/MOSS-TTS-ComfyUI /some/dir/moss_tts
    cd /some/dir && python -m pytest moss_tts/tests/

so the pack root is imported as the proper package `moss_tts`
(ComfyUI itself doesn't care — it loads __init__.py via importlib).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
