"""Test configuration. Run pytest from the *parent* of a validly-named symlink
(or install dir) of the pack, e.g.:

    ln -s /path/to/ComfyUI-MOSS-TTS-v15 /some/dir/mosstts_v15
    cd /some/dir && python -m pytest mosstts_v15/tests/

so the pack root is imported as the proper package `mosstts_v15`
(ComfyUI itself doesn't care — it loads __init__.py via importlib).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
