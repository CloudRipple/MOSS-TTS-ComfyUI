"""ComfyUI-MOSS-TTS-v15 — OpenMOSS MOSS-TTS v1.5 (Local-Transformer + Delay).

Plain re-export so the ComfyUI registry's static node parser can discover the
nodes. No try/except guard here on purpose: import errors are caught and
logged by ComfyUI's own custom-node loader.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
