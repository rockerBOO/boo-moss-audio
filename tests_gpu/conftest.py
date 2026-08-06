import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = PACKAGE_ROOT.parents[1]

if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Deliberately does NOT force comfy.cli_args.args.cpu = True, unlike
# tests/conftest.py -- these tests exercise real GPU load/offload/reload
# behavior and must see the real CUDA device. comfy.model_management caches
# VRAM/device state at import time for the whole process, so this suite
# cannot share a pytest process with tests/ (which forces CPU). Run this
# directory as its own `pytest tests_gpu/` invocation.

import importlib.util

_spec = importlib.util.spec_from_file_location("boo_moss_audio_nodes", PACKAGE_ROOT / "nodes.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules["boo_moss_audio_nodes"] = _module
_spec.loader.exec_module(_module)
