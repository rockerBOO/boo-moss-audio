import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = PACKAGE_ROOT.parents[1]

if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# comfy.model_management computes total_vram at import time by calling
# torch.cuda.current_device(), which raises on any machine with no NVIDIA
# driver (e.g. CPU-only CI runners) unless args.cpu is already set. Force it
# before nodes.py (transitively) imports comfy.model_management, since these
# tests exercise pure node logic and never need a real GPU.
import comfy.cli_args

comfy.cli_args.args.cpu = True

# ComfyUI's own top-level nodes.py is also importable as "nodes", and whichever
# one lands in sys.modules first under that name wins for every later `import
# nodes`. Load this package's nodes.py explicitly by path and register it
# under its own name so it can't be shadowed by, or shadow, ComfyUI's.
_spec = importlib.util.spec_from_file_location("boo_moss_audio_nodes", PACKAGE_ROOT / "nodes.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules["boo_moss_audio_nodes"] = _module
_spec.loader.exec_module(_module)
