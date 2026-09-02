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

# The package directory is named "boo-moss-audio" (hyphen), which is not a
# valid Python identifier, so submodules like music_caption can't be
# imported as "boo_moss_audio.music_caption" via a plain sys.path entry.
# Register a "boo_moss_audio" package alias in sys.modules pointing at this
# directory so tests can import it as if it were a normally named package.
if "boo_moss_audio" not in sys.modules:
    # Set up parent packages for proper relative imports
    _parent_pkg = sys.modules.get("comfy") or sys.modules.get("custom_nodes")
    _pkg_spec = importlib.util.spec_from_file_location(
        "boo_moss_audio", PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    _pkg_module = importlib.util.module_from_spec(_pkg_spec)
    sys.modules["boo_moss_audio"] = _pkg_module
    # Make relative imports work by registering parent packages
    if _parent_pkg is not None:
        parent_name = _parent_pkg.__name__
        if f"{parent_name}.boo_moss_audio" not in sys.modules:
            sys.modules[f"{parent_name}.boo_moss_audio"] = _pkg_module
    _pkg_spec.loader.exec_module(_pkg_module)
