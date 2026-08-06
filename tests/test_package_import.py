"""Reproduces ComfyUI's actual custom-node loading mechanism (see
`load_custom_node` in ComfyUI's own nodes.py), which never adds this
package's directory to sys.path -- only tests/conftest.py does that, for the
test suite only. This guards against imports that happen to work under
pytest (via conftest.py's sys.path insertion) but would break under
ComfyUI's real loader.
"""

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_init_py_imports_via_comfyui_custom_node_loader_without_sys_path_entry():
    module_path = str(PACKAGE_ROOT)
    sys_module_name = module_path.replace(".", "_x_")

    saved_sys_path = list(sys.path)
    saved_sys_modules = dict(sys.modules)
    sys.path = [p for p in sys.path if p not in ("", str(PACKAGE_ROOT))]
    for name in ("nodes", "boo_moss_audio_nodes", sys_module_name):
        sys.modules.pop(name, None)

    try:
        module_spec = importlib.util.spec_from_file_location(
            sys_module_name, str(PACKAGE_ROOT / "__init__.py")
        )
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)  # must not raise ModuleNotFoundError

        assert hasattr(module, "comfy_entrypoint")
    finally:
        sys.path = saved_sys_path
        sys.modules.clear()
        sys.modules.update(saved_sys_modules)
