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


def _load_without_sys_path_entry(sys_module_name: str):
    """Load __init__.py the way ComfyUI's real loader does: without ever
    adding PACKAGE_ROOT to sys.path. Caller must restore sys.path/sys.modules.
    """
    sys.path = [p for p in sys.path if p not in ("", str(PACKAGE_ROOT))]
    for name in ("nodes", "boo_moss_audio_nodes", sys_module_name):
        sys.modules.pop(name, None)

    module_spec = importlib.util.spec_from_file_location(
        sys_module_name, str(PACKAGE_ROOT / "__init__.py")
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[sys_module_name] = module
    module_spec.loader.exec_module(module)  # must not raise ModuleNotFoundError
    return module


def test_init_py_imports_via_comfyui_custom_node_loader_without_sys_path_entry():
    sys_module_name = str(PACKAGE_ROOT).replace(".", "_x_")

    saved_sys_path = list(sys.path)
    saved_sys_modules = dict(sys.modules)
    try:
        module = _load_without_sys_path_entry(sys_module_name)
        assert hasattr(module, "comfy_entrypoint")
    finally:
        sys.path = saved_sys_path
        sys.modules.clear()
        sys.modules.update(saved_sys_modules)


async def test_music_caption_rewriter_execute_without_sys_path_entry():
    """Regression test: nodes.py's execute() does `from .music_caption import
    CaptionRewriter` inside the method body (deferred import), so it's never
    exercised by importing __init__.py alone. Under ComfyUI's real loader
    (PACKAGE_ROOT never added to sys.path), an absolute `from music_caption
    import ...` anywhere in the call chain raises ModuleNotFoundError here
    even though it silently works under pytest via conftest.py's sys.path hack.

    This test itself runs inside pytest-asyncio's event loop, matching how
    ComfyUI's execution.py awaits node execute() methods from within its own
    running loop -- so it also catches execute() internally calling
    asyncio.run(), which raises "cannot be called from a running event loop"
    in that context (execute() must be `async def` and `await` its work).
    """
    sys_module_name = (str(PACKAGE_ROOT) + "_execute").replace(".", "_x_")

    saved_sys_path = list(sys.path)
    saved_sys_modules = dict(sys.modules)
    try:
        module = _load_without_sys_path_entry(sys_module_name)
        extension = await module.comfy_entrypoint()
        node_classes = await extension.get_node_list()
        rewriter_cls = next(
            n for n in node_classes if n.__name__ == "BooMusicCaptionRewriter"
        )

        def fake_generate(*args, **kwargs):
            return "Macro genre: pop"

        result = await rewriter_cls.execute({"generate": fake_generate}, "pop", "")
        assert result is not None
    finally:
        sys.path = saved_sys_path
        sys.modules.clear()
        sys.modules.update(saved_sys_modules)
