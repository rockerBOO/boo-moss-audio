import subprocess

import pytest
import torch
from safetensors.torch import load_file, save_file

from vendor.moss_audio.comfy_quant_patch import patch_quantized_linears

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

CTQ_CHECKOUT = "/home/rockerboo/code/others/convert_to_quant"


def _quantize_one_linear(tmp_path):
    torch.manual_seed(0)
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    bias = torch.randn(256, dtype=torch.bfloat16)
    src = tmp_path / "in.safetensors"
    out = tmp_path / "out.safetensors"
    save_file({"proj.weight": weight, "proj.bias": bias}, str(src))

    result = subprocess.run(
        [
            "uv", "run", "--with-editable", ".", "--with-requirements", "requirements.txt",
            "--with", "numpy", "--with", "scipy", "ctq",
            "-i", str(src), "-o", str(out),
            "--int8", "--scaling-mode", "row", "--dynamic-convrot", "--convrot-group-size", "256",
            "--comfy_quant", "--save-quant-metadata", "--device", "cuda", "--simple",
        ],
        cwd=CTQ_CHECKOUT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return weight, bias, load_file(str(out))


def test_patched_linear_forward_matches_bf16_reference_within_tolerance(tmp_path):
    weight, bias, quantized_sd = _quantize_one_linear(tmp_path)

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(512, 256)

    device = torch.device("cuda")
    model = _Holder()
    # quantized_sd's keys are already fully-qualified against `model` (ctq preserves
    # the "proj." prefix baked into the tensor names saved in _quantize_one_linear),
    # so they're used as-is here -- no re-prefixing, and the state dict is loaded
    # onto the top-level model rather than the "proj" submodule directly.
    patched = patch_quantized_linears(
        model, quantized_sd,
        compute_dtype=torch.bfloat16, device=device,
    )
    assert patched == ["proj"]

    model.load_state_dict(quantized_sd, strict=False)
    model = model.to(device)

    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.proj(x)
        out_ref = torch.nn.functional.linear(
            x, weight.to(device), bias.to(device)
        )

    rel_err = (out - out_ref).abs().mean() / out_ref.abs().mean()
    assert rel_err < 0.05  # INT8 ConvRot; ~1.2% observed in the design-time spike
