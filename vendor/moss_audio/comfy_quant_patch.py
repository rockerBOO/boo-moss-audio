"""Patches ComfyUI's quantized-op kernels into an existing (already-constructed)
MossAudioModel module tree, in place.

comfy.ops.mixed_precision_ops(...).Linear is the only ComfyUI Linear replacement that
actually round-trips `<key>.comfy_quant` metadata into a real QuantizedTensor -- confirmed
against comfy.ops.manual_cast.Linear, which looks like the obvious choice but silently
falls through to a plain (non-quantized) load, producing nonsense output with no error.
See docs/superpowers/specs/2026-08-24-moss-audio-8b-thinking-nvfp4-convrot-design.md in
the quant-tooling repo for the spike that established this.
"""

import torch
import torch.nn as nn

import comfy.ops as comfy_ops


def _split_module_path(model: nn.Module, dotted_name: str):
    *parent_parts, attr = dotted_name.split(".")
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)
    return parent, attr


def patch_quantized_linears(
    model: nn.Module,
    quantized_state_dict: dict,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> list:
    """Replace every nn.Linear submodule of `model` that has a matching
    `<name>.comfy_quant` key in `quantized_state_dict` with
    comfy.ops.mixed_precision_ops(...).Linear, in place.

    Submodules with no matching comfy_quant key are left as plain nn.Linear, to be
    loaded at source precision from the same state dict elsewhere. Returns the dotted
    module names that were patched.
    """
    quant_ops = comfy_ops.mixed_precision_ops({}, compute_dtype=compute_dtype)
    patched = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if f"{name}.comfy_quant" not in quantized_state_dict:
            continue

        parent, attr = _split_module_path(model, name)
        replacement = quant_ops.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=device,
            dtype=compute_dtype,
        )
        setattr(parent, attr, replacement)
        patched.append(name)

    return patched
