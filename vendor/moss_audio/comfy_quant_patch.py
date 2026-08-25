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
from accelerate import init_empty_weights

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


def build_quantized_model(config, quantized_state_dict, compute_dtype, device):
    """Construct a MossAudioModel with all *parameters* on the meta device (no real
    weight tensors allocated), patch its comfy_quant-tagged Linear submodules per
    patch_quantized_linears, then load the full state dict with assign=True -- binding
    each provided tensor directly as that submodule's parameter/buffer rather than
    copying into a pre-allocated one (meta-device parameters have no real storage to
    copy into).

    Uses accelerate.init_empty_weights() rather than a bare `with torch.device("meta")`
    block: the latter also meta-izes *buffers*, including non-persistent ones (e.g.
    rotary-embedding inv_freq, the audio encoder's inv_timescales) that are deliberately
    excluded from state_dict()/checkpoints because they're deterministic functions of
    config, not learned data. Those buffers would then have no checkpoint key to ever
    load real data from, leaving them permanently stuck on the meta device -- surfaced
    by a `NotImplementedError: Cannot copy out of meta tensor; no data!` the first time
    anything (e.g. a later `.to(device)`, or a forward pass touching that buffer) tried
    to materialize them. init_empty_weights() defaults to include_buffers=False, so
    buffers are constructed normally (real, on the current default device) while only
    parameters go to meta.
    """
    from .modeling_moss_audio import MossAudioModel

    with init_empty_weights():
        model = MossAudioModel(config)

    patch_quantized_linears(model, quantized_state_dict, compute_dtype, device)

    missing, unexpected = model.load_state_dict(
        quantized_state_dict, strict=False, assign=True
    )
    return model, list(missing), list(unexpected)
