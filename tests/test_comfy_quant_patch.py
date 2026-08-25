import torch
import torch.nn as nn

from vendor.moss_audio.comfy_quant_patch import patch_quantized_linears


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)


def test_patch_quantized_linears_only_replaces_matching_keys():
    model = _TinyModel()
    fake_state_dict = {
        "a.weight": torch.zeros(8, 8, dtype=torch.int8),
        "a.weight_scale": torch.ones(8, 1),
        "a.comfy_quant": torch.zeros(1, dtype=torch.uint8),
        # "b" has no comfy_quant entry -- must stay a plain nn.Linear, untouched.
    }

    patched = patch_quantized_linears(
        model, fake_state_dict, compute_dtype=torch.bfloat16, device=torch.device("cpu")
    )

    assert patched == ["a"]
    assert model.a.__class__ is not nn.Linear
    assert hasattr(model.a, "factory_kwargs")  # marker unique to comfy's quantized Linear
    assert model.b.__class__ is nn.Linear


def test_patch_quantized_linears_returns_empty_list_when_nothing_matches():
    model = _TinyModel()
    patched = patch_quantized_linears(
        model, {}, compute_dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert patched == []
    assert model.a.__class__ is nn.Linear
    assert model.b.__class__ is nn.Linear
