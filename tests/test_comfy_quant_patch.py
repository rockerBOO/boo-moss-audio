import torch
import torch.nn as nn
from transformers import Qwen3Config

from vendor.moss_audio.comfy_quant_patch import build_quantized_model, patch_quantized_linears
from vendor.moss_audio.configuration_moss_audio import MossAudioConfig
from vendor.moss_audio.modeling_moss_audio import MossAudioModel


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


def _tiny_config():
    return MossAudioConfig(
        audio_config={
            "d_model": 32, "output_dim": 32, "num_mel_bins": 16,
            "encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": 64,
            "downsample_hidden_size": 32,
        },
        language_config=Qwen3Config(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, intermediate_size=64, vocab_size=100,
        ),
        adapter_hidden_size=32,
    )


def _fake_quantized_layer0_self_attn_state_dict(config):
    """One real decoder layer's self_attn projections quantized (fabricated
    int8_tensorwise tensors, not run through ctq -- Task 1's test already covers
    the swap mechanism; this covers the full-model load path), everything else
    left unquantized (plain bf16, as MossAudioModel's own __init__ initializes it)."""
    hidden = config.language_config.hidden_size
    sd = {}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        key = f"language_model.layers.0.self_attn.{proj}"
        sd[f"{key}.weight"] = torch.zeros(hidden, hidden, dtype=torch.int8)
        sd[f"{key}.weight_scale"] = torch.ones(hidden, 1)
        sd[f"{key}.comfy_quant"] = torch.tensor(
            list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8
        )
    return sd


def test_build_quantized_model_patches_and_loads_without_missing_or_unexpected():
    config = _tiny_config()
    quantized_sd = _fake_quantized_layer0_self_attn_state_dict(config)

    # A meta-constructed model has every other tensor as a meta placeholder; assign=True
    # only succeeds if *every* remaining key is also provided. Build the rest of the
    # state dict from a real (non-meta) instance of the same tiny config so every
    # non-quantized key has a real tensor to assign.
    reference = MossAudioModel(config)
    full_sd = dict(reference.state_dict())
    full_sd.update(quantized_sd)  # overwrites layer 0 self_attn's plain .weight keys
    # with the quantized weight/weight_scale/comfy_quant keys fabricated above.

    model, missing, unexpected = build_quantized_model(
        config, full_sd, compute_dtype=torch.bfloat16, device=torch.device("cpu")
    )

    assert missing == []
    assert unexpected == []
    assert model.language_model.layers[0].self_attn.q_proj.__class__ is not torch.nn.Linear
    assert model.language_model.layers[0].self_attn.k_proj.quant_format == "int8_tensorwise"
    # An unpatched Linear elsewhere (e.g. lm_head) must be untouched.
    assert model.lm_head.__class__ is torch.nn.Linear
