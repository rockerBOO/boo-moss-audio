import pytest
import torch
from transformers import Qwen3Config

from vendor.moss_audio.comfy_quant_patch import build_quantized_model
from vendor.moss_audio.configuration_moss_audio import MossAudioConfig
from vendor.moss_audio.modeling_moss_audio import MossAudioModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def _tiny_config():
    return MossAudioConfig(
        audio_config={
            "d_model": 32, "output_dim": 32, "num_mel_bins": 16,
            "encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": 64,
            "downsample_hidden_size": 32,
        },
        language_config=Qwen3Config(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, intermediate_size=128, vocab_size=100,
        ),
        adapter_hidden_size=32,
    )


def _quantize_all_self_attn_projs(config, reference_state_dict):
    """Fabricates comfy_quant-tagged int8_tensorwise tensors for every self_attn
    projection across both tiny decoder layers, reusing the reference model's real
    (random-initialized) weight *values* rather than zeros, so generate() exercises
    real (if untrained) numbers rather than degenerate all-zero activations."""
    sd = dict(reference_state_dict)
    for layer_idx in range(config.language_config.num_hidden_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"language_model.layers.{layer_idx}.self_attn.{proj}"
            w = sd.pop(f"{key}.weight").float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
            sd[f"{key}.weight"] = (w / scale).round().clamp(-127, 127).to(torch.int8)
            sd[f"{key}.weight_scale"] = scale
            sd[f"{key}.comfy_quant"] = torch.tensor(
                list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8
            )
    return sd


def test_generate_runs_with_quantized_self_attn_projections():
    config = _tiny_config()
    device = torch.device("cuda")

    reference = MossAudioModel(config).to(device)
    quantized_sd = _quantize_all_self_attn_projs(config, reference.state_dict())

    model, missing, unexpected = build_quantized_model(
        config, quantized_sd, compute_dtype=torch.bfloat16, device=device
    )
    assert missing == []
    assert unexpected == []
    model = model.to(device).eval()

    input_ids = torch.randint(0, config.language_config.vocab_size, (1, 4), device=device)
    with torch.no_grad():
        out = model.language_model(input_ids=input_ids)

    assert out.last_hidden_state.shape == (1, 4, config.language_config.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()
