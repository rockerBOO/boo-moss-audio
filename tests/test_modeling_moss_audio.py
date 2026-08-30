import math

import torch
from transformers import Qwen3Config

from vendor.moss_audio.configuration_moss_audio import MossAudioConfig
from vendor.moss_audio.modeling_moss_audio import MossAudioModel, SinusoidsPositionEmbedding


def _expected_pos_emb(seq_len: int, embedding_dim: int) -> torch.Tensor:
    log_timescale_increment = math.log(10000.0) / (embedding_dim // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(embedding_dim // 2).float())
    scaled_time = torch.arange(seq_len).float().unsqueeze(1) * inv_timescales.unsqueeze(0)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).unsqueeze(0)


def test_forward_output_is_finite_and_matches_expected_formula():
    embed = SinusoidsPositionEmbedding(num_positions=1500, embedding_dim=1280)

    out = embed.forward(seq_len=50, device=torch.device("cpu"))

    assert torch.isfinite(out).all()
    assert torch.allclose(out, _expected_pos_emb(50, 1280), atol=1e-5)


def test_repeated_calls_are_stable_regardless_of_module_state():
    # inv_timescales used to be cached as a `persistent=False` buffer,
    # which left it dependent on whatever weight-loading path materialized
    # the model onto its target device -- one that intermittently produced
    # garbage instead of the real computed values (see modeling_moss_audio.py
    # SinusoidsPositionEmbedding for the full story). Recomputing it fresh
    # every call, from module attributes that never leave the module, is
    # what actually guarantees this doesn't happen again -- verify calling
    # forward() repeatedly, including after moving the module, never
    # produces anything but the correct output.
    embed = SinusoidsPositionEmbedding(num_positions=1500, embedding_dim=1280)
    embed.to(torch.device("cpu"))

    for _ in range(3):
        out = embed.forward(seq_len=50, device=torch.device("cpu"))
        assert torch.isfinite(out).all()
        assert torch.allclose(out, _expected_pos_emb(50, 1280), atol=1e-5)


def test_no_registered_buffers_or_parameters():
    # Regression guard: the fix's entire point is that this module must not
    # cache inv_timescales as an nn.Module buffer/parameter, since that's
    # exactly the mechanism that let it go uninitialized. If a future change
    # reintroduces a buffer here, this test should catch it.
    embed = SinusoidsPositionEmbedding(num_positions=1500, embedding_dim=1280)

    assert list(embed.buffers()) == []
    assert list(embed.parameters()) == []


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


def test_forward_with_logits_to_keep_only_computes_last_position_logits():
    """transformers' GenerationMixin.generate() sets logits_to_keep=1 during
    prefill so models only materialize lm_head logits for the last token
    position, instead of the whole (often huge, audio-token-heavy) input
    sequence -- see generation/utils.py's _supports_logits_to_keep(), which
    gates this purely on the parameter being declared in forward()'s
    signature, not just accepted via **kwargs. Without an explicit
    `logits_to_keep` parameter, MossAudioModel materializes full-sequence
    logits on every prefill, which is what causes CUDA OOMs on long audio
    inputs (each audio second maps to a run of audio tokens)."""
    config = _tiny_config()
    model = MossAudioModel(config)
    model.eval()

    seq_len = 7
    input_ids = torch.randint(0, config.language_config.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = model(input_ids=input_ids, logits_to_keep=1, use_cache=False)

    assert out.logits.shape[1] == 1


def test_forward_default_logits_to_keep_still_computes_full_sequence():
    """labels-based training (shift_logits/shift_labels) needs full-sequence
    logits, so the default must remain 0 (== keep everything), matching
    transformers' own Qwen2ForCausalLM.forward() convention."""
    config = _tiny_config()
    model = MossAudioModel(config)
    model.eval()

    seq_len = 7
    input_ids = torch.randint(0, config.language_config.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    assert out.logits.shape[1] == seq_len
