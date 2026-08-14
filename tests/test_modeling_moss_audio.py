import math

import torch

from vendor.moss_audio.modeling_moss_audio import SinusoidsPositionEmbedding


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
