import sys

import pytest
import torch

nodes = sys.modules["boo_moss_audio_nodes"]
BooMossAudioLoader = nodes.BooMossAudioLoader
BooMossAudioGenerate = nodes.BooMossAudioGenerate

import comfy.model_management as model_management

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_moss_audio_survives_load_offload_reload_generate_cycle():
    """Empirically verifies MOSS-Audio's HF modules (rotary embedding caches,
    KV-cache dtype, etc.) tolerate being moved between devices post-
    construction by ModelPatcher -- the one thing code review alone can't
    confirm (see docs/comfy-model-management-integration.md, step 7)."""

    load_device = model_management.get_torch_device()

    loader_output = BooMossAudioLoader.execute(
        model="MOSS-Audio-4B-Instruct", enable_time_marker=True
    )
    moss_audio_model = loader_output.args[0]
    patcher = moss_audio_model["patcher"]

    # 1. Load onto the GPU and confirm residency.
    model_management.load_models_gpu([patcher])
    assert next(patcher.model.parameters()).device.type == load_device.type
    assert all(p.device.type == load_device.type for p in patcher.model.parameters())

    # 2. Force eviction back to the offload device.
    model_management.unload_model_and_clones(patcher, all_devices=True)
    assert next(patcher.model.parameters()).device.type == patcher.offload_device.type
    assert all(p.device.type == patcher.offload_device.type for p in patcher.model.parameters())

    # 3. Reload and run a real end-to-end generate() call.
    sample_rate = 16000
    silence = torch.zeros(1, 1, sample_rate * 2)  # 2s mono silence
    audio = {"waveform": silence, "sample_rate": sample_rate}

    output = BooMossAudioGenerate.execute(
        moss_audio_model=moss_audio_model,
        audio=audio,
        prompt="Describe this audio in one sentence.",
        max_new_tokens=32,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        strip_thinking=True,
    )

    text = output.args[0]
    assert isinstance(text, str)
    assert len(text) > 0
