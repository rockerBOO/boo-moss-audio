import sys

import pytest
import torch

nodes = sys.modules["boo_moss_audio_nodes"]
BooMossAudioLoader = nodes.BooMossAudioLoader
BooMossAudioGenerate = nodes.BooMossAudioGenerate

import comfy.model_management as model_management

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_quantized_moss_audio_survives_load_offload_reload_generate_cycle():
    if not model_management.supports_nvfp4_compute(model_management.get_torch_device()):
        pytest.skip("requires a Blackwell GPU (SM >= 10.0)")

    load_device = model_management.get_torch_device()

    loader_output = BooMossAudioLoader.execute(
        model="MOSS-Audio-8B-Thinking", enable_time_marker=True, quantized=True
    )
    moss_audio_model = loader_output.args[0]
    patcher = moss_audio_model["patcher"]

    model_management.load_models_gpu([patcher])
    assert next(patcher.model.parameters()).device.type == load_device.type

    model_management.unload_model_and_clones(patcher, all_devices=True)
    assert next(patcher.model.parameters()).device.type == patcher.offload_device.type

    sample_rate = 16000
    silence = torch.zeros(1, 1, sample_rate * 2)
    audio = {"waveform": silence, "sample_rate": sample_rate}

    output = BooMossAudioGenerate.execute(
        moss_audio_model=moss_audio_model,
        audio=audio,
        prompt="Describe this audio in one sentence.",
        max_new_tokens=32,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        seed=0,
        strip_thinking=True,
    )

    text = output.args[0]
    assert isinstance(text, str)
    assert len(text) > 0
