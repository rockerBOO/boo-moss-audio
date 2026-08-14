import sys
from types import SimpleNamespace

import comfy.model_management
import comfy.model_patcher
import torch

nodes = sys.modules["boo_moss_audio_nodes"]
BooMossAudioLoader = nodes.BooMossAudioLoader
BooMossAudioGenerate = nodes.BooMossAudioGenerate
MOSS_AUDIO_REPOS = nodes.MOSS_AUDIO_REPOS
_local_model_dir = nodes._local_model_dir
_THINK_BLOCK_RE = nodes._THINK_BLOCK_RE


def test_loader_schema_exposes_all_four_model_variants():
    schema = BooMossAudioLoader.define_schema()
    inputs_by_id = {i.id: i for i in schema.inputs}
    assert set(inputs_by_id["model"].options) == set(MOSS_AUDIO_REPOS.keys())


def test_loader_schema_output_is_custom_moss_audio_type():
    schema = BooMossAudioLoader.define_schema()
    assert schema.outputs[0].io_type == "BOO_MOSS_AUDIO_MODEL"


def test_generate_schema_takes_moss_audio_model_and_audio_and_returns_string():
    schema = BooMossAudioGenerate.define_schema()
    inputs_by_id = {i.id: i for i in schema.inputs}
    assert inputs_by_id["moss_audio_model"].io_type == "BOO_MOSS_AUDIO_MODEL"
    assert inputs_by_id["audio"].io_type == "AUDIO"
    assert schema.outputs[0].io_type == "STRING"


def test_local_model_dir_uses_repo_name_only_not_org():
    local_dir = _local_model_dir("OpenMOSS-Team/MOSS-Audio-4B-Instruct")
    assert local_dir.endswith("MOSS-Audio-4B-Instruct")
    assert "OpenMOSS-Team" not in local_dir


def test_think_block_regex_strips_reasoning_but_keeps_the_answer():
    raw = "<think>reasoning about the audio</think>The clip is upbeat pop music."
    stripped = _THINK_BLOCK_RE.sub("", raw).strip()
    assert stripped == "The clip is upbeat pop music."


_LYRICS_RE = nodes._LYRICS_RE
_CAPTION_RE = nodes._CAPTION_RE


def test_lyrics_and_caption_regexes_split_both_labeled_sections():
    text = (
        "LYRICS:\n"
        "[verse]\n"
        "Lights are low\n"
        "STRUCTURED CAPTION:\n"
        "Global Metadata\n"
        "A mellow synth-pop track."
    )
    lyrics_match = _LYRICS_RE.search(text)
    caption_match = _CAPTION_RE.search(text)
    assert lyrics_match.group(1).strip() == "[verse]\nLights are low"
    assert caption_match.group(1).strip() == "Global Metadata\nA mellow synth-pop track."


def test_lyrics_regex_returns_none_when_only_caption_present():
    text = "STRUCTURED CAPTION:\nA mellow synth-pop track."
    assert _LYRICS_RE.search(text) is None
    assert _CAPTION_RE.search(text).group(1).strip() == "A mellow synth-pop track."


def test_caption_regex_returns_none_when_only_lyrics_present():
    text = "LYRICS:\n[verse]\nLights are low"
    assert _LYRICS_RE.search(text).group(1).strip() == "[verse]\nLights are low"
    assert _CAPTION_RE.search(text) is None


def test_both_regexes_return_none_when_neither_label_present():
    text = "The model produced plain unlabeled text."
    assert _LYRICS_RE.search(text) is None
    assert _CAPTION_RE.search(text) is None


def test_lyrics_regex_handles_empty_section_between_labels():
    text = "LYRICS:\nSTRUCTURED CAPTION:\nSome caption text."
    assert _LYRICS_RE.search(text).group(1).strip() == ""
    assert _CAPTION_RE.search(text).group(1).strip() == "Some caption text."


MossAudioModel = nodes.MossAudioModel
MossAudioProcessor = nodes.MossAudioProcessor


class _FakeHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_loader_wraps_model_in_model_patcher_with_pinned_dtype(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_local_model_dir", lambda repo_id: str(tmp_path))
    (tmp_path / "config.json").write_text("{}")

    fake_model = _FakeHFModel()
    captured = {}

    def fake_model_from_pretrained(local_dir, dtype):
        captured["local_dir"] = local_dir
        captured["dtype"] = dtype
        return fake_model

    def fake_processor_from_pretrained(local_dir, enable_time_marker):
        return object()

    monkeypatch.setattr(MossAudioModel, "from_pretrained", staticmethod(fake_model_from_pretrained))
    monkeypatch.setattr(
        MossAudioProcessor, "from_pretrained", staticmethod(fake_processor_from_pretrained)
    )

    output = BooMossAudioLoader.execute(model="MOSS-Audio-4B-Instruct", enable_time_marker=True)
    result = output.args[0]

    assert captured["local_dir"] == str(tmp_path)
    load_device = comfy.model_management.get_torch_device()
    expected_dtype = comfy.model_management.text_encoder_dtype(load_device)
    if expected_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        expected_dtype = torch.float16
    if expected_dtype == torch.float16 and comfy.model_management.should_use_bf16(load_device):
        expected_dtype = torch.bfloat16
    assert captured["dtype"] == expected_dtype

    assert set(result.keys()) == {"patcher", "processor", "model_id"}
    patcher = result["patcher"]
    assert isinstance(patcher, comfy.model_patcher.ModelPatcher)
    assert patcher.model is fake_model
    assert patcher.load_device == comfy.model_management.get_torch_device()
    assert patcher.offload_device == comfy.model_management.unet_offload_device()


class _FakePatcher:
    def __init__(self, model):
        self.model = model


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.audio_token_id = 999
        self.config = SimpleNamespace(mel_sr=16000)

    def __call__(self, text, audios, return_tensors):
        return _FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


def _fake_audio():
    return {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}


def _generate_kwargs(**overrides):
    kwargs = dict(
        audio=_fake_audio(),
        prompt="describe this",
        max_new_tokens=8,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        seed=0,
        strip_thinking=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_generate_registers_patcher_with_load_models_gpu_and_uses_patcher_model(monkeypatch):
    class FakeHFModel:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

    fake_model = FakeHFModel()
    patcher = _FakePatcher(fake_model)
    recorded_calls = []
    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda models, **kwargs: recorded_calls.append((models, kwargs)),
    )
    monkeypatch.setattr(comfy.model_management, "soft_empty_cache", lambda force=False: None)

    output = BooMossAudioGenerate.execute(
        moss_audio_model={"patcher": patcher, "processor": _FakeProcessor()},
        **_generate_kwargs(),
    )

    assert recorded_calls == [([patcher], {"force_full_load": True})]
    assert output.args[0] == "generated text"


def test_generate_cleans_up_without_unloading_model_on_success(monkeypatch):
    class FakeHFModel:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

    patcher = _FakePatcher(FakeHFModel())
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)

    gc_calls = []
    cache_calls = []
    unload_calls = []
    monkeypatch.setattr(nodes.gc, "collect", lambda: gc_calls.append(True))
    monkeypatch.setattr(
        comfy.model_management, "soft_empty_cache", lambda force=False: cache_calls.append(force)
    )
    monkeypatch.setattr(
        comfy.model_management,
        "unload_model_and_clones",
        lambda *a, **kw: unload_calls.append((a, kw)),
    )

    BooMossAudioGenerate.execute(
        moss_audio_model={"patcher": patcher, "processor": _FakeProcessor()},
        **_generate_kwargs(),
    )

    assert gc_calls == [True]
    assert cache_calls == [True]
    assert unload_calls == []


class _ReadOnlyDeviceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    @property
    def device(self):
        return next(self.parameters()).device


def test_make_device_settable_allows_assigning_device_after_patch():
    model = _ReadOnlyDeviceModel()
    original_device = model.device

    patched = nodes._make_device_settable(model)

    assert patched.device == original_device

    new_device = torch.device("cpu")
    patched.device = new_device
    assert patched.device == new_device


def test_generate_still_cleans_up_and_reraises_on_generate_failure(monkeypatch):
    class FailingHFModel:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            raise RuntimeError("boom")

    patcher = _FakePatcher(FailingHFModel())
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)

    gc_calls = []
    cache_calls = []
    monkeypatch.setattr(nodes.gc, "collect", lambda: gc_calls.append(True))
    monkeypatch.setattr(
        comfy.model_management, "soft_empty_cache", lambda force=False: cache_calls.append(force)
    )

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        BooMossAudioGenerate.execute(
            moss_audio_model={"patcher": patcher, "processor": _FakeProcessor()},
            **_generate_kwargs(),
        )

    assert gc_calls == [True]
    assert cache_calls == [True]
