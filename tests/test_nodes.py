import sys
from types import SimpleNamespace

import comfy.model_management
import comfy.model_patcher
import torch

nodes = sys.modules["boo_moss_audio_nodes"]
BooMossAudioLoader = nodes.BooMossAudioLoader
BooMossAudioGenerate = nodes.BooMossAudioGenerate
BooMossAudioMiniMaxMusic3PromptGenerate = nodes.BooMossAudioMiniMaxMusic3PromptGenerate
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
_STYLE_RE = nodes._STYLE_RE


def test_lyrics_and_style_regexes_split_both_labeled_sections():
    text = (
        "LYRICS:\n"
        "[verse]\n"
        "Lights are low\n"
        "STYLE:\n"
        "Global Metadata\n"
        "A mellow synth-pop track."
    )
    lyrics_match = _LYRICS_RE.search(text)
    style_match = _STYLE_RE.search(text)
    assert lyrics_match.group(1).strip() == "[verse]\nLights are low"
    assert style_match.group(1).strip() == "Global Metadata\nA mellow synth-pop track."


def test_lyrics_regex_returns_none_when_only_style_present():
    text = "STYLE:\nA mellow synth-pop track."
    assert _LYRICS_RE.search(text) is None
    assert _STYLE_RE.search(text).group(1).strip() == "A mellow synth-pop track."


def test_style_regex_returns_none_when_only_lyrics_present():
    text = "LYRICS:\n[verse]\nLights are low"
    assert _LYRICS_RE.search(text).group(1).strip() == "[verse]\nLights are low"
    assert _STYLE_RE.search(text) is None


def test_both_regexes_return_none_when_neither_label_present():
    text = "The model produced plain unlabeled text."
    assert _LYRICS_RE.search(text) is None
    assert _STYLE_RE.search(text) is None


def test_lyrics_regex_handles_empty_section_between_labels():
    text = "LYRICS:\nSTYLE:\nSome caption text."
    assert _LYRICS_RE.search(text).group(1).strip() == ""
    assert _STYLE_RE.search(text).group(1).strip() == "Some caption text."


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


def test_run_moss_audio_generate_assistant_prefill_wraps_prompt_and_reattaches_label(
    monkeypatch,
):
    patcher = _FakePatcher(
        SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
        )
    )
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)

    captured = {}

    class _CapturingProcessor(_FakeProcessor):
        def __call__(self, text, audios, return_tensors):
            captured["text"] = text
            return super().__call__(text, audios, return_tensors)

        def decode(self, ids, skip_special_tokens=True):
            return "[verse]\nLights are low"

    text = nodes._run_moss_audio_generate(
        moss_audio_model={"patcher": patcher, "processor": _CapturingProcessor()},
        **_generate_kwargs(assistant_prefill="LYRICS:\n"),
    )

    # The prompt sent to the processor must embed the audio span itself so
    # MossAudioProcessor skips its own chat-template auto-wrap and uses our
    # assistant-seeded prompt verbatim.
    assert "<|audio_bos|><|AUDIO|><|audio_eos|>" in captured["text"]
    assert captured["text"].endswith("<|im_start|>assistant\nLYRICS:\n")
    # The prefill isn't part of what the model generated, so it's reattached
    # to the decoded output here rather than expected from decode() itself.
    assert text == "LYRICS:\n[verse]\nLights are low"


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


def test_minimax_prompt_generate_schema_has_two_string_outputs():
    schema = BooMossAudioMiniMaxMusic3PromptGenerate.define_schema()
    inputs_by_id = {i.id: i for i in schema.inputs}
    assert inputs_by_id["moss_audio_model"].io_type == "BOO_MOSS_AUDIO_MODEL"
    assert inputs_by_id["audio"].io_type == "AUDIO"
    assert [o.io_type for o in schema.outputs] == ["STRING", "STRING"]
    assert [o.id for o in schema.outputs] == ["lyrics", "style_notes"]


def test_minimax_prompt_generate_splits_labeled_output_into_two_results(monkeypatch):
    patcher = _FakePatcher(
        SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
        )
    )
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)

    class _LabeledProcessor(_FakeProcessor):
        def decode(self, ids, skip_special_tokens=True):
            # The "LYRICS:\n" label itself is supplied by the assistant-turn
            # prefill, not generated -- decode() only returns what comes after it.
            return "[verse]\nLights are low\nSTYLE:\nA mellow synth-pop track."

    output = BooMossAudioMiniMaxMusic3PromptGenerate.execute(
        moss_audio_model={"patcher": patcher, "processor": _LabeledProcessor()},
        **_generate_kwargs(),
    )

    assert output.args[0] == "[verse]\nLights are low"
    assert output.args[1] == "A mellow synth-pop track."


def test_minimax_prompt_generate_returns_empty_string_for_missing_label(monkeypatch):
    patcher = _FakePatcher(
        SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
        )
    )
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)

    class _StyleOnlyProcessor(_FakeProcessor):
        def decode(self, ids, skip_special_tokens=True):
            return "STYLE:\nA mellow synth-pop track."

    output = BooMossAudioMiniMaxMusic3PromptGenerate.execute(
        moss_audio_model={"patcher": patcher, "processor": _StyleOnlyProcessor()},
        **_generate_kwargs(),
    )

    assert output.args[0] == ""
    assert output.args[1] == "A mellow synth-pop track."


def test_minimax_prompt_generate_prefill_guarantees_lyrics_label_even_when_style_label_is_missing(
    monkeypatch,
):
    patcher = _FakePatcher(
        SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
        )
    )
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)

    class _UnlabeledProcessor(_FakeProcessor):
        def decode(self, ids, skip_special_tokens=True):
            return "This is a mellow synth-pop track with a dreamy chorus."

    output = BooMossAudioMiniMaxMusic3PromptGenerate.execute(
        moss_audio_model={"patcher": patcher, "processor": _UnlabeledProcessor()},
        **_generate_kwargs(),
    )

    # The assistant-turn prefill ("LYRICS:\n") guarantees the LYRICS: label always
    # exists, so a model that skips straight to unlabeled prose lands in `lyrics`
    # rather than vanishing into two empty outputs -- there's just no way to tell
    # it apart from real lyrics without the STYLE: boundary.
    assert output.args[0] == "This is a mellow synth-pop track with a dreamy chorus."
    assert output.args[1] == ""


def test_minimax_prompt_generate_is_registered_in_extension_node_list():
    import asyncio

    extension = nodes.BooMossAudioExtension()
    node_list = asyncio.run(extension.get_node_list())
    assert BooMossAudioMiniMaxMusic3PromptGenerate in node_list
