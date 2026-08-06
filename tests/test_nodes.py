import sys

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
