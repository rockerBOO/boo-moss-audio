# boo-moss-audio

ComfyUI custom nodes for [OpenMOSS's MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio)
audio-understanding models — speech transcription, music/mood analysis, and
general audio captioning/QA in one model.

## Nodes

- **BOO MOSS-Audio Loader** — picks one of the four released checkpoints
  (`MOSS-Audio-4B-Instruct`, `MOSS-Audio-4B-Thinking`, `MOSS-Audio-8B-Instruct`,
  `MOSS-Audio-8B-Thinking`) and downloads it from Hugging Face on first use to
  `ComfyUI/models/moss-audio/<model-name>`. Outputs a `BOO_MOSS_AUDIO_MODEL`.
- **BOO MOSS-Audio Generate** — takes a `BOO_MOSS_AUDIO_MODEL`, a native
  ComfyUI `AUDIO` input, and a text prompt, and returns the model's text
  response as a `STRING`. The default prompt asks for explicitly labeled
  `LYRICS:`/`STYLE:` sections — plain requests like "transcribe the lyrics,
  then describe the mood" reliably get the model to transcribe the lyrics but
  drop the mood/style description entirely once it runs out of words to
  transcribe; forcing two required, labeled sections fixes that (verified
  empirically across both greedy and sampled decoding). Thinking-variant
  `<think>...</think>` reasoning blocks are stripped by default
  (`strip_thinking`).

Wire `BOO MOSS-Audio Generate`'s `STRING` output into any downstream
prompt-enhancement node (e.g. [boo-textgen](https://github.com/rockerboo/boo-textgen))
to feed a transcript/mood description into further prompt generation.

## Example workflow

`example_workflows/basic_caption.json` — `LoadAudio` → `BOO MOSS-Audio
Loader` + `BOO MOSS-Audio Generate` → `Preview as Text`. Load it in ComfyUI
(Workflow → Open), pick an audio file in the `LoadAudio` node, and queue the
prompt.

<img width="1762" height="1020" alt="BOO MOSS-Audio basic captioning workflow in ComfyUI" src="https://github.com/user-attachments/assets/cb3ae9ac-d3ca-4b38-9580-cab8e3728c83" />

## Requirements

Runs inside a ComfyUI checkout, same as any custom node package. The only
extra dependency is `huggingface_hub`, used to download checkpoints — see
`pyproject.toml`/`requirements.txt`. `torch`/`torchaudio`/`transformers`
already bundled with ComfyUI are used as-is.

## Vendored MOSS-Audio code

`vendor/moss_audio/` contains four small files copied from the
[MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio) repository
(`modeling_moss_audio.py`, `processing_moss_audio.py`,
`configuration_moss_audio.py`, `audio_io.py`) rather than a `pip install` of
the upstream package. Two reasons:

1. MOSS-Audio's Hugging Face checkpoints only declare `trust_remote_code`
   support for `AutoConfig`/`AutoProcessor` in `config.json`, not `AutoModel`
   — there's no modeling code bundled in the model repo itself, so
   `AutoModel.from_pretrained(..., trust_remote_code=True)` fails with
   "Unrecognized configuration class". The upstream repo's own `infer.py`
   works around this by importing `MossAudioModel` directly from its `src/`
   package; these vendored files let this ComfyUI package do the same.
2. `pip install`ing the upstream `moss-audio` package pulls in `gradio` and
   `streamlit` (used only by its own demo `app.py`) as unconditional base
   dependencies, which force-upgrades `huggingface_hub` to a version
   incompatible with the `transformers` version ComfyUI itself depends on —
   this actually broke ComfyUI's own model loading when tried. Vendoring
   just the inference-path files avoids that dependency tree entirely.

See `vendor/moss_audio/NOTICE` for provenance, source commit, license basis
(Apache-2.0), and the one line changed from upstream (an import path fix).

## Tests

```bash
source /path/to/ComfyUI/.venv/bin/activate
pytest
```

### GPU integration tests

`tests_gpu/` holds an opt-in suite that downloads a real checkpoint and
exercises a full load → offload → reload → generate cycle against an actual
CUDA GPU, verifying the model survives being moved between devices by
ComfyUI's `ModelPatcher`. It's excluded from the default `pytest` run
(not in `pyproject.toml`'s `testpaths`) since it needs a GPU, network
access, and downloads a multi-gigabyte checkpoint. Run it explicitly:

```bash
pytest tests_gpu/
```
