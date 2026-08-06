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
  ComfyUI `AUDIO` input, and a text prompt (e.g. "Transcribe any speech
  verbatim, then describe the music and mood."), and returns the model's text
  response as a `STRING`. Thinking-variant `<think>...</think>` reasoning
  blocks are stripped by default (`strip_thinking`).

Wire `BOO MOSS-Audio Generate`'s `STRING` output into any downstream
prompt-enhancement node (e.g. [boo-textgen](https://github.com/rockerboo/boo-textgen))
to feed a transcript/mood description into further prompt generation.

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
