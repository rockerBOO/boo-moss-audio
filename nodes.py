import gc
import logging
import os
import re
from typing import Any

import folder_paths
import torch
import torchaudio
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

try:
    # Normal case: ComfyUI loads this package as "boo-moss-audio" (via
    # __init__.py), so nodes.py is imported as a submodule and relative
    # imports resolve.
    from .vendor.moss_audio.modeling_moss_audio import MossAudioModel
    from .vendor.moss_audio.processing_moss_audio import MossAudioProcessor
except ImportError:
    # tests/conftest.py loads this file standalone by path (not as part of
    # the package), so it has no parent package for a relative import to
    # resolve against. It instead puts this package's directory directly on
    # sys.path.
    from vendor.moss_audio.modeling_moss_audio import MossAudioModel
    from vendor.moss_audio.processing_moss_audio import MossAudioProcessor

# MOSS-Audio's Hugging Face checkpoints only declare `trust_remote_code`
# support for AutoConfig/AutoProcessor, not AutoModel -- there's no modeling
# code bundled in the model repo itself. See vendor/moss_audio/NOTICE for why
# MossAudioModel/MossAudioProcessor are vendored here instead.
MOSS_AUDIO_FOLDER = "moss-audio"
folder_paths.add_model_folder_path(
    MOSS_AUDIO_FOLDER, os.path.join(folder_paths.models_dir, MOSS_AUDIO_FOLDER)
)

MOSS_AUDIO_REPOS = {
    "MOSS-Audio-4B-Instruct": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    "MOSS-Audio-4B-Thinking": "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
    "MOSS-Audio-8B-Instruct": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    "MOSS-Audio-8B-Thinking": "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
}

# MOSS-Audio's Thinking variants wrap chain-of-thought reasoning in
# <think>...</think> before the actual answer. Downstream prompt-enhancement
# consumers want the answer only, so BooMossAudioGenerate strips it by default.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# MiniMax Music 3 prompt generation splits the model's single response into
# two ComfyUI outputs. Each regex captures everything after its label up to
# the next label (or end of string); DOTALL so section text can span lines.
_LYRICS_RE = re.compile(r"LYRICS:\s*(.*?)(?=STRUCTURED CAPTION:|\Z)", re.DOTALL)
_CAPTION_RE = re.compile(r"STRUCTURED CAPTION:\s*(.*)", re.DOTALL)


def _local_model_dir(repo_id: str) -> str:
    folder = folder_paths.get_folder_paths(MOSS_AUDIO_FOLDER)[0]
    return os.path.join(folder, repo_id.split("/", 1)[1])


def _run_moss_audio_generate(
    moss_audio_model: dict,
    audio: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    strip_thinking: bool,
) -> str:
    import comfy.model_management as model_management

    patcher = moss_audio_model["patcher"]
    processor = moss_audio_model["processor"]

    model_management.load_models_gpu([patcher], force_full_load=True)
    hf_model = patcher.model

    waveform = audio["waveform"][0].mean(dim=0)  # downmix to mono: [samples]
    sample_rate = audio["sample_rate"]
    target_sr = processor.config.mel_sr
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    raw_audio = waveform.cpu().numpy()

    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(hf_model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(hf_model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

    # transformers' GenerationMixin.generate() has no `generator` kwarg --
    # sampling draws from torch's global RNG, so seeding it here is the
    # only way to make do_sample=True runs reproducible.
    torch.manual_seed(seed)

    try:
        with torch.no_grad():
            generated_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                num_beams=1,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                use_cache=True,
            )
    finally:
        # A steady-state MOSS-Audio load doesn't need to unload the model
        # itself after every call -- load_models_gpu is idempotent, so
        # the next call just reuses the resident weights. But transient
        # CUDA state from this generate() call (KV cache, activations)
        # should still be released here, including on the exception path
        # (KeyboardInterrupt / ComfyUI's own interrupt exception).
        gc.collect()
        model_management.soft_empty_cache(force=True)

    input_len = inputs["input_ids"].shape[1]
    text = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
    if strip_thinking:
        text = _THINK_BLOCK_RE.sub("", text).strip()

    return text


def _make_device_settable(model):
    """comfy.model_patcher.ModelPatcher assigns directly to `model.device`
    during offload/reload (comfy/model_patcher.py:346,348,1095,1149,1992),
    but HF's PreTrainedModel.device (inherited via ModuleUtilsMixin) is a
    read-only computed property with no setter -- confirmed via AttributeError
    on the first real load_models_gpu() offload. Overriding `device` as a
    plain settable property on a dynamic per-instance subclass lets
    ModelPatcher track device state the same way it does for comfy-native
    models, which set `self.device` as an ordinary attribute.
    """
    patched_cls = type(
        f"{type(model).__name__}ComfyPatchable",
        (type(model),),
        {
            "device": property(
                lambda self: self.__dict__["_comfy_device"],
                lambda self, value: self.__dict__.__setitem__("_comfy_device", value),
            )
        },
    )
    model.__class__ = patched_cls
    model._comfy_device = next(model.parameters()).device
    return model


BooMossAudioModel = io.Custom("BOO_MOSS_AUDIO_MODEL")


class BooMossAudioLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BooMossAudioLoader",
            display_name="BOO MOSS-Audio Loader",
            category="audio",
            description=(
                "Loads a MOSS-Audio audio-understanding model (OpenMOSS). Downloads "
                "the checkpoint from Hugging Face to models/moss-audio on first use."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=list(MOSS_AUDIO_REPOS.keys()),
                    default="MOSS-Audio-4B-Instruct",
                    tooltip=(
                        "Instruct variants follow instructions directly; Thinking "
                        "variants reason step-by-step before answering (slower, "
                        "wraps reasoning in <think> tags)."
                    ),
                ),
                io.Boolean.Input(
                    "enable_time_marker",
                    default=True,
                    tooltip="Insert explicit time tokens so the model can reason about when things happen.",
                ),
            ],
            outputs=[BooMossAudioModel.Output(display_name="moss_audio_model")],
            search_aliases=["moss", "openmoss", "audio understanding", "asr", "captioning"],
        )

    @classmethod
    def execute(cls, model: str, enable_time_marker: bool) -> io.NodeOutput:
        import comfy.model_patcher
        from comfy import model_management
        from huggingface_hub import snapshot_download

        repo_id = MOSS_AUDIO_REPOS[model]
        local_dir = _local_model_dir(repo_id)
        if not os.path.isdir(local_dir) or not os.listdir(local_dir):
            logging.info("BooMossAudioLoader: downloading %s to %s", repo_id, local_dir)
            snapshot_download(repo_id=repo_id, local_dir=local_dir)

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        dtype = model_management.text_encoder_dtype(load_device)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            # MOSS-Audio has no comfy.ops casting layers, so fp8 (from
            # --fp8-e4m3fn-text-enc/--fp8-e5m2-text-enc) would crash on the first
            # F.conv2d/F.linear call. Fall back to a dtype it can actually run.
            dtype = torch.float16
        if dtype == torch.float16 and model_management.should_use_bf16(load_device):
            # Every MOSS-Audio checkpoint's config.json declares bfloat16; prefer it
            # over the text_encoder_dtype() default of float16 when the device
            # supports it.
            dtype = torch.bfloat16

        hf_model = MossAudioModel.from_pretrained(local_dir, dtype=dtype)
        hf_model.eval()
        hf_model = _make_device_settable(hf_model)
        patcher = comfy.model_patcher.ModelPatcher(
            hf_model, load_device=load_device, offload_device=offload_device
        )
        processor = MossAudioProcessor.from_pretrained(
            local_dir,
            enable_time_marker=enable_time_marker,
        )

        return io.NodeOutput({"patcher": patcher, "processor": processor, "model_id": model})


class BooMossAudioGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BooMossAudioGenerate",
            display_name="BOO MOSS-Audio Generate",
            category="audio",
            description=(
                "Runs a MOSS-Audio model over an audio input (speech, music, or "
                "ambient sound) and returns a text description — transcript, "
                "mood/genre analysis, or answer to a question, depending on the prompt."
            ),
            inputs=[
                BooMossAudioModel.Input("moss_audio_model"),
                io.Audio.Input("audio"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="""You are an audio analyst. Do BOTH of the following, in order, and do not skip either:
1) SPEECH: Transcribe every spoken or sung word verbatim, in order. Start a new tagged block only when switching between spoken dialogue and sung lyrics, writing "[spoken]" or "[sung]" once at the start of that block, not before every line. Inside a sung block, put each lyric line on its own line (a literal line break between lines), the way song lyrics are normally printed; keep spoken dialogue in flowing prose. If there is no speech or singing at all, write "(none)".
2) STYLE: Describe the musical/ambient style, genre, instrumentation, tempo, and overall mood.
Always include both sections, labeled exactly "SPEECH:" and "STYLE:". Never repeat the same word, phrase, or sound more than twice in a row.

Example output:
SPEECH: [spoken] Hey, are you seeing this?
[sung]
Lights are low,
we're moving slow,
dancing where the shadows grow,
holding on to what we know.
STYLE: A mellow synth-pop track with a laid-back tempo, warm analog pads, and a soft four-on-the-floor beat. The mood is intimate and dreamy.

Now analyze the given audio in the same format.""",
                ),
                io.Int.Input("max_new_tokens", default=1024, min=1, max=8192),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=50, min=0, max=500),
                io.Float.Input("repetition_penalty", default=1.0, min=1.0, max=2.0, step=0.01),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for sampling (only affects output when temperature > 0).",
                ),
                io.Boolean.Input(
                    "strip_thinking",
                    default=True,
                    tooltip="Remove <think>...</think> reasoning blocks from Thinking-variant output.",
                ),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(
        cls,
        moss_audio_model: dict,
        audio: dict,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
        strip_thinking: bool,
    ) -> io.NodeOutput:
        text = _run_moss_audio_generate(
            moss_audio_model=moss_audio_model,
            audio=audio,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            strip_thinking=strip_thinking,
        )
        return io.NodeOutput(text)


class BooMossAudioMiniMaxMusic3PromptGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BooMossAudioMiniMaxMusic3PromptGenerate",
            display_name="BOO MOSS-Audio MiniMax Music 3 Prompt Generate",
            category="audio",
            description=(
                "Runs a MOSS-Audio model over a reference audio clip and returns "
                "the two inputs MiniMax Music 3 consumes: lyrics (with section "
                "tags) and a structured caption (Global Metadata, Vocal Details, "
                "Arrangement), split into separate outputs."
            ),
            inputs=[
                BooMossAudioModel.Input("moss_audio_model"),
                io.Audio.Input("audio"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="""You are an audio analyst. Your job is to convert an audio clip into the two inputs
MiniMax Music 3 actually consumes: **LYRICS** and **STRUCTURED CAPTION**. Do BOTH,
in order, and do not skip either.

---

## 1) LYRICS

Transcribe every sung word verbatim, in order, and lay it out using MiniMax's
section-tag contract:

- Break the song into sections using only these tags, in whatever order they
  actually occur: `[intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge]
  [instrumental] [solo] [outro]`.
- **Every tag goes on its own line, with nothing else on that line.** Text placed
  on the same line as a tag is silently dropped by MiniMax's input contract, so
  never write `[verse] Morning light…` — the tag and the lyric must be on
  separate lines.
- Under each tag, write the sung lyric lines exactly as sung, one line per line
  break, the way lyrics are normally printed.
- Instrumental passages (solos, breakdowns with no vocal) get their own
  `[instrumental]` or `[solo]` tag and no lyric text underneath.
- If a section repeats near-verbatim (e.g. a repeated chorus), still transcribe
  it in full under its own tag — don't abbreviate with "repeat chorus."
- Never repeat the same word, phrase, or sound more than twice in a row, even if
  that's what's audibly sung (e.g. long ad-lib repetitions) — cap it at two and
  note the effect in STRUCTURED CAPTION instead (e.g. "extended repeated ad-lib
  outro") rather than transcribing it exhaustively.
- If there is **spoken (non-sung) dialogue**, note that MiniMax's lyric field has
  no native tag for it. Transcribe it separately, after the tagged lyric block,
  under a clearly marked heading `Spoken dialogue (not part of MiniMax lyric
  input):`, in flowing prose, so the information isn't lost — but do not mix it
  into the tagged section above.
- If there is no speech or singing at all, write "(none)" for this whole section.

## 2) STRUCTURED CAPTION

Describe the music using exactly these three headings, in this order, ~250–450
words total. This is the field that carries all musical control in MiniMax — the
model follows it over time, not as one global tag.

**Global Metadata**
- Basic Attributes: genre/subgenres, tempo (exact BPM only if genuinely
  confident; otherwise a qualitative tempo like "driving" or "unhurried"), and
  key/scale only if clearly identifiable.
- Global Emotional Progression: the arc from open to close as a story — where it
  starts, where it peaks, how it resolves.
- Application Scenarios & Imagery: a concrete scene the track evokes.
- Sonics & Production Profile: stereo width, frequency balance, dynamics
  (polished/compressed vs. natural/uncompressed).

**Vocal Details** (omit this heading's content and state "instrumental — no
vocal, lead melody carried by [instrument]" if there's no singing)
- Vocal Gender & Timbre: explicit, e.g. "Singer A (Female), warm mezzo-soprano."
- Vocal Style: delivery per section — restrained in verse, belted in chorus, etc.
- Harmony/Backing Vocals: doubles, stacked harmonies, call-and-response, and where.
- Vocal FX: reverb, delay, saturation — only where actually audible.

**Arrangement**
- Instrument Lifecycle (Primary/Secondary): what anchors the track start to
  finish, what enters/exits/transforms along the way.
- Groove & Foundation Progression: what the rhythm section does per section —
  verse vs. chorus vs. bridge.
- Embellishments, Textures & Spatial FX: risers, sweeps, reverb tails, ear candy.

---

## Rules

- Label the two outputs exactly `LYRICS:` and `STRUCTURED CAPTION:`.
- Don't fabricate precision (exact BPM/key) you're not confident about.
- An explicit vocal gender, instrumental requirement, or exclusion stated once
  must not be contradicted later in the caption.
- Keep the caption describing *music*, not words — lyric content stays in the
  LYRICS section only.
- Write the Global Emotional Progression as a story with tension, release, and
  climax, not a static list of adjectives.""",
                ),
                io.Int.Input("max_new_tokens", default=1024, min=1, max=8192),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=50, min=0, max=500),
                io.Float.Input("repetition_penalty", default=1.0, min=1.0, max=2.0, step=0.01),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for sampling (only affects output when temperature > 0).",
                ),
                io.Boolean.Input(
                    "strip_thinking",
                    default=True,
                    tooltip="Remove <think>...</think> reasoning blocks from Thinking-variant output.",
                ),
            ],
            outputs=[
                io.String.Output("lyrics"),
                io.String.Output("structured_caption"),
            ],
        )

    @classmethod
    def execute(
        cls,
        moss_audio_model: dict,
        audio: dict,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
        strip_thinking: bool,
    ) -> io.NodeOutput:
        text = _run_moss_audio_generate(
            moss_audio_model=moss_audio_model,
            audio=audio,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            strip_thinking=strip_thinking,
        )

        lyrics_match = _LYRICS_RE.search(text)
        lyrics = lyrics_match.group(1).strip() if lyrics_match else ""

        caption_match = _CAPTION_RE.search(text)
        structured_caption = caption_match.group(1).strip() if caption_match else ""

        return io.NodeOutput(lyrics, structured_caption)


class BooMusicCaptionRewriter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BooMusicCaptionRewriter",
            display_name="BOO Music Caption Rewriter",
            category="audio",
            description=(
                "Takes style keywords and optional lyrics, routes through a bundled "
                "MiniMax Music 3 reference library using an external LLM with an "
                "8-stage agent pipeline, and returns a structured caption (Global "
                "Metadata, Vocal Details, Arrangement)."
            ),
            inputs=[
                io.AnyType.Input(
                    "llm_model",
                    tooltip="External LLM model (e.g., Gemma).",
                ),
                io.String.Input(
                    "style_keywords",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Style description: genres, moods, instrumentation, "
                        "cultural cues, fusion combinations."
                    ),
                ),
                io.String.Input(
                    "lyrics",
                    multiline=True,
                    default="",
                    tooltip="Optional lyrics with bracketed section tags like [Verse], [Chorus].",
                ),
            ],
            outputs=[io.String.Output("caption")],
            search_aliases=["music caption", "mini max", "minimax", "template", "reference"],
        )

    @classmethod
    async def execute(
        cls,
        llm_model: Any,
        style_keywords: str,
        lyrics: str,
    ) -> io.NodeOutput:
        from .music_caption import CaptionRewriter

        rewriter = CaptionRewriter(llm_model)
        caption = await rewriter.rewrite(style_keywords, lyrics)
        return io.NodeOutput(caption)


class BooMossAudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BooMossAudioLoader,
            BooMossAudioGenerate,
            BooMossAudioMiniMaxMusic3PromptGenerate,
            BooMusicCaptionRewriter,
        ]


async def comfy_entrypoint() -> BooMossAudioExtension:
    return BooMossAudioExtension()
