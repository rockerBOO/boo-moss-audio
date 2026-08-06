import logging
import os
import re

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


def _local_model_dir(repo_id: str) -> str:
    folder = folder_paths.get_folder_paths(MOSS_AUDIO_FOLDER)[0]
    return os.path.join(folder, repo_id.split("/", 1)[1])


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
        import comfy.model_management as model_management
        import comfy.model_patcher
        from huggingface_hub import snapshot_download

        repo_id = MOSS_AUDIO_REPOS[model]
        local_dir = _local_model_dir(repo_id)
        if not os.path.isdir(local_dir) or not os.listdir(local_dir):
            logging.info("BooMossAudioLoader: downloading %s to %s", repo_id, local_dir)
            snapshot_download(repo_id=repo_id, local_dir=local_dir)

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        dtype = model_management.text_encoder_dtype(load_device)

        hf_model = MossAudioModel.from_pretrained(local_dir, dtype=dtype)
        hf_model.eval()
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
                    default="Transcribe any speech verbatim, then describe the music and mood.",
                ),
                io.Int.Input("max_new_tokens", default=1024, min=1, max=8192),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=50, min=0, max=500),
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
        strip_thinking: bool,
    ) -> io.NodeOutput:
        hf_model = moss_audio_model["model"]
        processor = moss_audio_model["processor"]

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

        with torch.no_grad():
            generated_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                num_beams=1,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                top_k=top_k,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        text = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
        if strip_thinking:
            text = _THINK_BLOCK_RE.sub("", text).strip()

        return io.NodeOutput(text)


class BooMossAudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BooMossAudioLoader,
            BooMossAudioGenerate,
        ]


async def comfy_entrypoint() -> BooMossAudioExtension:
    return BooMossAudioExtension()
