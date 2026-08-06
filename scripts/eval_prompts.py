"""Compare BooMossAudioGenerate prompt variants against a real audio file on a
real GPU. Loads the model once and runs every prompt against it, so you can
judge candidate default prompts (or debug regressions) without repeatedly
paying model-load cost.

Usage (from a ComfyUI checkout, with its venv active):

    uv run --project /path/to/ComfyUI python scripts/eval_prompts.py \\
        --audio /path/to/clip.mp3 \\
        --prompts-file scripts/prompts.json \\
        --repeats 2

`--prompts-file` is a JSON object of {name: prompt_text}; if omitted, this
script just runs BooMossAudioGenerate's current schema default. Non-mp3/wav
inputs must be decoded to wav first (e.g. via ffmpeg), since this script
loads audio through scipy rather than torchaudio/torchcodec.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = PACKAGE_ROOT.parents[1]


def _load_nodes_module():
    for p in (COMFYUI_ROOT, PACKAGE_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("boo_moss_audio_nodes", PACKAGE_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["boo_moss_audio_nodes"] = module
    spec.loader.exec_module(module)
    return module


def _load_audio(wav_path: str) -> dict:
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    waveform = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
    return {"waveform": waveform, "sample_rate": sr}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, help="Path to a mono/stereo .wav file")
    parser.add_argument("--model", default="MOSS-Audio-4B-Instruct", choices=[
        "MOSS-Audio-4B-Instruct", "MOSS-Audio-4B-Thinking",
        "MOSS-Audio-8B-Instruct", "MOSS-Audio-8B-Thinking",
    ])
    parser.add_argument("--prompts-file", help="JSON object of {name: prompt_text}")
    parser.add_argument("--repeats", type=int, default=1, help="Trials per prompt, each with a different seed")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    nodes = _load_nodes_module()

    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())
    else:
        default_prompt = next(
            inp.default
            for inp in nodes.BooMossAudioGenerate.define_schema().inputs
            if getattr(inp, "id", None) == "prompt"
        )
        prompts = {"schema_default": default_prompt}

    audio = _load_audio(args.audio)

    print(f"Loading {args.model}...", flush=True)
    loader_output = nodes.BooMossAudioLoader.execute(model=args.model, enable_time_marker=True)
    moss_audio_model = loader_output.args[0]
    print("Model loaded.", flush=True)

    for name, prompt in prompts.items():
        for trial in range(args.repeats):
            seed = 42 + trial
            torch.manual_seed(seed)
            print(f"\n===== {name} (trial {trial}, seed={seed}) =====")
            print(f"PROMPT: {prompt}")
            output = nodes.BooMossAudioGenerate.execute(
                moss_audio_model=moss_audio_model,
                audio=audio,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                strip_thinking=True,
            )
            print(f"OUTPUT:\n{output.args[0]}")
            print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
