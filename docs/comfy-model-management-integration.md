# ComfyUI model-management integration — resolved

This TODO was implemented. See:

- `docs/superpowers/specs/2026-08-06-comfy-model-management-integration-design.md`
  for the design.
- `nodes.py`'s `BooMossAudioLoader`/`BooMossAudioGenerate` for the
  implementation (`ModelPatcher` wrap in the loader, `load_models_gpu`
  registration in generate, `_make_device_settable` working around HF
  `PreTrainedModel.device` being a read-only property, and
  `force_full_load=True` working around `ModelPatcher`'s per-module lowvram
  accounting not understanding MOSS-Audio's plain HF submodules).
- `tests_gpu/test_moss_audio_gpu_roundtrip.py` for the empirical
  load→offload→reload→generate verification, confirmed passing end-to-end
  on real hardware (RTX 5070 Ti).
