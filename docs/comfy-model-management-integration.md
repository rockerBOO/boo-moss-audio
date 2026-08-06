# ComfyUI model-management integration — empirically blocked

Tasks 1-2 of the implementation plan (see
`docs/superpowers/specs/2026-08-06-comfy-model-management-integration-design.md`)
wrapped the loaded HF model in `comfy.model_patcher.ModelPatcher` and
registered it with `load_models_gpu` before inference
(`nodes.py`'s `BooMossAudioLoader`/`BooMossAudioGenerate`). Code review of
that change passed, but step 7 below — the one thing code review alone
can't confirm — was flagged as needing a real GPU load → offload → reload →
generate cycle.

`tests_gpu/test_moss_audio_gpu_roundtrip.py` runs exactly that cycle
against a real CUDA GPU and a real checkpoint. It fails, and the failure is
a genuine incompatibility, not a test bug:

```
model_management.load_models_gpu([patcher])
...
comfy/model_patcher.py:1149: in unpatch_model
    self.model.device = device_to
AttributeError: property 'device' of 'MossAudioModel' object has no setter
```

`ModelPatcher` assumes `self.model` exposes a plain, settable `.device`
attribute — true for ComfyUI's own model classes (e.g. `comfy/model_base.py`
sets `self.device` as a normal instance attribute), and several places in
`comfy/model_patcher.py` (`partially_load`/`unpatch_model`/`patch_model`,
around lines 346-348, 1095, 1149, 1297, 1992) assign to it directly.
`MossAudioModel`, like any `transformers.PreTrainedModel`, instead inherits
`device` as a **read-only computed property** from
`transformers.modeling_utils.ModuleUtilsMixin` (`get_parameter_device(self)`)
— there is no setter, so the assignment raises `AttributeError` the first
time `ModelPatcher` tries to move the model off the load device.

This means wrapping a raw HF `PreTrainedModel` in `ModelPatcher` as
implemented by Tasks 1-2 does not actually work end-to-end: the model loads
fine the first time (construction-time `device_map=...`/`.to()` calls don't
go through this path), but any subsequent offload/reload — the exact thing
`ModelPatcher` exists to do — crashes on the `device` property.

## Status

**Not resolved.** The GPU integration test
(`tests_gpu/test_moss_audio_gpu_roundtrip.py`) exists and correctly detects
this; do not weaken or delete its assertions to make it pass. Fixing this
needs a follow-up change beyond the scope of the current plan, e.g. one of:

- Subclass/monkeypatch `MossAudioModel` to add a settable `device` attribute
  (e.g. override the property, or set `type(model).device` to a plain
  attribute-backed property) before wrapping it in `ModelPatcher`.
- Give `ModelPatcher` a model wrapper (comfy-native `nn.Module`) that holds
  the HF model as a submodule and owns its own settable `device`, mirroring
  `boo-textgen/minimax_h3_tail.py`'s `_build_tail()` pattern more closely
  instead of wrapping the raw `PreTrainedModel` directly.
- Confirm whether newer/older `transformers` versions differ here (this was
  observed against `transformers==4.57.3`) before committing to a fix.

Re-run `pytest tests_gpu/ -v` after any such fix; it should go from
`AttributeError` to a clean pass (`load_models_gpu` succeeds, offload drops
the model back to the offload device, and a real `generate()` call returns
non-empty text) before this doc is marked resolved.
