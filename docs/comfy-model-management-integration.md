# TODO: integrate MOSS-Audio with ComfyUI's model management

Not implemented yet. Written down for a future agent/session instead of
being done inline, because it's a real but bounded change (medium-sized —
roughly a day of careful implementation + testing), not a quick fix.

## Current state (the problem)

`BooMossAudioLoader.execute` (`nodes.py`) loads the HF model with
`device_map=model_management.get_torch_device()` and holds a plain reference
to it for the node's lifetime. This model is completely invisible to
ComfyUI's VRAM manager (`comfy.model_management`): once loaded, it
permanently occupies VRAM and ComfyUI has no way to swap it out when a
diffusion model or another CLIP needs the space, unlike every other
model-holding node in ComfyUI (including this repo's sibling package,
boo-textgen, whose MiniMax H3 / Qwen3-VL CLIP nodes *do* participate in
VRAM-aware loading).

## What Qwen3-VL / MiniMax H3 do differently (the precedent)

ComfyUI's `comfy.model_patcher.ModelPatcher` (`comfy/model_patcher.py:340`)
wraps *any* `nn.Module` — it only needs `state_dict()`/`.to(device)` to
exist, and stamps bookkeeping attributes onto the module rather than
requiring a comfy-authored architecture. Size accounting comes from
`comfy.model_management.module_size()` (`comfy/model_management.py:623`),
which just sums `state_dict()` tensor `.nbytes` — architecture-agnostic.
Passing a `ModelPatcher` to `comfy.model_management.load_models_gpu([...])`
(`comfy/model_management.py:901`) registers it in the global
`current_loaded_models` registry, making it visible for eviction/reload
alongside every other loaded model.

`boo-textgen/minimax_h3_tail.py`'s `_build_tail()` (around line 265-298) is
the concrete template for this in our own codebase: it builds a plain
`nn.Module`, wraps it in `ModelPatcher(tail, load_device=..., offload_device=...)`,
registers it with `load_models_gpu([clip.patcher, tail_patcher], memory_required=...)`
right before generation, and explicitly unloads it afterward in a
`try/finally` via `comfy.model_management.unload_model_and_clones(tail_patcher, ..., all_devices=True)`
plus `gc.collect()` + `soft_empty_cache(force=True)`. That module is
comfy-authored (uses `comfy.ops`), though — it doesn't prove a HF
`PreTrainedModel` specifically behaves correctly under this treatment, which
is the real open question below.

## What would actually need to change

1. **Drop `device_map=...`** from `MossAudioModel.from_pretrained` — load on
   CPU so ComfyUI's `ModelPatcher.to(device)` calls own placement instead of
   fighting HF's own `accelerate` device-map dispatch (mixing the two is a
   real hazard, not just a style preference).
2. **Wrap the loaded model**: `ModelPatcher(hf_model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device())`.
3. **Move the actual `load_models_gpu([patcher], ...)` call into
   `BooMossAudioGenerate.execute`**, not the loader — mirrors
   `generate_with_tail`'s pattern of loading right before use so unrelated
   VRAM pressure can evict a MOSS-Audio model sitting idle between runs.
   `BooMossAudioLoader` should return the (CPU-resident, unregistered)
   patcher; `BooMossAudioGenerate` registers it for the duration of the call.
4. **Run inference against `patcher.model`**, not the original `hf_model`
   reference — after `ModelPatcher` construction, `patcher.model` is the
   instance ComfyUI's offload machinery actually moves between devices.
5. **Pin a concrete dtype** instead of `dtype="auto"` — should follow the
   same pattern as `comfy.model_management.text_encoder_dtype`/`unet_dtype`
   so the chosen dtype is consistent with what `load_device` can actually
   run and with what ComfyUI expects when it moves the model.
6. **No sub-layer lowvram streaming will be available** — MOSS-Audio's
   modules aren't wrapped in `comfy.ops`-casting classes, so `ModelPatcher`
   can only do whole-model load/offload for it, not the partial per-layer
   eviction some comfy-native models get under tight VRAM. That's an
   accepted limitation, not a blocker.
7. **Verify empirically** that MOSS-Audio's HF modules tolerate being moved
   between devices post-construction by `ModelPatcher` — some HF models
   cache device-dependent buffers (rotary embedding caches, KV-cache dtype)
   at construction or first-forward time that can go stale across an
   offload/reload cycle. This needs a real load → offload → reload →
   generate test, not just code review.
8. **Follow the defensive cleanup pattern** `generate_with_tail` uses:
   `try/finally` around the generate call, catching `BaseException` (so
   `KeyboardInterrupt`/ComfyUI's own interrupt exception still triggers
   cleanup), even though — unlike the tail — a steady-state MOSS-Audio load
   doesn't need to *unload* after every call. `load_models_gpu` is
   idempotent/registry-aware, so calling it every `BooMossAudioGenerate.execute`
   is fine and is how ComfyUI naturally keeps a model loaded across runs
   while still allowing eviction under pressure elsewhere.

## Why this wasn't just done inline

The change touches the load/generate split (state currently returned by the
loader would need to become a patcher instead of a live model+processor
dict), requires dtype decisions that need testing against actual VRAM
behavior, and step 7 specifically can't be verified by code reading alone —
it needs a real GPU session with an actual offload/reload cycle exercised
before generating, which wasn't done as part of this write-up.
