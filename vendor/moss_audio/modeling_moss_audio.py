from typing import Optional, List, Union, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils.auto_docstring import auto_docstring
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

# Changed from `from src.configuration_moss_audio import ...` (see
# vendor/moss_audio/NOTICE) so this module resolves as part of the
# boo-moss-audio.vendor.moss_audio package rather than a top-level `src`.
from .configuration_moss_audio import MossAudioEncoderConfig, MossAudioConfig

# Flip this to True to print isfinite() checkpoints through the audio
# encoder (conv stem, positions, per-layer, adapter). For chasing the
# intermittent "audio_embeds contains non-finite values" error -- toggle
# on, reproduce, and the first non-finite checkpoint printed pinpoints the
# stage. Leave off by default; the per-layer check adds overhead.
DEBUG_NAN_CHECKS = False


class SinusoidsPositionEmbedding(nn.Module):
    # inv_timescales used to be a `persistent=False` registered buffer,
    # cached at __init__ time as a micro-optimization. persistent=False
    # means it's excluded from state_dict(), so nothing in a
    # checkpoint-driven weight-loading path (HF's meta-device deferred
    # init, or a host app's own weight-casting/device-placement pass that
    # walks the state_dict) has a reason to touch it. Intermittently, it
    # was observed on GPU with correct shape/dtype/device but garbage
    # values (e.g. min=-1.89e38, max=9.18e-39 -- nowhere near the
    # buffer's true (0, 1] range), consistent with the tensor being
    # materialized as uninitialized memory rather than the real computed
    # values. `arange(seq_len) * garbage` then overflows float32 to
    # +/-inf, and sin/cos of that is nan, poisoning every layer
    # downstream. It's cheap to compute (320 floats, one exp call) and
    # depends only on this call's arguments, so recompute it fresh on
    # every forward instead of caching it anywhere that a weight-loading
    # path could leave stale or uninitialized.
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_timescale = 10000.0

    def forward(self, seq_len: int, device: torch.device):
        log_timescale_increment = math.log(self.max_timescale) / (self.embedding_dim // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment
            * torch.arange(self.embedding_dim // 2, device=device, dtype=torch.float32)
        )
        if DEBUG_NAN_CHECKS:
            print(
                f"[DEBUG] inv_timescales isfinite={bool(torch.isfinite(inv_timescales).all())} "
                f"device={inv_timescales.device} dtype={inv_timescales.dtype} "
                f"shape={tuple(inv_timescales.shape)} "
                f"min={inv_timescales.min().item()} max={inv_timescales.max().item()}"
            )
        scaled_time = torch.arange(
            seq_len, device=device, dtype=inv_timescales.dtype
        ).unsqueeze(1) * inv_timescales.unsqueeze(0)
        sin_emb = torch.sin(scaled_time)
        cos_emb = torch.cos(scaled_time)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=1)
        return pos_emb.unsqueeze(0)


class MossAudioEncoder(nn.Module):
    """Audio encoder with conv-stem downsampling and Whisper transformer layers."""

    def __init__(self, config: MossAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.gelu = nn.GELU()

        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        # 128 mel bins / 8 = 16 after 3 convs with stride=2
        self.stem_proj = nn.Linear(config.downsample_hidden_size * 16, config.d_model)
        self.embed_positions = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_proj = (
            nn.Linear(config.d_model, config.output_dim, bias=False)
            if config.output_dim != config.d_model
            else nn.Identity()
        )

        self.deepstack_encoder_layer_indexes = list(
            config.deepstack_encoder_layer_indexes or []
        )
        self._deepstack_capture_map = {
            layer_idx: capture_idx
            for capture_idx, layer_idx in enumerate(self.deepstack_encoder_layer_indexes)
        }

        self.n_window = int(config.n_window)
        self.chunk_frames = int(self.n_window * 2)
        self.conv_chunksize = int(config.conv_chunksize)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        def conv_out_len(L):
            return (L - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(lengths)))

    def _encode_chunk_batch(
        self,
        input_features: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode a batch of (already padded) chunks through the conv stem and
        transformer layers. Returns (last_hidden, ordered_deepstack_hidden_states).
        """
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        downsampled_lengths = self._compute_downsampled_length(seq_lengths)

        # [B, n_mels, T] -> [B, 1, n_mels, T]
        x = input_features.unsqueeze(1)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))

        # [B, C, F, T] -> [B, T, C*F]
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.stem_proj(x)

        max_len = int(downsampled_lengths.max().item())
        if x.size(1) > max_len:
            x = x[:, :max_len, :]

        if DEBUG_NAN_CHECKS:
            print(
                f"[DEBUG] post-conv-stem x isfinite={bool(torch.isfinite(x).all())} shape={tuple(x.shape)} "
                f"min={x.min().item()} max={x.max().item()} absmax={x.abs().max().item()} "
                f"bf16_max={torch.finfo(x.dtype).max}"
            )

        positions = self.embed_positions(x.shape[1], x.device)
        positions_cast = positions.to(x.dtype)
        if DEBUG_NAN_CHECKS:
            print(
                f"[DEBUG] positions isfinite={bool(torch.isfinite(positions_cast).all())} "
                f"min={positions_cast.min().item()} max={positions_cast.max().item()}"
            )
        x = x + positions_cast
        if DEBUG_NAN_CHECKS:
            print(f"[DEBUG] post-positions x isfinite={bool(torch.isfinite(x).all())}")

        padding_mask = (
            torch.arange(x.size(1), device=x.device)[None, :] >= downsampled_lengths[:, None]
        )
        attention_mask = (1.0 - (~padding_mask).to(dtype=x.dtype)) * torch.finfo(x.dtype).min
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        if DEBUG_NAN_CHECKS:
            print(
                f"[DEBUG] downsampled_lengths={downsampled_lengths.tolist()} "
                f"x.size(1)={x.size(1)} attention_mask isfinite={bool(torch.isfinite(attention_mask).all())}"
            )

        deepstack_hidden_states: List[Optional[torch.Tensor]] = [None] * len(
            self.deepstack_encoder_layer_indexes
        )
        first_bad_layer = None
        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(
                x,
                attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )
            # transformers >=4.51 WhisperEncoderLayer.forward returns the
            # hidden-state tensor directly; older versions returned
            # (hidden_states, attn_weights). Indexing [0] unconditionally
            # (upstream's assumption) silently slices the batch dimension
            # of the tensor on newer transformers instead of unpacking a
            # tuple, corrupting every layer after the first.
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            if DEBUG_NAN_CHECKS and first_bad_layer is None and not torch.isfinite(x).all():
                first_bad_layer = layer_idx
                print(f"[DEBUG] FIRST non-finite x at layer_idx={layer_idx}")
            capture_idx = self._deepstack_capture_map.get(layer_idx)
            if capture_idx is not None:
                deepstack_hidden_states[capture_idx] = x

        if DEBUG_NAN_CHECKS:
            print(f"[DEBUG] post-all-layers x isfinite={bool(torch.isfinite(x).all())} first_bad_layer={first_bad_layer}")

        x = self.layer_norm(x)
        x = self.out_proj(x)
        if DEBUG_NAN_CHECKS:
            print(f"[DEBUG] post-layer_norm-and-out_proj x isfinite={bool(torch.isfinite(x).all())}")

        ordered_deepstack_hidden_states = [
            h for h in deepstack_hidden_states if h is not None
        ]
        if not isinstance(self.out_proj, nn.Identity):
            ordered_deepstack_hidden_states = [
                self.out_proj(h) for h in ordered_deepstack_hidden_states
            ]
        return x, ordered_deepstack_hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        output_deepstack_hidden_states: bool = True,
    ) -> BaseModelOutputWithPast:
        if input_features.dim() == 3:
            if feature_lens is None:
                feature_lens = torch.full(
                    (input_features.size(0),),
                    input_features.size(-1),
                    dtype=torch.long,
                    device=input_features.device,
                )
            else:
                feature_lens = feature_lens.to(
                    device=input_features.device, dtype=torch.long
                )
            valid_chunks = [
                input_features[i, :, : int(feature_lens[i].item())]
                for i in range(int(input_features.shape[0]))
            ]
            input_features = torch.cat(valid_chunks, dim=1)
        elif input_features.dim() != 2:
            raise ValueError(
                f"Expected [n_mels, T] or [B, n_mels, T], got {tuple(input_features.shape)}."
            )

        if feature_lens is None:
            feature_lens = torch.tensor(
                [int(input_features.shape[1])],
                device=input_features.device,
                dtype=torch.long,
            )
        else:
            feature_lens = feature_lens.to(
                device=input_features.device, dtype=torch.long
            )

        chunk_frames = int(self.chunk_frames)
        chunk_num = torch.ceil(
            feature_lens.to(torch.float32) / float(chunk_frames)
        ).long()
        chunk_lengths = torch.full(
            (int(chunk_num.sum().item()),),
            chunk_frames,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % chunk_frames
        chunk_lengths[chunk_lengths == 0] = chunk_frames

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        feature_lens_after_cnn = self._compute_downsampled_length(chunk_lengths)
        t_down_max = (
            int(feature_lens_after_cnn.max().item())
            if feature_lens_after_cnn.numel() > 0
            else 0
        )
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(int(L.item()), dtype=torch.bool, device=padded_feature.device)
                for L in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        if padded_mask_after_cnn.shape[1] < t_down_max:
            padded_mask_after_cnn = F.pad(
                padded_mask_after_cnn,
                (0, t_down_max - padded_mask_after_cnn.shape[1]),
                value=False,
            )

        num_deepstack = len(self.deepstack_encoder_layer_indexes)
        padded_embeds: List[torch.Tensor] = []
        deepstack_padded_embeds: List[List[torch.Tensor]] = [
            [] for _ in range(num_deepstack)
        ]
        for feat_chunk, len_chunk in zip(
            padded_feature.split(self.conv_chunksize, dim=0),
            chunk_lengths.split(self.conv_chunksize, dim=0),
        ):
            out, deepstack_outs = self._encode_chunk_batch(feat_chunk, len_chunk)
            if out.shape[1] < t_down_max:
                out = F.pad(out, (0, 0, 0, t_down_max - out.shape[1]))
            padded_embeds.append(out)
            if output_deepstack_hidden_states and num_deepstack > 0:
                if len(deepstack_outs) != num_deepstack:
                    raise RuntimeError(
                        "Deepstack output count does not match configured layer indexes."
                    )
                for capture_idx, ds in enumerate(deepstack_outs):
                    if ds.shape[1] < t_down_max:
                        ds = F.pad(ds, (0, 0, 0, t_down_max - ds.shape[1]))
                    deepstack_padded_embeds[capture_idx].append(ds)

        if padded_embeds:
            padded_embed = torch.cat(padded_embeds, dim=0)
        else:
            padded_embed = torch.empty(
                (0, t_down_max, self.config.output_dim),
                device=padded_feature.device,
            )

        valid_tokens = padded_embed[padded_mask_after_cnn]  # [N_valid, D]
        last_hidden_state = valid_tokens.unsqueeze(0)  # [1, N_valid, D]

        deepstack_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_deepstack_hidden_states and num_deepstack > 0:
            collected: List[torch.Tensor] = []
            for chunks_list in deepstack_padded_embeds:
                if chunks_list:
                    ds = torch.cat(chunks_list, dim=0)
                    collected.append(ds[padded_mask_after_cnn].unsqueeze(0))
                else:
                    collected.append(
                        torch.empty(
                            (1, 0, self.config.output_dim),
                            device=padded_feature.device,
                            dtype=padded_embed.dtype,
                        )
                    )
            deepstack_states = tuple(collected)

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            hidden_states=deepstack_states,
        )


class GatedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@auto_docstring
class MossAudioPreTrainedModel(PreTrainedModel):
    config_class = MossAudioConfig
    config: MossAudioConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Qwen3DecoderLayer}


class MossAudioModel(MossAudioPreTrainedModel, GenerationMixin):
    config_class = MossAudioConfig
    _tied_weights_keys: List[str] = []

    def __init__(self, config: MossAudioConfig):
        super().__init__(config)

        self.audio_encoder = MossAudioEncoder(config.audio_config)
        self.language_model = Qwen3Model(config.language_config)

        self.audio_adapter = GatedMLP(
            input_size=config.audio_config.output_dim,
            hidden_size=config.adapter_hidden_size,
            output_size=config.language_config.hidden_size,
        )

        deepstack_k = len(getattr(config.audio_config, "deepstack_encoder_layer_indexes", []) or [])
        if config.deepstack_num_inject_layers is not None:
            deepstack_k = min(deepstack_k, int(config.deepstack_num_inject_layers))
        self.deepstack_audio_merger_list = nn.ModuleList(
            [
                GatedMLP(
                    input_size=config.audio_config.output_dim,
                    hidden_size=config.adapter_hidden_size,
                    output_size=config.language_config.hidden_size,
                )
                for _ in range(deepstack_k)
            ]
        )

        self.vocab_size = config.language_config.vocab_size
        self.lm_head = nn.Linear(config.language_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_audio_features(self, input_features, feature_lens):
        audio_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lens=feature_lens,
            output_deepstack_hidden_states=True,
        )
        deepstack = list(audio_outputs.hidden_states) if audio_outputs.hidden_states is not None else None
        return audio_outputs.last_hidden_state, deepstack

    def _apply_deepstack_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        audio_input_mask: torch.Tensor,
        deepstack_embeds: torch.Tensor,
    ) -> torch.Tensor:
        audio_input_mask = audio_input_mask.to(hidden_states.device)
        deepstack_embeds = deepstack_embeds.to(hidden_states.device, hidden_states.dtype)
        flat = deepstack_embeds.reshape(-1, deepstack_embeds.shape[-1])
        hs = hidden_states.clone()
        hs[audio_input_mask] = hs[audio_input_mask] + flat
        return hs

    def _register_llm_deepstack_hooks(
        self,
        audio_input_mask: torch.Tensor,
        deepstack_audio_embeds: List[torch.Tensor],
    ):
        if deepstack_audio_embeds is None or len(deepstack_audio_embeds) == 0:
            return []

        layers = getattr(self.language_model, "layers", None)
        if layers is None:
            raise RuntimeError("Qwen3Model does not expose `.layers`; cannot register DeepStack hooks.")

        num_inject = len(deepstack_audio_embeds)
        handles = []

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= num_inject:
                break

            def _make_llm_hook(k: int):
                def _hook(_module, _inputs, _output):
                    if isinstance(_output, (tuple, list)):
                        hs = _output[0]
                        new_hs = self._apply_deepstack_to_hidden_states(
                            hs, audio_input_mask, deepstack_audio_embeds[k]
                        )
                        return (new_hs,) + tuple(_output[1:])
                    else:
                        return self._apply_deepstack_to_hidden_states(
                            _output, audio_input_mask, deepstack_audio_embeds[k]
                        )

                return _hook

            handles.append(layer.register_forward_hook(_make_llm_hook(layer_idx)))

        return handles

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        hook_handles = []
        if audio_data is not None:
            if audio_input_mask is None:
                raise ValueError("audio_input_mask is required when audio_data is provided.")

            if DEBUG_NAN_CHECKS:
                print(
                    f"[DEBUG] audio_data isfinite={bool(torch.isfinite(audio_data).all())} "
                    f"shape={tuple(audio_data.shape)} dtype={audio_data.dtype} "
                    f"min={audio_data.min().item()} max={audio_data.max().item()}"
                )
            encoder_out, deepstack = self.get_audio_features(audio_data, audio_data_seqlens)
            if DEBUG_NAN_CHECKS:
                print(
                    f"[DEBUG] encoder_out (pre-adapter) isfinite={bool(torch.isfinite(encoder_out).all())} "
                    f"shape={tuple(encoder_out.shape)} dtype={encoder_out.dtype}"
                )
            audio_embeds = self.audio_adapter(encoder_out)
            if DEBUG_NAN_CHECKS:
                print(
                    f"[DEBUG] audio_embeds (post-adapter) isfinite={bool(torch.isfinite(audio_embeds).all())}"
                )

            if not torch.isfinite(audio_embeds).all():
                raise RuntimeError(
                    "audio_embeds contains non-finite values (nan/inf) after the audio "
                    "encoder + adapter -- the audio encoder produced a bad embedding "
                    "before generation even started, not a decoding-time instability."
                )

            audio_token_count = int(audio_input_mask.to(torch.int32).sum().item())
            if audio_token_count != int(audio_embeds.shape[1]):
                raise ValueError(
                    f"Audio token count mismatch: audio_input_mask has {audio_token_count} audio tokens, "
                    f"but audio_embeds has length {int(audio_embeds.shape[1])}."
                )

            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

            if deepstack is not None and len(self.deepstack_audio_merger_list) > 0:
                deepstack_audio_embeds = []
                for i, x in enumerate(deepstack[: len(self.deepstack_audio_merger_list)]):
                    ds = self.deepstack_audio_merger_list[i](x)
                    if int(ds.shape[1]) != audio_token_count:
                        raise ValueError(
                            f"DeepStack audio seq_len mismatch at index {i}: "
                            f"expected {audio_token_count}, got {int(ds.shape[1])}."
                        )
                    deepstack_audio_embeds.append(ds)

                try:
                    hook_handles = self._register_llm_deepstack_hooks(audio_input_mask, deepstack_audio_embeds)
                except Exception:
                    for h in hook_handles:
                        h.remove()
                    raise

        try:
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            for h in hook_handles:
                h.remove()

        hidden_states = outputs[0]
        # transformers' GenerationMixin.generate() only detects support for this
        # optimization via inspect.signature() on an explicit `logits_to_keep`
        # parameter (generation/utils.py's _supports_logits_to_keep()), so it must
        # be declared above rather than left to **kwargs. Without slicing here,
        # every prefill call materializes lm_head logits for the whole input
        # sequence -- including every audio token -- instead of just the last
        # position needed to sample the next token, which is what was causing
        # CUDA OOMs on long audio inputs.
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"logits contain non-finite values (nan/inf) after lm_head, "
                f"logits.shape={tuple(logits.shape)}, "
                f"had_audio_this_call={audio_data is not None}, "
                f"hidden_states.isfinite={bool(torch.isfinite(hidden_states).all())}. "
                "This is raised here to fail with a Python traceback before "
                "generate()'s multinomial sampling hits CUDA's device-side "
                "assert, which aborts the whole process instead of raising "
                "a catchable exception."
            )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            shift_logits = shift_logits.view(-1, self.config.language_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration: bool = False,
        **kwargs,
    ):
        # transformers' GenerationMixin.prepare_inputs_for_generation (base
        # implementation) changed in a way upstream MOSS-Audio's override
        # doesn't account for: input-slicing is now driven by the
        # `next_sequence_length` kwarg passed in by `generate()`, and
        # `cache_position` is no longer threaded through `model_kwargs`
        # between steps (the old `cache_position[0] > 0` check upstream
        # used to detect decode-vs-prefill therefore always saw a stale/
        # absent cache_position and never sliced `input_ids` down to the
        # newest token, growing `input_ids` one token ahead of the
        # prefill-length `audio_input_mask` on every decode step). Delegate
        # the standard slicing/mask/position-id bookkeeping to the base
        # implementation, using its `is_first_iteration` flag -- which
        # `generate()` does pass correctly -- to gate the audio-specific
        # inputs instead.
        audio_data = kwargs.pop("audio_data", None)
        audio_input_mask = kwargs.pop("audio_input_mask", None)
        audio_data_seqlens = kwargs.pop("audio_data_seqlens", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration:
            model_inputs["audio_data"] = audio_data
            model_inputs["audio_input_mask"] = audio_input_mask
            model_inputs["audio_data_seqlens"] = audio_data_seqlens
        else:
            model_inputs["audio_data"] = None
            model_inputs["audio_input_mask"] = None
            model_inputs["audio_data_seqlens"] = None

        return model_inputs


__all__ = [
    "MossAudioEncoderConfig",
    "MossAudioConfig",
    "MossAudioModel",
]
