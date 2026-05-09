from typing import Optional

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks.
    # Must NOT be torch.Tensor — that would make isinstance(any_tensor, BlockMask) always
    # True and silently break the mask-conversion logic below.
    class BlockMask:
        pass

class A2DQwen3Config(transformers.Qwen3Config):
    model_type = "a2d-qwen3"  # <- NEW model_type


class A2DQwen3Model(transformers.Qwen3Model):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # 1) If no mask is provided → treat all tokens as valid (no padding)
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long
                )

            # 2) If mask is already 4D or BlockMask, use it for all layers
            if (
                isinstance(attention_mask, BlockMask)
                or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
            ):
                causal_mask_mapping = {"full_attention": attention_mask}
                if self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = attention_mask
            else:
                # 3) Otherwise, decide between Causal (Generation) or Bidirectional (Training)
                # FIX: If we are not training and past_key_values is None, we likely want a causal mask
                # for the current segment (e.g. during sampling steps without KV cache).
                if past_key_values is not None or not self.training:
                    # GENERATION or EVAL: Use original causal mask logic
                    from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask
                    mask_kwargs = {
                        "config": self.config,
                        "input_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        "cache_position": cache_position,
                        "past_key_values": past_key_values,
                        "position_ids": position_ids,
                    }
                    causal_mask_mapping = {
                        "full_attention": create_causal_mask(**mask_kwargs),
                    }
                    if self.has_sliding_layers:
                        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
                else:
                    # TRAINING: Use bidirectional padding-only mask
                    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
                    attention_mask = AttentionMaskConverter._expand_mask(
                        mask=attention_mask,
                        dtype=self.dtype,
                        tgt_len=inputs_embeds.shape[1]
                    )
                    causal_mask_mapping = {"full_attention": attention_mask}
                    if self.has_sliding_layers:
                        causal_mask_mapping["sliding_attention"] = attention_mask

        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Get the correct mask for this layer
            layer_mask = None
            if isinstance(causal_mask_mapping, dict):
                layer_mask = causal_mask_mapping.get(decoder_layer.attention_type, None)
            else:
                layer_mask = causal_mask_mapping
            
            # -------------------------------------------------------------
            # FIX: Ensure mask shape compatibility for SDPA re-expansion
            # -------------------------------------------------------------
            if layer_mask is not None and not isinstance(layer_mask, BlockMask) and layer_mask.ndim == 4:
                # Standard SDPA mask shape is [batch, 1, query_length, key_length]
                if hidden_states.shape[1] < layer_mask.shape[2]:
                    # Slicing is okay, it results in a singleton that can be expanded later
                    layer_mask = layer_mask[:, :, -hidden_states.shape[1]:, :]
                
                # Check if the mask key_length matches the current KV length.
                past_len = past_key_values.get_seq_length(i) if past_key_values is not None else 0
                current_kv_len = past_len + hidden_states.shape[1]
                if layer_mask.shape[3] != current_kv_len:
                    if layer_mask.shape[3] == 1:
                        # Singleton dimension is expand-friendly in SDPA
                        pass
                    else:
                        # Mismatched length. We MUST slice it to avoid 'expand' error.
                        if layer_mask.shape[3] > current_kv_len:
                            layer_mask = layer_mask[:, :, :, :current_kv_len]
                        else:
                            # Mismatched and too short. Try to slice query as well to force match?
                            # No, the query is already sliced. If key length is still mismatched,
                            # we have a serious state inconsistency. Fallback to None.
                            if self.config._attn_implementation == "sdpa":
                                layer_mask = None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class A2DQwen3LMHeadModel(transformers.Qwen3ForCausalLM):
    supports_gradient_checkpointing = True
    config: A2DQwen3Config

    def __init__(self, config):
        transformers.Qwen3PreTrainedModel.__init__(self, config)
        self.model = A2DQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


transformers.AutoConfig.register("a2d-qwen3", A2DQwen3Config)
transformers.AutoModel.register(A2DQwen3Config, A2DQwen3LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen3Config, A2DQwen3LMHeadModel)
