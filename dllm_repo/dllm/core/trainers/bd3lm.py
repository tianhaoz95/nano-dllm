"""
References:

Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models:
https://arxiv.org/abs/2503.09573
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.utils.collators import CollatorWrapper

from .mdlm import MDLMConfig, MDLMTrainer


@dataclass
class AppendEOSBlockWrapper(CollatorWrapper):
    block_size: int = 32

    def before(self, features):
        for ex in features:
            ids = ex["input_ids"]
            labs = ex["labels"]

            assert isinstance(ids, list) and isinstance(labs, list)

            L = len(ids)
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                ex["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
                ex["labels"] = labs + [self.tokenizer.eos_token_id] * pad_len
        return features


def _create_bd3lm_attention_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


@dataclass
class BD3LMConfig(MDLMConfig):
    block_size: int = 32


class BD3LMTrainer(MDLMTrainer):

    def __init__(
        self,
        args: BD3LMConfig,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.block_size = args.block_size

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self, "_preprocess_inputs"):
            inputs = self._preprocess_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # Capture accuracy from metrics if present
        if hasattr(outputs, "metrics") and "accuracy" in outputs.metrics:
            self._current_accuracy = outputs.metrics["accuracy"].detach().item()

        return loss.detach() / self.args.gradient_accumulation_steps

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if hasattr(self, "_current_accuracy"):
            logs["accuracy"] = self._current_accuracy
        super().log(logs, *args, **kwargs)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: list[dict[str, Any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the block diffusion language modeling loss.
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        # === 2. Apply stochastic masking ===
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # ── Forward pass through the model (block-diffusion) ──
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        unwrapped_model = self.accelerator.unwrap_model(model)

        # Cache the attention mask to avoid redundant creation/transfer
        if unwrapped_model.config._attn_implementation == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask
            if not hasattr(self, "_cached_mask_params") or self._cached_mask_params != (l, self.block_size, input_ids.device):
                self._cached_mask = create_block_mask(
                    partial(_create_bd3lm_attention_mask, block_size=self.block_size, n=l),
                    B=None, H=None, Q_LEN=l * 2, KV_LEN=l * 2,
                    device=input_ids.device
                )
                self._cached_mask_params = (l, self.block_size, input_ids.device)
            attention_mask = self._cached_mask
        elif unwrapped_model.config._attn_implementation == "sdpa":
            if not hasattr(self, "_cached_mask_params") or self._cached_mask_params != (l, self.block_size, input_ids.device):
                attention_mask = _create_bd3lm_attention_mask(
                    b=None, h=None,
                    q_idx=torch.arange(l * 2, device=input_ids.device)[:, None],
                    kv_idx=torch.arange(l * 2, device=input_ids.device)[None, :],
                    block_size=self.block_size,
                    n=l,
                )
                self._cached_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
                self._cached_mask_params = (l, self.block_size, input_ids.device)
            attention_mask = self._cached_mask
        else:
            attention_mask = None

        base_pos = (
            torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        )
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        # Optimization: Only compute logits for the tokens we care about (the first half x_t)
        # However, transformers.Trainer.compute_loss usually calls model() which computes all logits.
        # We unwrap and call the backbone directly if possible.
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # Check if it's our A2D model that supports returning hidden states
        if hasattr(unwrapped_model, "model") and hasattr(unwrapped_model, "lm_head"):
            hidden_states = unwrapped_model.model(
                input_ids=concat_input_ids,
                attention_mask=attention_mask,
                position_ids=concat_position_ids,
            ).last_hidden_state
            
            # Use only the first half and only masked positions for the LM head
            # This is a huge optimization for both compute and memory
            masked_hidden = hidden_states[:, :l][masked_mask]
            masked_labels = input_ids[masked_mask]
            
            if masked_hidden.shape[0] > 0:
                logits = unwrapped_model.lm_head(masked_hidden)
                loss_weights = self._compute_loss_weights(t=t, inputs=inputs, masked_mask=masked_mask)
                # Compute weight per masked token
                masked_weights = loss_weights[masked_mask]
                
                loss = F.cross_entropy(logits, masked_labels, reduction="none")
                loss = (loss * masked_weights).sum()

                # Calculate Accuracy
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    accuracy = (preds == masked_labels).float().mean()
                
                # Normalize loss
                if self.loss_norm_type == "token":
                    loss /= maskable_mask.sum().clamp_min(1)
                elif self.loss_norm_type == "sequence":
                    loss /= (maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b).sum()
                elif self.loss_norm_type == "batch":
                    loss /= b
                
                # Mock outputs for compatibility if needed
                outputs = transformers.modeling_outputs.MaskedLMOutput(loss=loss, logits=None)
                if not hasattr(outputs, "metrics"):
                    outputs.metrics = {}
                outputs.metrics["accuracy"] = accuracy.detach()
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                outputs = transformers.modeling_outputs.MaskedLMOutput(loss=loss, logits=None)
                if not hasattr(outputs, "metrics"):
                    outputs.metrics = {}
                outputs.metrics["accuracy"] = torch.tensor(0.0, device=input_ids.device)
        else:
            # Fallback to standard slow path
            outputs = model(
                input_ids=concat_input_ids,
                attention_mask=attention_mask,
                position_ids=concat_position_ids,
            )
            logits = outputs.logits[:, :l]
            loss_weights = self._compute_loss_weights(t=t, inputs=inputs, masked_mask=masked_mask)
            loss = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
            loss = (loss * loss_weights * masked_mask.to(loss.dtype)).sum()

            # Calculate Accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Only care about masked positions
                mask = masked_mask
                if mask.any():
                    accuracy = (preds[mask] == input_ids[mask]).float().mean()
                else:
                    accuracy = torch.tensor(0.0, device=input_ids.device)
            
            if self.loss_norm_type == "token":
                loss /= maskable_mask.sum().clamp_min(1)
            elif self.loss_norm_type == "sequence":
                loss /= (maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b).sum()
            elif self.loss_norm_type == "batch":
                loss /= b
            outputs.loss = loss
            if not hasattr(outputs, "metrics"):
                outputs.metrics = {}
            outputs.metrics["accuracy"] = accuracy.detach()

        return (loss, outputs) if return_outputs else loss

