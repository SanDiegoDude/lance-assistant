"""Monkey-patch HF's Qwen2_5_VLAttention to add Qwen3-style per-head QK-norm.

Lance fine-tuned a Qwen2.5-VL-3B backbone but added `q_norm`/`k_norm`
RMSNorms (shape `[head_dim]`) to every layer's self-attention. HF's stock
implementation has no such modules, so the corresponding weights are
silently discarded and the model produces gibberish.

This module registers `q_norm`/`k_norm` modules on each
`Qwen2_5_VLAttention` and rewires its `forward` to apply them after the
projection / head-split but before RoPE. Apply *before* loading the model.

Usage:
    from lance.qknorm_patch import patch_qwen2_5_vl_qknorm
    patch_qwen2_5_vl_qknorm()
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


_PATCHED = False


def patch_qwen2_5_vl_qknorm() -> None:
    """Idempotent: install QK-norm on transformers.Qwen2_5_VLAttention."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLAttention,
        Qwen2RMSNorm,
        apply_multimodal_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )

    original_init = Qwen2_5_VLAttention.__init__

    def patched_init(self, config, layer_idx: Optional[int] = None) -> None:
        original_init(self, config, layer_idx)
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=eps)

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Qwen3-style: RMSNorm over head_dim on Q and K, before RoPE.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    M.Qwen2_5_VLAttention.__init__ = patched_init
    M.Qwen2_5_VLAttention.forward = patched_forward
    _PATCHED = True
