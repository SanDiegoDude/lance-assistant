"""Lance unified multimodal model — dual-expert MoT inference.

This is a from-scratch torch reimplementation that loads weights directly
from `Lance_3B/model.safetensors` (or `Lance_3B_Video/model.safetensors`).
We do NOT subclass HF transformers because their `Qwen2_5_VL` forward pass
has no concept of per-token modality routing.

Architectural correspondence (see `notes/ARCHITECTURE.md`):

  - Each of 36 decoder layers has TWO sets of weights:
      * "understanding expert" (no suffix)  – used for text and ViT tokens
      * "generation expert"   (_moe_gen)    – used for noisy VAE latents
    Both experts share a single multi-head attention pass (full S x S),
    but with separate Q/K/V/O projections, separate per-head q/k RMSNorm,
    separate input/post-attention RMSNorm and separate SwiGLU MLP.

  - Q and K get Qwen3-style per-head RMSNorm before RoPE.

  - The generation pathway adds:
      vae2llm        : Linear(48 -> 2048)
      llm2vae        : Linear(2048 -> 48)         (predicts flow velocity)
      latent_pos_embed.pos_embed: [4096 or 126976, 2048]   (MaPE bank)
      time_embedder  : Linear(256 -> 2048) -> SiLU -> Linear(2048 -> 2048)

This file is the "core" model; the high-level T2I / T2V generate loop lives
in `lance.flow_match`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class LanceConfig:
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 128000
    mrope_section: tuple = (16, 24, 24)
    vae_latent_dim: int = 48
    time_freq_dim: int = 256
    max_latent_tokens: int = 4096  # 4096 for Lance_3B, 126976 for video

    @classmethod
    def from_llm_config(cls, path: Path | str, *, video: bool = False) -> "LanceConfig":
        cfg = json.loads(Path(path).read_text())
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        mrope_section = tuple(cfg["rope_scaling"]["mrope_section"])
        return cls(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            max_position_embeddings=cfg["max_position_embeddings"],
            mrope_section=mrope_section,
            max_latent_tokens=126976 if video else 4096,
        )


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class SwiGLU(nn.Module):
    """Qwen2 MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding followed by a 2-layer MLP. Matches BAGEL/DiT."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dtype = self.mlp[0].weight.dtype
        return self.mlp(self.sinusoidal(t, self.freq_dim).to(dtype))


# ---------------------------------------------------------------------------
# multimodal RoPE (Qwen 2.5-VL style)
# ---------------------------------------------------------------------------


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) with shape [seq_len, head_dim] for standard 1D RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [S, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope_1d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard 1D RoPE applied to last dim.

    q, k: [S, H, D]
    cos, sin: [S, D]
    """

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    cos = cos.unsqueeze(1)  # [S, 1, D]
    sin = sin.unsqueeze(1)
    q_out = (q.float() * cos.float() + rotate_half(q.float()) * sin.float()).to(q.dtype)
    k_out = (k.float() * cos.float() + rotate_half(k.float()) * sin.float()).to(k.dtype)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Dual-expert attention and MLP
# ---------------------------------------------------------------------------


def _split_apply(x: torch.Tensor, mask: torch.Tensor, fn_u, fn_g, out_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Apply fn_u to x[~mask] and fn_g to x[mask], scatter back."""
    if mask.all():
        return fn_g(x)
    if not mask.any():
        return fn_u(x)
    out_u = fn_u(x[~mask])
    out_g = fn_g(x[mask])
    shape = list(out_shape) if out_shape is not None else list(out_u.shape)
    shape[0] = x.shape[0]
    out = out_u.new_empty(shape)
    out[~mask] = out_u
    out[mask] = out_g
    return out


class LanceAttention(nn.Module):
    def __init__(self, config: LanceConfig):
        super().__init__()
        H, D = config.num_attention_heads, config.head_dim
        Kh = config.num_key_value_heads
        hidden = config.hidden_size
        self.num_heads = H
        self.num_kv_heads = Kh
        self.head_dim = D
        self.kv_groups = H // Kh

        self.q_proj = nn.Linear(hidden, H * D, bias=True)
        self.k_proj = nn.Linear(hidden, Kh * D, bias=True)
        self.v_proj = nn.Linear(hidden, Kh * D, bias=True)
        self.o_proj = nn.Linear(H * D, hidden, bias=False)
        self.q_norm = RMSNorm(D, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(D, eps=config.rms_norm_eps)

        # generation expert
        self.q_proj_moe_gen = nn.Linear(hidden, H * D, bias=True)
        self.k_proj_moe_gen = nn.Linear(hidden, Kh * D, bias=True)
        self.v_proj_moe_gen = nn.Linear(hidden, Kh * D, bias=True)
        self.o_proj_moe_gen = nn.Linear(H * D, hidden, bias=False)
        self.q_norm_moe_gen = RMSNorm(D, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(D, eps=config.rms_norm_eps)

    def forward(
        self,
        h: torch.Tensor,             # [S, hidden]
        mod_mask: torch.Tensor,      # [S] bool, True = gen
        cos: torch.Tensor,           # [S, head_dim]
        sin: torch.Tensor,           # [S, head_dim]
        is_causal: bool,
        attn_mask: Optional[torch.Tensor] = None,  # [S, S] bool, True = allowed
    ) -> torch.Tensor:
        S = h.shape[0]
        H, Kh, D = self.num_heads, self.num_kv_heads, self.head_dim

        # Q/K/V projections, dual-expert
        q = _split_apply(h, mod_mask, self.q_proj, self.q_proj_moe_gen)
        k = _split_apply(h, mod_mask, self.k_proj, self.k_proj_moe_gen)
        v = _split_apply(h, mod_mask, self.v_proj, self.v_proj_moe_gen)

        q = q.view(S, H, D)
        k = k.view(S, Kh, D)
        v = v.view(S, Kh, D)

        # q/k norms, dual-expert
        q = _split_apply(q, mod_mask, self.q_norm, self.q_norm_moe_gen)
        k = _split_apply(k, mod_mask, self.k_norm, self.k_norm_moe_gen)

        q, k = apply_rope_1d(q, k, cos, sin)

        # GQA: repeat K/V to match Q heads
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # [B=1, H, S, D] for SDPA
        q = q.transpose(0, 1).unsqueeze(0).contiguous()
        k = k.transpose(0, 1).unsqueeze(0).contiguous()
        v = v.transpose(0, 1).unsqueeze(0).contiguous()

        if attn_mask is not None:
            # SDPA wants either bool or float mask of shape [..., S, S]; we
            # pass bool (True = keep) which SDPA converts to -inf for False.
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.squeeze(0).transpose(0, 1).contiguous().view(S, H * D)

        out = _split_apply(attn, mod_mask, self.o_proj, self.o_proj_moe_gen)
        return out


class LanceDecoderLayer(nn.Module):
    def __init__(self, config: LanceConfig):
        super().__init__()
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.self_attn = LanceAttention(config)
        self.mlp = SwiGLU(hidden, config.intermediate_size)
        self.mlp_moe_gen = SwiGLU(hidden, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps)
        self.input_layernorm_moe_gen = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden, eps)

    def forward(
        self,
        h: torch.Tensor,
        mod_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = h
        h_norm = _split_apply(h, mod_mask, self.input_layernorm, self.input_layernorm_moe_gen)
        h = residual + self.self_attn(h_norm, mod_mask, cos, sin, is_causal, attn_mask=attn_mask)

        residual = h
        h_norm = _split_apply(h, mod_mask, self.post_attention_layernorm, self.post_attention_layernorm_moe_gen)
        h = residual + _split_apply(h_norm, mod_mask, self.mlp, self.mlp_moe_gen)
        return h


# ---------------------------------------------------------------------------
# Full Lance generation model
# ---------------------------------------------------------------------------


class LanceForGeneration(nn.Module):
    """Lance unified model with both experts loaded; used for image/video gen."""

    def __init__(self, config: LanceConfig):
        super().__init__()
        self.config = config

        # main LM
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LanceDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # generation pathway
        self.vae2llm = nn.Linear(config.vae_latent_dim, config.hidden_size, bias=True)
        self.llm2vae = nn.Linear(config.hidden_size, config.vae_latent_dim, bias=True)
        self.time_embedder = TimestepEmbedder(config.hidden_size, config.time_freq_dim)
        # MaPE — a learned positional bank in LLM hidden space, indexed per latent token.
        self.latent_pos_embed_pos_embed = nn.Parameter(
            torch.zeros(config.max_latent_tokens, config.hidden_size)
        )

    # ------------------------ weight loading -----------------------------

    def load_lance_safetensors(self, path: str | Path, *, dtype: torch.dtype = torch.bfloat16) -> dict[str, int]:
        """Load weights directly from a Lance safetensors file.

        Returns a dict with bookkeeping counts: kept / dropped / missing.
        """
        counts = {"kept": 0, "missing": 0, "skipped_vit": 0}
        own = self.state_dict()
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            for k in keys:
                if k.startswith("vit_model."):
                    counts["skipped_vit"] += 1
                    continue
                mapped = _lance_key_to_lance_model(k)
                if mapped is None:
                    continue
                if mapped not in own:
                    counts["missing"] += 1
                    continue
                t = f.get_tensor(k).to(dtype)
                if t.shape != own[mapped].shape:
                    raise RuntimeError(f"shape mismatch for {mapped}: ckpt {t.shape} vs model {own[mapped].shape}")
                own[mapped].copy_(t)
                counts["kept"] += 1
        return counts

    # ------------------------ forward pass --------------------------------

    def forward_packed(
        self,
        token_ids: torch.LongTensor,         # [S] int — input token ids (will be embedded)
        latent_embeds: Optional[torch.Tensor],  # [N_lat, hidden] or None — already embedded latent tokens
        latent_positions: Optional[torch.LongTensor],  # [N_lat] int — positions inside the [S] sequence where latents go
        position_ids: torch.LongTensor,      # [S] int — sequence positions for RoPE (1D, no mrope)
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,  # [S, S] bool mask (True = keep)
    ) -> torch.Tensor:
        """Run the LLM on a single packed sequence with mixed modalities.

        Returns hidden states at every position, with shape [S, hidden].
        Apply `self.llm2vae(h[latent_positions])` for velocity prediction.
        """
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
        S = token_ids.shape[0]

        # embed text tokens; overwrite at latent positions
        h = self.embed_tokens(token_ids).to(dtype)
        mod_mask = torch.zeros(S, dtype=torch.bool, device=device)
        if latent_embeds is not None and latent_positions is not None and latent_positions.numel() > 0:
            h[latent_positions] = latent_embeds.to(dtype)
            mod_mask[latent_positions] = True

        # RoPE cache: standard 1D positional RoPE (per-head)
        cos, sin = build_rope_cache(
            seq_len=int(position_ids.max().item()) + 1,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            device=device,
            dtype=dtype,
        )
        cos = cos[position_ids]  # [S, D]
        sin = sin[position_ids]

        # SDPA needs the mask broadcastable over [B=1, H, S, S]
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.to(device=device).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            h = layer(h, mod_mask, cos, sin, is_causal=is_causal, attn_mask=sdpa_mask)

        # final norm, dual expert
        h = _split_apply(h, mod_mask, self.norm, self.norm_moe_gen)
        return h


# ---------------------------------------------------------------------------
# key remap: ckpt name -> LanceForGeneration state_dict key
# ---------------------------------------------------------------------------


def _lance_key_to_lance_model(k: str) -> Optional[str]:
    """Remap a Lance ckpt key to our LanceForGeneration state_dict layout."""
    # top-level extras
    if k == "latent_pos_embed.pos_embed":
        return "latent_pos_embed_pos_embed"
    if k.startswith("time_embedder."):
        return k  # e.g. time_embedder.mlp.0.weight
    if k.startswith(("vae2llm.", "llm2vae.")):
        return k
    # LM body
    if k == "language_model.model.embed_tokens.weight":
        return "embed_tokens.weight"
    if k == "language_model.lm_head.weight":
        return "lm_head.weight"
    if k == "language_model.model.norm.weight":
        return "norm.weight"
    if k == "language_model.model.norm_moe_gen.weight":
        return "norm_moe_gen.weight"
    if k.startswith("language_model.model.layers."):
        rest = k[len("language_model.model.layers.") :]  # e.g. "0.self_attn.q_proj.weight"
        idx, sub = rest.split(".", 1)
        return f"layers.{idx}.{sub}"
    # vit_model.* handled by caller (we skip)
    return None
