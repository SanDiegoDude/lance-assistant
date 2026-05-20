"""Flow-matching text-to-image / text-to-video sampling for Lance.

Implements the BAGEL-style flow-matching loop adapted for Lance:

  1. Tokenize a prompt with the Qwen chat template.
  2. Build a packed sequence  [prompt tokens] <|vision_start|>
                              <|image_pad|> x N_latents <|vision_end|>
     where N_latents = T_lat * H_lat * W_lat for the desired image/video.
  3. Initialise x_T ~ N(0, I) in the 48-channel VAE latent space.
  4. For t in linspace(1, 0, steps):
        emb_lat = vae2llm(x_t) + time_embedder(t) + latent_pos_embed[MaPE_idx]
        h = LanceForGeneration.forward_packed(...) with bidirectional attn
        v_t = llm2vae(h[latent_positions])
        x_t <- x_t - v_t * dt           (rectified flow, data->noise dir)
  5. Decode x_0 with the Wan 2.2 VAE -> pixels.

We support both a "conditional only" sampler (`cfg_scale = 1.0`) and a
classifier-free guidance variant using an empty-prompt second forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer

from .lance_model import LanceConfig, LanceForGeneration


VISION_START_TOKEN = 151652
VISION_END_TOKEN = 151653
VISION_PAD_TOKEN = 151654
IMAGE_PAD_TOKEN = 151655
VIDEO_PAD_TOKEN = 151656
IM_START_TOKEN = 151644
IM_END_TOKEN = 151645


@dataclass
class T2IRequest:
    prompt: str
    height: int = 512                # in pixels
    width: int = 512                 # in pixels
    num_steps: int = 24
    timestep_shift: float = 1.0
    cfg_scale: float = 1.0
    negative_prompt: str = ""
    seed: Optional[int] = None
    # generation only takes a single image, so T_pix = 1 + 4 * (T_lat - 1).
    # For T2V set num_frames > 1 (must be one of {1, 5, 9, ..., 121}).
    num_frames: int = 1
    # CFG is only applied when t is in this interval. BAGEL uses (0.4, 1.0)
    # which empirically prevents low-t velocity blow-up in long sequences.
    # Use (0.0, 1.0) to apply CFG everywhere (original behavior).
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    # BAGEL-style global CFG renorm: rescale the CFG-mixed velocity so its
    # norm equals the conditional velocity's norm (clamped). 0.0 disables.
    cfg_renorm_min: float = 0.0


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def build_t2i_prompt_ids(
    tok, prompt: str, n_latent_tokens: int, *, video: bool = False
) -> torch.LongTensor:
    """Build the packed token sequence: chat-style prompt + vision span.

    For video generation we substitute `<|video_pad|>` for `<|image_pad|>` so
    that Lance's gen-expert knows it's emitting temporally-extended latents.
    """
    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    base_ids: list[int] = tok(text, add_special_tokens=False).input_ids
    base_ids.append(VISION_START_TOKEN)
    pad_token = VIDEO_PAD_TOKEN if video else IMAGE_PAD_TOKEN
    base_ids.extend([pad_token] * n_latent_tokens)
    base_ids.append(VISION_END_TOKEN)
    return torch.tensor(base_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# MaPE indexing
# ---------------------------------------------------------------------------


def build_3d_causal_mask(
    S: int,
    latent_positions: torch.Tensor,
    t_lat: int, h_lat: int, w_lat: int,
    device,
) -> torch.Tensor:
    """Lance's "generalized 3D causal attention".

    Layout assumption: packed sequence is

        [ text prefix ... <vision_start> <pad>*N_lat <vision_end> ]

    Rules:
      - text+vision_start (prefix): bidirectional within itself
      - latents at frame t_i: attend to prefix + all latents at frame t_j <= t_i
      - vision_end (suffix): attends to everything before it
      - prefix tokens never attend to future latents/suffix (standard causal
        boundary)

    Returns a bool [S, S] mask where True = "i can attend to j".
    """
    spatial = h_lat * w_lat
    n_lat = t_lat * spatial
    assert latent_positions.numel() == n_lat, "latent_positions count mismatch"

    # is-latent / is-prefix / is-suffix per position
    is_latent = torch.zeros(S, dtype=torch.bool, device=device)
    is_latent[latent_positions.to(device)] = True
    first_lat = int(latent_positions[0].item())
    last_lat = int(latent_positions[-1].item())
    is_prefix = torch.zeros(S, dtype=torch.bool, device=device)
    is_prefix[:first_lat] = True
    is_suffix = torch.zeros(S, dtype=torch.bool, device=device)
    is_suffix[last_lat + 1:] = True

    # frame index for each latent token (in packed-sequence order, latent_k
    # corresponds to frame k // spatial)
    frame_idx_lat = torch.arange(n_lat, device=device) // spatial  # [n_lat]
    frame_idx = torch.full((S,), -1, dtype=torch.long, device=device)
    frame_idx[latent_positions.to(device)] = frame_idx_lat

    # mask[i, j] = "i attends to j"
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)

    # i in prefix:
    mask[is_prefix.unsqueeze(1) & is_prefix.unsqueeze(0)] = True

    # i in latents -> prefix
    mask[is_latent.unsqueeze(1) & is_prefix.unsqueeze(0)] = True

    # i in latents -> latents (causal across frames)
    fi = frame_idx.unsqueeze(1)  # [S, 1]
    fj = frame_idx.unsqueeze(0)  # [1, S]
    lat_to_lat = is_latent.unsqueeze(1) & is_latent.unsqueeze(0) & (fj <= fi)
    mask = mask | lat_to_lat

    # i in suffix -> attends to everything before (and itself)
    suf_to_pre_lat = is_suffix.unsqueeze(1) & (is_prefix.unsqueeze(0) | is_latent.unsqueeze(0))
    mask = mask | suf_to_pre_lat
    mask = mask | (is_suffix.unsqueeze(1) & is_suffix.unsqueeze(0))

    return mask


def make_mape_indices(t_lat: int, h_lat: int, w_lat: int, max_per_side: int = 64) -> torch.LongTensor:
    """MaPE index per latent token, BAGEL/Lance "extrapolate" scheme.

    The positional bank is sized assuming a virtual MAX_PER_SIDE x MAX_PER_SIDE
    grid per frame (and MAX_T frames for video). The id for a latent at (t,h,w)
    is::

        id = t * (max_per_side ** 2) + h * max_per_side + w

    Smaller-than-max grids therefore use a *sparse* subset of the bank
    (e.g. a 32x32 latent grid uses ids {h*64 + w : h,w < 32}, leaving the rest
    untouched). This is what BAGEL's `get_flattened_position_ids_extrapolate`
    does and matches Lance_3B's max_per_side=64 (since 4096 = 64**2).
    """
    grid_t = torch.arange(t_lat).view(-1, 1, 1).expand(t_lat, h_lat, w_lat)
    grid_h = torch.arange(h_lat).view(1, -1, 1).expand(t_lat, h_lat, w_lat)
    grid_w = torch.arange(w_lat).view(1, 1, -1).expand(t_lat, h_lat, w_lat)
    flat = grid_t * (max_per_side * max_per_side) + grid_h * max_per_side + grid_w
    return flat.reshape(-1).long()


# ---------------------------------------------------------------------------
# Wan 2.2 VAE compression schedule
# ---------------------------------------------------------------------------


# Wan 2.2 VAE: 4x temporal compression with the first frame as a key frame,
# and 16x spatial compression (8x from conv stages + 2x from patchify).
def vae_spatial_compression() -> int:
    return 16


def vae_temporal_to_latent(num_frames: int) -> int:
    if num_frames == 1:
        return 1
    # First frame is its own latent, then each subsequent group of 4 -> 1 latent.
    assert (num_frames - 1) % 4 == 0, "num_frames must be 1 or 1 + 4k"
    return 1 + (num_frames - 1) // 4


# ---------------------------------------------------------------------------
# Flow-matching schedule
# ---------------------------------------------------------------------------


def flow_match_schedule(num_steps: int, shift: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (timesteps[num_steps], dts[num_steps]).

    timesteps[i] = t at step i (decreasing 1 -> 0); we always include t=0 in
    the breakpoint set, so the integrator uses N actual steps.
    """
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    if shift != 1.0:
        ts = shift * ts / (1.0 + (shift - 1.0) * ts)
    dts = ts[:-1] - ts[1:]  # positive
    return ts[:-1].to(dtype), dts.to(dtype)


# ---------------------------------------------------------------------------
# Core T2I / T2V generator
# ---------------------------------------------------------------------------


class LanceGenerator:
    """High-level wrapper around LanceForGeneration for T2I / T2V."""

    def __init__(
        self,
        model: LanceForGeneration,
        tokenizer,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.tok = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.eval()

    @torch.no_grad()
    def _flow_forward(
        self,
        prompt_ids: torch.LongTensor,     # [S_prompt]
        x_t: torch.Tensor,                # [N_lat, C=48]  flattened latent tokens
        t_lat: int, h_lat: int, w_lat: int,
        t_scalar: torch.Tensor,           # scalar t in [0, 1]
    ) -> torch.Tensor:
        """Run the LLM once and return predicted velocity for the latent tokens."""
        device = self.device
        n_lat = x_t.shape[0]

        # latent embed = vae2llm(x_t) + time_embed(t) + MaPE
        e_lat = self.model.vae2llm(x_t.to(self.dtype))
        t_emb = self.model.time_embedder(t_scalar.to(torch.float32).expand(n_lat).to(device))
        mape_idx = make_mape_indices(t_lat, h_lat, w_lat).to(device)
        pos_emb = self.model.latent_pos_embed_pos_embed[mape_idx].to(self.dtype)
        latent_embeds = e_lat + t_emb.to(self.dtype) + pos_emb

        # Build packed sequence
        # Layout: prompt_ids (text)   <vision_start>   image_pad*N   <vision_end>
        # We need:
        #   - full token_ids vector with shape [S]
        #   - position_ids for RoPE (text gets 0..len(text)-1, latents share a single id)
        S_prompt = prompt_ids.shape[0]            # includes <vision_start>, pad..., <vision_end>
        S = S_prompt
        token_ids = prompt_ids.to(device)
        # latent positions inside the packed sequence
        # in prompt_ids we already inserted IMAGE_PAD or VIDEO_PAD x n_lat
        # between v_start and v_end. Accept either.
        latent_positions_mask = (token_ids == IMAGE_PAD_TOKEN) | (token_ids == VIDEO_PAD_TOKEN)
        latent_positions = torch.nonzero(latent_positions_mask, as_tuple=False).flatten()
        assert latent_positions.numel() == n_lat, (
            f"prompt_ids must contain exactly n_lat={n_lat} IMAGE_PAD tokens; got {latent_positions.numel()}"
        )

        # Text uses 0..S-1; all latents share the first IMAGE_PAD position.
        # (This is what the image checkpoint was empirically trained with —
        # the BAGEL-style "whole vision block shares one position" scheme
        # causes velocities to diverge for both image and video Lance ckpts.)
        position_ids = torch.arange(S, device=device, dtype=torch.long)
        first_lat = int(latent_positions[0].item())
        position_ids[latent_positions] = first_lat

        # Lance paper describes "generalized 3D causal attention". For T2I
        # (t_lat==1) this collapses to bidirectional and matches the image
        # path. For T2V, causal-across-time + bidirectional-within-frame
        # produces structurally correct first frames; later frames still
        # drift somewhat but mode collapse is avoided.
        attn_mask = None
        if t_lat > 1:
            attn_mask = build_3d_causal_mask(
                S=S, latent_positions=latent_positions,
                t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, device=device,
            )
        h = self.model.forward_packed(
            token_ids=token_ids,
            latent_embeds=latent_embeds,
            latent_positions=latent_positions,
            position_ids=position_ids,
            is_causal=False,
            attn_mask=attn_mask,
        )
        v_t = self.model.llm2vae(h[latent_positions])  # [N_lat, 48]
        return v_t.to(torch.float32)

    @torch.no_grad()
    def generate(self, req: T2IRequest) -> torch.Tensor:
        """Run T2I/T2V flow-matching loop. Returns latent of shape [48, T_lat, H_lat, W_lat]."""
        S = vae_spatial_compression()
        assert req.height % S == 0 and req.width % S == 0, (
            f"height/width must be multiples of {S}; got {req.height}x{req.width}"
        )
        h_lat = req.height // S
        w_lat = req.width // S
        t_lat = vae_temporal_to_latent(req.num_frames)
        n_lat = t_lat * h_lat * w_lat

        if n_lat > self.model.config.max_latent_tokens:
            raise RuntimeError(
                f"requested {n_lat} latent tokens (T={t_lat} H={h_lat} W={w_lat}) "
                f"but MaPE bank only has {self.model.config.max_latent_tokens} rows"
            )

        # Empirically the video checkpoint was trained with `<|image_pad|>` as
        # the latent placeholder for BOTH image and video — feeding
        # `<|video_pad|>` blows up the velocity norms badly.
        prompt_ids = build_t2i_prompt_ids(
            self.tok, req.prompt, n_lat, video=False
        ).to(self.device)

        gen = torch.Generator(device="cpu")
        if req.seed is not None:
            gen.manual_seed(req.seed)
        x_t = torch.randn(n_lat, 48, generator=gen, dtype=torch.float32).to(self.device)

        ts, dts = flow_match_schedule(req.num_steps, req.timestep_shift, device=self.device, dtype=torch.float32)

        cfg = req.cfg_scale
        uncond_prompt_ids = None
        if cfg > 1.0:
            uncond_prompt_ids = build_t2i_prompt_ids(
                self.tok, req.negative_prompt, n_lat, video=False
            ).to(self.device)

        cfg_lo, cfg_hi = req.cfg_interval
        for i, t in enumerate(ts):
            t_val = t.item()
            v = self._flow_forward(prompt_ids, x_t, t_lat, h_lat, w_lat, t)
            use_cfg = cfg > 1.0 and cfg_lo <= t_val <= cfg_hi
            if use_cfg:
                v_uncond = self._flow_forward(uncond_prompt_ids, x_t, t_lat, h_lat, w_lat, t)
                v_mixed = v_uncond + cfg * (v - v_uncond)
                if req.cfg_renorm_min > 0.0:
                    # global CFG renorm: rescale mixed velocity to match
                    # conditional velocity's norm (clamped to [renorm_min, 1]).
                    norm_v = v.norm()
                    norm_mixed = v_mixed.norm()
                    scale = (norm_v / (norm_mixed + 1e-8)).clamp(
                        min=req.cfg_renorm_min, max=1.0
                    )
                    v = v_mixed * scale
                else:
                    v = v_mixed
            x_t = x_t - v * dts[i].item()
            print(
                f"  step {i + 1:3d}/{len(ts)}  t={t_val:.4f}  "
                f"|x_t|={x_t.norm().item():.3f}  |v|={v.norm().item():.3f}"
                f"  cfg={'on' if use_cfg else 'off'}",
                flush=True,
            )

        # reshape to [48, T_lat, H_lat, W_lat]
        x = x_t.view(t_lat, h_lat, w_lat, 48).permute(3, 0, 1, 2).contiguous()
        return x
