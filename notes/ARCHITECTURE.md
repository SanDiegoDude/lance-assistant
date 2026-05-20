# Lance architecture — reverse-engineered from weights

Source: `bytedance-research/Lance` on Hugging Face (model card empty, no code
released as of 2026-05-16). Project page: <https://lance-project.github.io/>.

We reconstructed the architecture by reading the safetensors **headers** over
HTTP range requests (`scripts/inspect_safetensors_remote.py`), so all of the
findings below are from authoritative tensor names + shapes, not guesses.

Per-tensor dumps live alongside this file:
- `Lance_3B__model.safetensors.keys.txt`         (1021 tensors, 6.18B params)
- `Lance_3B_Video__model.safetensors.keys.txt`   (1411 tensors, 7.10B params)
- `Qwen2.5-VL-ViT__vit.safetensors.keys.txt`     ( 390 tensors, 0.67B params)


## TL;DR

Lance = **BAGEL** ([ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/BAGEL))
re-trained on a **Qwen2.5-VL-3B** backbone with a **Wan 2.2 VAE** (48-channel
latents) for generation, extended to video, and using a learned positional
bank ("MaPE") instead of BAGEL's 2D position system.

Every transformer block is a **Mixture-of-Transformers** (MoT): two parallel
sets of weights (understanding expert + generation expert, suffix `_moe_gen`)
inside a single shared self-attention pass. Tokens are routed to one expert
based on their modality (text / ViT-semantic → understanding, VAE-latent →
generation).


## Repo layout (bytedance-research/Lance)

```
Lance_3B/                          # image-only LLM checkpoint
  llm_config.json                  # Qwen2_5_VL config (hidden=2048, 36 layers)
  generation_config.json
  model.safetensors                # LLM + projectors + time embed + MaPE (no ViT)
  tokenizer.json, vocab.json, merges.txt

Lance_3B_Video/                    # video-extended checkpoint
  llm_config.json                  # same as Lance_3B
  model.safetensors                # ^ + ViT bundled + bigger MaPE (31 frames)
  tokenizer.json, tokenizer_config.json, vocab.json, merges.txt

Qwen2.5-VL-ViT/                    # shared semantic encoder (used by Lance_3B)
  config.json
  vit.safetensors                  # 668M params, BF16

Wan2.2_VAE.pth                     # generative VAE (Wan 2.2, 48 latent ch)
README.md
```

Total storage: ~57 GB.


## Backbone: Qwen2.5-VL-3B (modified)

From `Lance_3B/llm_config.json`:

| field | value |
|---|---|
| `architectures` | `Qwen2_5_VLForConditionalGeneration` |
| `model_type` | `qwen2_5_vl` |
| `hidden_size` | 2048 |
| `intermediate_size` | 11008 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 16 (head_dim = 128) |
| `num_key_value_heads` | 2 (GQA) |
| `max_position_embeddings` | 128000 |
| `rope_theta` | 1e6 |
| `rope_scaling.type` | `mrope`, sections `[16, 24, 24]` |
| `tie_word_embeddings` | true (but stored untied; both `embed_tokens` and `lm_head` materialised in F32) |
| `vocab_size` | 151936 |

Vision-related special tokens (Qwen 2.5-VL standard):
- `<|vision_start|>`: 151652
- `<|vision_end|>`: 151653
- `<|vision_pad|>`: 151654
- `<|image_pad|>`: 151655   (placeholder for ViT or VAE image tokens)
- `<|video_pad|>`: 151656


## Per-layer weight keys (the MoT split)

Every one of the 36 layers carries **both** an understanding expert (no suffix)
and a generation expert (suffix `_moe_gen`). The two experts share the
attention computation but have their own Q/K/V/O projections, q/k norms,
input/post-attention norms, and MLP. Final `model.norm` is also duplicated.

```
language_model.model.layers.<L>.input_layernorm.weight                 [2048]
language_model.model.layers.<L>.input_layernorm_moe_gen.weight         [2048]

language_model.model.layers.<L>.self_attn.q_proj.{weight,bias}         [2048, 2048] / [2048]
language_model.model.layers.<L>.self_attn.k_proj.{weight,bias}         [ 256, 2048] / [ 256]
language_model.model.layers.<L>.self_attn.v_proj.{weight,bias}         [ 256, 2048] / [ 256]
language_model.model.layers.<L>.self_attn.o_proj.weight                [2048, 2048]
language_model.model.layers.<L>.self_attn.q_norm.weight                [ 128]   # per-head RMSNorm
language_model.model.layers.<L>.self_attn.k_norm.weight                [ 128]
language_model.model.layers.<L>.self_attn.{q,k,v,o}_proj_moe_gen.*
language_model.model.layers.<L>.self_attn.{q,k}_norm_moe_gen.weight

language_model.model.layers.<L>.post_attention_layernorm.weight        [2048]
language_model.model.layers.<L>.post_attention_layernorm_moe_gen.weight [2048]

language_model.model.layers.<L>.mlp.{gate,up,down}_proj.weight         (SwiGLU, no bias)
language_model.model.layers.<L>.mlp_moe_gen.{gate,up,down}_proj.weight

language_model.model.embed_tokens.weight                               [151936, 2048]
language_model.model.norm.weight                                       [2048]
language_model.model.norm_moe_gen.weight                               [2048]
language_model.lm_head.weight                                          [151936, 2048]
```

Storage dtype is **F32** for the LLM (not bf16, despite the config flag).
That is why each LM safetensors file is ~24 GB on disk for a 6B-param model.

The naming convention exactly matches BAGEL's `modeling/bagel/qwen2_navit.py`
(see `q_proj_moe_gen`, `mlp_moe_gen`, `norm_moe_gen`, etc.). BAGEL's
packed-sequence forward with modality-indexed routing is the closest existing
reference implementation.


## Generation pathway (added on top of the LLM)

```
vae2llm.weight        [2048, 48]      # VAE latent ch -> LLM hidden
vae2llm.bias          [2048]
latent_pos_embed.pos_embed                                # the "MaPE" bank
    Lance_3B:         [  4096, 2048]   # up to 4096 spatial positions (64x64)
    Lance_3B_Video:   [126976, 2048]   # up to 31 frames x 64 x 64 = 31 * 4096
time_embedder.mlp.0.{weight,bias}    [2048, 256] / [2048]   # sinusoidal-time MLP
time_embedder.mlp.2.{weight,bias}    [2048, 2048] / [2048]
llm2vae.weight        [48, 2048]      # LLM hidden -> velocity in VAE latent space
llm2vae.bias          [48]
```

Interpretation:
- Generation is **flow-matching / diffusion** in the latent space of the
  Wan 2.2 VAE (48 channels).
- A noisy latent `x_t ∈ R^{B, 48, T_lat, H_lat, W_lat}` is flattened spatially+temporally,
  projected by `vae2llm` to LLM hidden 2048, plus the sliced `latent_pos_embed`
  for the current 3D shape, plus a broadcast `time_embedder(t)` signal.
- Those latent tokens enter the shared LLM sequence (with text/ViT
  conditioning tokens) and are processed using the **generation expert**.
- The final hidden states at latent positions are projected by `llm2vae` to
  predict the velocity (flow-matching target), and a flow-matching ODE step
  produces `x_{t-Δ}`.
- Iterate until `t=0`, then decode with the Wan 2.2 VAE → pixels / frames.

The **MaPE** = "Multimodal-aware Positional Embedding" mentioned in the paper.
A single learned table indexed by the latent token's 3D position, added in
LLM hidden space — separate from the Qwen mrope used for text & ViT tokens.
This is how Lance reduces "positional interference among heterogeneous visual
tokens" (their words).


## Understanding pathway (ViT side)

For Lance_3B, the ViT is **not** bundled. You load
`Qwen2.5-VL-ViT/vit.safetensors` (668M, BF16) separately. Shapes confirm
it is the stock Qwen 2.5-VL ViT:

```
patch_embed.proj.weight     [1280, 3, 2, 14, 14]   # temporal_patch=2, patch=14
blocks.<i>.norm{1,2}, attn.{qkv, proj}, mlp.{gate,up,down}_proj   i=0..31
merger.ln_q.weight          [1280]
merger.mlp.0.{w,b}          [5120, 5120] / [5120]   # 4*1280 -> 5120
merger.mlp.2.{w,b}          [2048, 5120] / [2048]   # -> LLM hidden
```

For Lance_3B_Video the same ViT lives inside the checkpoint under
`vit_model.*`, presumably finetuned on video frames.

ViT-produced semantic tokens are inserted at `<|image_pad|>` / `<|video_pad|>`
positions (Qwen 2.5-VL convention) and processed by the **understanding
expert** of every LLM block.


## What's identical between Lance_3B and Lance_3B_Video

Empirically (`comm -23`/`comm -13` on the two key dumps):
- Every key in `Lance_3B` exists in `Lance_3B_Video` (same names, same shapes).
- The only deltas in the LM side are the row count of `latent_pos_embed`
  (4096 → 126976) — i.e. the MaPE bank is bigger to address temporal slots.
- `Lance_3B_Video` additionally bundles a `vit_model.*` subtree (390 extra
  tensors, identical schema to the standalone Qwen2.5-VL ViT).

So the video model is structurally a **superset** of the image model. The
single safetensors that's most worth loading first for testing is
`Lance_3B/model.safetensors` (smaller, simpler, no bundled ViT).


## Putting it together — inference recipes

### A. Multimodal understanding (VQA / captioning)

1. Tokenize prompt with the bundled Qwen tokenizer (chat template applies).
2. For each input image/video, encode with the Qwen2.5-VL ViT → semantic
   token sequence. Splice in at `<|image_pad|>` positions; build mrope
   position ids per Qwen 2.5-VL convention.
3. Run the LLM. Every token uses the **understanding expert** path
   (no `_moe_gen` weights touched).
4. Sample text autoregressively from `lm_head`.

This is essentially vanilla Qwen 2.5-VL inference, modulo loading only the
understanding-expert subset of the weights.

### B. Text-to-image / video generation (flow matching)

1. Tokenize prompt; mark generation region with image/video pad tokens.
2. Initialise `x_T = N(0, I)` in VAE latent space at the target 3D shape.
3. For each denoising step at time `t`:
   - flatten latents → `vae2llm` → add MaPE slice → add `time_embedder(t)`
   - prepend text tokens, run LLM with mixed expert routing
   - read latent positions out via `llm2vae` → velocity `v`
   - flow-matching ODE update: `x_{t-Δ} = x_t - Δ * v`
4. Decode `x_0` with Wan 2.2 VAE.

### C. Image editing / multi-turn editing

Concatenate (source image ViT tokens) + (instruction text) + (target image
latents). Same MoT layer stack; routing depends on the per-token modality.
This is the BAGEL editing recipe — almost certainly carried over verbatim.


## Open questions to confirm during implementation

1. **Time conditioning injection point.** BAGEL applies `time_embedder(t)`
   as an additive bias to the gen-expert layernorm input. Lance's
   `time_embedder.mlp` shape (256 → 2048 → 2048) is identical to BAGEL's,
   so the same injection point likely applies. **Confirmed** (and matches
   the official code at
   `refs/lance_official/modeling/lance/lance.py`): `vae_embed =
   vae2llm(x_t) + timestep_embed + latent_pos_embed`.
2. **MaPE indexing for 3D video latents.** **Confirmed**: BAGEL's
   `get_flattened_position_ids_extrapolate` (max_per_side=64) is byte-for-
   byte what the official `data/data_utils.py::get_flattened_position_ids
   _extrapolate_video` does: `id = t * 64**2 + h * 64 + w`. **Surprise**:
   the 126976-row bank in the checkpoint is **not learned** — it's a fixed
   3D sin-cos embedding initialized once and stored as a
   `requires_grad=False` parameter (see `PositionEmbedding3D` in
   `modeling/lance/modeling_utils.py`).
3. **CFG support.** **Confirmed**: CFG interval `(0.4, 1.0)` plus global
   renorm clamped to `[renorm_min, 1.0]` is what the official video script
   uses too.
4. **ViT image preprocessing.** The shipped Qwen2.5-VL-ViT config is
   stock; the standard `Qwen2_5_VLProcessor` from HF transformers works
   verbatim once `q_norm` / `k_norm` are patched in (see `qknorm_patch.py`).


## Post-mortem: things we got wrong (post-official-release)

ByteDance shipped [the official repo](https://github.com/bytedance/Lance)
two days after this work started. Diffing against
`refs/lance_official/`, here's what we missed:

### 1. Mrope, not 1D RoPE

Our scratch impl used Qwen2 1D RoPE with all latent tokens sharing the
first vision-pad position. Lance actually uses **Qwen 2.5-VL's mrope**:
`apply_multimodal_rotary_pos_emb` with `mrope_section=(16, 24, 24)` that
splits the head_dim across (t, h, w) axes. Position IDs are a `[3, 1, L]`
tensor produced by `language_model.get_rope_index`. For images, T_lat=1
collapses the time axis and our shared-position scheme works by accident.
For video this is **fatal** — without temporal mrope, frames are
indistinguishable in the rotation and the model either mode-collapses or
drifts.

### 2. Latent block lives at high RoPE positions

`shift_position_ids(..., pos_shift=1000, shift_attn_mode=["full_noise",
"full"])` pushes the latent block to position ~1000 so it doesn't
interfere with text. We had latents pinned at position 0.

### 3. "Generalized 3D causal attention" = causal text + bidir noise block

Not a custom 3D mask. The paper's phrase describes the *block structure*:
`create_sparse_mask` in `data/data_utils.py` builds
`(causal OR full_and_noise) AND remove_noise AND sample`, i.e.

- text tokens (`causal` mode) → standard causal LM
- VAE latent tokens (`noise` mode) → bidirectional within the block, can
  attend to all earlier text via the causal-OR clause
- ViT understanding tokens (`full` mode) → bidirectional within the ViT
  block
- `remove_noise_mask` prevents noise-block i from attending to noise-block
  j (relevant for batched editing tasks)

The whole thing is compiled into a `flex_attention` block mask, not a
materialized `[S, S]` bool tensor.

### 4. Default video resolution is 480p, 81 frames

Their `inference_lance.sh` ships defaults of `video_480p` / `832x480` /
`81 frames` / 30 steps / `cfg_text_scale=4.0` /
`validation_timestep_shift=3.5`. We were testing at 256² which was too
small for the trained distribution, and the model failed gracefully into
mode collapse.

### 5. `latent_patch_size` is `(1, 1, 1)` at inference

The `LanceConfig` default is `(1, 2, 2)` (each LLM token = a 1×2×2 chunk
of VAE latents) but every shipped inference script overrides it to
`(1, 1, 1)`. That's why the checkpoint has `vae2llm: Linear(48, 2048)` —
matches `1*1*1*48`. We got this right out of luck.

### Net effect

Items 1-3 explain the T2V failure entirely. Item 4 made the mode collapse
look worse. Item 5 was a bullet we dodged.

The official code in `refs/lance_official/` is the right way to actually
run Lance for image/video generation and editing; our scratch code stays
in this repo as a learning artifact + as an HF-compatible understanding
pathway, which is a strictly different (and simpler) thing.
