# lance-assistant

A chat-style multimodal assistant for **ByteDance Lance**, a 3B unified
multimodal model that does text-to-image, text-to-video, image / video
editing, image / video understanding, and text chat — all from a single
checkpoint. Originally a reverse-engineering project (May 16, 2026) because
ByteDance posted [the weights](https://huggingface.co/bytedance-research/Lance)
saying *"code coming soon"* and we got tired of waiting. **Two days later
they shipped [bytedance/Lance](https://github.com/bytedance/Lance)**, so this
repo wraps the official inference code in:

- a **FastAPI + Preact chat web UI** with streaming tool-call bubbles,
- an optional **agentic VLM orchestrator layer** (LM Studio / Ollama / vLLM /
  OpenAI) that gives Lance an actual conversational brain,
- **auto-downloading weights** on first run,
- a handful of **CLI shells** for direct one-shot inference,
- and an `ARCHITECTURE.md` post-mortem of what reverse-engineering got
  right vs wrong vs the official release.

## TL;DR — just run Lance

```bash
git clone https://github.com/SanDiegoDude/lance-assistant.git
cd lance-assistant

./scripts/setup.sh              # creates venv, installs deps, clones official Lance
cp .env.example .env            # optional: configure orchestrator (LM Studio etc.)
./scripts/run_webui.sh          # open http://localhost:7861
```

On first launch the server auto-downloads ~32 GB of Lance weights from
Hugging Face into `weights/Lance_hf/` (resumable). After that, model load
is fast and the UI is ready in ~30 s.

### How it's organized

**A. Chat-style web UI (recommended)** — all seven Lance tasks in one
ChatGPT-style chat, optionally **driven by an agentic VLM orchestrator**
running on your local network (LM Studio / Ollama / vLLM / OpenAI — any
OpenAI-compatible endpoint). The orchestrator handles conversation,
reasoning, code-fenced markdown, and visual understanding; Lance is its
tool for actually producing new pixels.

The orchestrator chats with the user and emits OpenAI-style function
calls when it needs new pixels. We then run Lance, hand the result back
to the orchestrator so it can *see* what was produced, and let it
respond with a confirmation. Tools exposed to the orchestrator:

| Tool                              | What it runs                          |
|---|---|
| `generate_image(prompt, aspect)`  | Lance t2i (~2 min)                   |
| `generate_video(prompt, ...)`     | Lance t2v (~3 min at 192p, ~26 min at 480p) |
| `edit_image(instruction)`         | Lance image_edit on the most recent image (~1 min) |
| `edit_video(instruction)`         | Lance video_edit on the most recent video (slow) |

Image and video **understanding** are handled by the orchestrator VLM
itself — it can already see whatever you attach, no Lance call needed.

If the orchestrator is unreachable (LM Studio not running, etc.) the
UI shows an "orchestrator: offline" pill and falls back to **Lance-native
dispatch**: the Output: pills (Auto / Text / Image / Video / Understand)
pick a task directly from your prompt and attachment.

**Recommended orchestrator models** (load one in LM Studio):

- `qwen2.5-vl-7b-instruct`   — excellent at function calling + vision, 5-7 GB Q4
- `pixtral-12b`              — strong vision, 7-9 GB Q4
- `mistral-small-3.1-24b`    — best reasoning if you have VRAM
- `gemma-3-12b-it`           — good vision + tool calls
- `llama-3.2-11b-vision`     — adequate but weaker at tool calls

The orchestrator MUST support OpenAI-style `tools` / `tool_calls` to
drive Lance. If it doesn't, you can still use Lance-native dispatch.

**B. CLI shell wrapper** — direct one-shot invocation of the official
`inference_lance.py`, identical to what ByteDance ships:

```bash
./scripts/run_official.sh t2i   # 768x768 image, ~2 min on a single H100/GB10
./scripts/run_official.sh t2v   # 832x480 / 81f / 5 sec, ~26 min
./scripts/run_official.sh image_edit
./scripts/run_official.sh video_edit
./scripts/run_official.sh x2t_image   # image understanding
./scripts/run_official.sh x2t_video   # video understanding

# Outputs land in refs/lance_official/results/<task>_sample_*/
# Edit refs/lance_official/config/examples/*.json to change the prompts.
```

The official Lance code vendors Qwen2 and Qwen2.5-VL forks locally so it
doesn't depend on HF internals and runs cleanly under either the official
pins (torch 2.5.1+cu124 / transformers 4.49 / flash-attn 2.6.3, ~Hopper
and older) or our newer Blackwell-friendly stack (torch 2.9+cu128 /
transformers 4.56 / flash-attn 2.8.3). The only patch needed was a soft import of
`decord` in `refs/lance_official/data/datasets_custom/validation_dataset.py`
(Python 3.12 has no `decord` wheel; T2I and T2V don't need it anyway — only
the video-editing / video-understanding tasks do). Heads-up: their pinned
torch+cu124 stack will *not* work on Blackwell (sm_121) — you need cu128+.

Sample outputs in `out/` (all generated on a single GB10 with our env):

| File | Task | Notes |
|---|---|---|
| `out/webui_t2i_corgi_768.png`            | T2I via web UI, 768x768          | "corgi in a sunlit meadow, golden hour" |
| `out/webui_image_edit_corgi_night.png`   | image_edit via web UI            | same corgi, "change to a moonlit night with stars" — note the identity is preserved across the edit |
| `out/webui_t2v_balloon_192p_13f.mp4`     | T2V via web UI, 192p / 13 frames | fast smoke-test render, ~41 sec |
| `out/official_t2i_girl_768.png`          | T2I, 768x768, 30 steps, CFG 4    | the prompt-0 girl-with-piano portrait from `t2i_example.json` |
| `out/official_t2i_cat_stop_sign.png`     | T2I, 768x768                     | rainbow STOP-sign cat (prompt 1) — text rendering works |
| `out/official_t2i_rainbow_fox.png`       | T2I, 768x768                     | anthropomorphic rainbow fox (prompt 2) |
| `out/official_t2v_piano_480p.mp4`        | T2V, 832x480 / 81f / 12 fps      | woman at grand piano, the prompt asks for "begins from a medium view and gradually moves into a close facial framing" and the model **delivers that camera move** |
| `out/official_t2v_unicorn_480p.mp4`      | T2V, 832x480 / 81f / 12 fps      | pastel unicorn in cloud valley |
| `out/scratch_t2i_mountain_512.png`       | T2I via our scratch impl         | for posterity — our hand-rolled flow_match.py |

Timings on a single GB10: T2I ≈ 2 min / image at 768², T2V ≈ 26 min / clip
at 832x480 / 81f / 30 steps. (Most of that is the LLM forward on the 33K
latent-token sequence.)

## Status

| Pathway                  | Our scratch impl | Official code |
|---|---|---|
| Text chat + image VQA    | ✅ works (`scripts/chat_text.py`, `scripts/chat_image.py`) | ✅ works (`--task x2t_image` / `x2t_video`) |
| Text-to-image (T2I)      | ✅ works at 512x512 (`scripts/t2i.py`)                   | ✅ works at 768x768 (`--task t2i`) |
| Text-to-video (T2V)      | ❌ mode collapse / drift                                  | ✅ works at 480p / 832x480 / 81f (`--task t2v`) |
| Image editing            | ❌ not implemented                                        | ✅ `--task image_edit` |
| Multi-turn editing       | ❌ not implemented                                        | ✅ (built into image_edit dataset) |
| Video editing            | ❌ not implemented                                        | ✅ `--task video_edit` |

The "scratch impl" stuff stays in this repo as a learning artifact and
because the understanding pathway + T2I work cleanly on top of plain HF
transformers (no `flex_attention` / `flash_attn_varlen_func` required).

## What we got right (and wrong) reverse-engineering

Reverse-engineering from safetensors keys before they published code, we
got most of it right but missed two things that matter a lot for video:

**Right ✅**
- Backbone is Qwen 2.5-VL 3B (36 layers, hidden 2048, GQA 16/2, head_dim
  128). Per-head QK-norm (Qwen3-style RMSNorm before RoPE) is present —
  HF's stock `Qwen2_5_VLAttention` doesn't have it, so loading Lance into
  vanilla HF silently drops 72 RMSNorm weights and outputs gibberish. We
  patched it in [`src/lance/qknorm_patch.py`](src/lance/qknorm_patch.py).
- Dual-expert MoT: each transformer layer has two parallel weight sets, one
  for understanding tokens (text + ViT features) and one with `_moe_gen`
  suffix for generation tokens (VAE latents). Attention is shared, MLP and
  layernorms are not.
- Wan 2.2 VAE (48 latent channels, 4× temporal / 16× spatial compression)
  for generation, Qwen 2.5-VL ViT (14² spatial × 2 temporal patch) for
  understanding.
- MaPE positional bank `[126976, 2048]` for video = `31 latent frames × 64²`
  (max grid). Indexing `id = t*64² + h*64 + w` matches BAGEL's
  `get_flattened_position_ids_extrapolate` (also confirmed in official
  `data/data_utils.py::get_flattened_position_ids_extrapolate_video`).
  Surprise: official code actually keeps this bank as a **3D sin-cos
  embedding initialized once and frozen** (`requires_grad=False`) — the
  values just happen to live in the checkpoint.
- `time_embedder.mlp: 256 → 2048 → 2048` is a stock DiT-style timestep
  embedder added to every latent token before the LLM.
- CFG with both interval `(0.4, 1.0)` and global renorm (clamp `[min, 1]`)
  to keep flow-matching trajectories stable at low t.

**Wrong / missed ❌**
- *RoPE is mrope, not 1D RoPE.* We used plain 1D RoPE with all latents
  sharing the first vision-pad position. Lance actually uses Qwen 2.5-VL's
  **multimodal rotary position embedding** with `mrope_section=(16,24,24)`
  splitting head_dim across (t, h, w) axes. For images it doesn't matter
  much (T_lat=1, our shared-position scheme accidentally produces sane
  output). For video, this is fatal — without proper temporal mrope
  coordinates the model has no way to differentiate frames in the
  rotation, so it either mode-collapses or drifts.
- *Latent block gets a position shift.* `pos_shift ≈ 1000` is applied to
  the latents so they sit at high RoPE positions away from text. We had
  them sitting at position 0 (the first vision-pad index).
- *"Generalized 3D causal attention"* (paper's phrase) is just **standard
  causal text + bidirectional within the latent noise block**, implemented
  via `data/data_utils.py::create_sparse_mask` (flex-attention). My
  elaborate hand-rolled `build_3d_causal_mask` was misreading the paper.
- *Default video resolution is 480p (832×480), not 256².* And 81 frames
  (5 sec @ 16 fps), not 49.

## What's in this repo

```
notes/ARCHITECTURE.md        post-mortem of the safetensors-key map and
                             what we got right / wrong
refs/lance_official/         git clone --depth 1 of bytedance/Lance
refs/wan22/                  Wan 2.2 reference (for the VAE)
refs/bagel/                  BAGEL reference (similar architecture)
.env / .env.example          orchestrator config (LM Studio host, key, model)
webui/
  server.py                  FastAPI server: singleton Lance pipeline,
                             conversation state, agentic + native runners,
                             /api/conversations endpoint with SSE streaming
  orchestrator.py            OpenAI-compatible client (httpx) + Lance tool
                             schemas + system prompt
  static/index.html          Preact + HTM + Tailwind + marked.js chat UI
                             (single file, no build step)
  tmp/                       uploads, request JSONs, generated media
                             (gitignored)
scripts/
  run_webui.sh               launch the chat UI (recommended)
  run_official.sh            wrapper for refs/lance_official inference_lance.sh
  fetch_configs.py           pull just the json/tokenizer files (~1 MB)
  inspect_safetensors_remote.py  HTTP-range reader for safetensors headers
  chat_text.py               text-only chat demo (HF Qwen2.5-VL + qknorm)
  chat_image.py              image VQA demo
  t2i.py                     our scratch T2I (works at 512²; for posterity)
  t2v.py                     our scratch T2V (broken; left as learning trail)
src/lance/
  download.py                hf_hub_download CLI for the weights
  verify.py                  structural checker
  extract_understanding.py   Lance→HF Qwen2.5-VL state-dict remap
  qknorm_patch.py            the QK-norm monkey-patch for HF transformers
  lance_model.py             scratch MoT implementation (incomplete; mrope
                             missing, see post-mortem)
  flow_match.py              scratch T2I/T2V loop (T2I works at 512²)
  wan_vae.py                 Wan 2.2 VAE encode/decode port
weights/Lance_hf/            downloaded weights (gitignored)
out/                         generated samples (gitignored)
```

### Web UI architecture

Single FastAPI process. Holds:

- The official Lance video model (`lance_3b_video`) loaded once and
  reused for all media tasks: t2i, t2v, image_edit, video_edit,
  x2t_image, x2t_video. (No reason to load both image and video
  checkpoints — the video one is a strict superset.)
- An `OrchestratorClient` that speaks the OpenAI chat-completions API
  with streaming + function calling. Talks to whatever VLM you're
  running on the local network.
- Our extracted HF understanding ckpt (`weights/lance_3b_understand`)
  — only used by the *legacy* native-dispatch text path; the
  orchestrator handles text chat natively when available.
- Per-conversation state: full chat history, including image data URLs
  the orchestrator embeds in its context, and the path to the most
  recent visible image/video so `edit_*` tools know what to target.

Frontend is a single HTML file using Preact + HTM + Tailwind via CDN +
marked.js + DOMPurify + highlight.js for rendering. No build step.

#### Request flow with orchestrator

```
   user types "draw me a butterfly"
       ↓
   POST /api/conversations/{id}/messages
       ↓
   Server: append user msg to conversation
       ↓
   OrchestratorClient.stream_chat(history, tools=[generate_image, ...])
       ↓
   Orchestrator decides: "I should call generate_image('a butterfly...')"
       ↓
   Server receives tool_call, runs Lance pipeline (~2 min)
       ↓
   Server emits SSE: tool_start → tool_result with media_url
       ↓
   Server adds tool message to conversation with image embedded as
     data URL (so the orchestrator can see what was produced)
       ↓
   OrchestratorClient.stream_chat(history with tool result)
       ↓
   Orchestrator responds: "Here's the butterfly you asked for!"
       ↓
   Server streams text_delta events
       ↓
   Frontend renders markdown bubble, image bubble with download/reuse
```

#### Request flow without orchestrator (fallback)

```
   user types "draw me a butterfly", picks "Image" mode pill
       ↓
   POST /api/conversations/{id}/messages
       ↓
   Server: decide_task(prompt, "image", None) → "t2i"
       ↓
   Run Lance pipeline directly
       ↓
   Emit tool_start / tool_result events
       ↓
   Done. No chat memory between turns.
```

#### Dispatch (native fallback)

```
                 │  no mode override   │  user picked mode
   nothing       │  text chat          │  text / t2i / t2v
   image attach  │  question? x2t      │  understand / image_edit
                 │  else  image_edit   │
   video attach  │  question? x2t      │  understand / video_edit
                 │  else  video_edit   │
```

#### Multi-turn editing

In agentic mode this is transparent — the orchestrator remembers the
previous generation and calls `edit_image(...)` with the right reference.
In native mode the "↻ Use as input for next message" button on each
generated bubble re-attaches the file for the next request.

#### Blackwell (sm_121a) workarounds in `webui/server.py`

Three patches the webui applies that the official code doesn't ship,
all driven by sm_121a being too new for the cu128 / triton 3 PTX
assembler we have:

1. `vit_config._attn_implementation = "sdpa"` — the default
   `flash_attention_2` path calls `flash_attn.layers.rotary` whose
   Triton kernel fails ptxas. SDPA uses cuDNN, works.
2. `qwen2_navit.flex_attention = <eager flex_attention>` — the official
   code does `flex_attention = torch.compile(flex_attention)` at module
   import; Inductor tries to compile the kernel for sm_121a and bails.
   Reverting to eager flex_attention is slower but actually runs.
3. `torch._dynamo.config.suppress_errors = True` — belt-and-braces
   fallback for any other `torch.compile` site that might trip on
   sm_121a; degrades to eager instead of failing.

Once the cu128 / triton stack catches up to sm_121a these can all go
away.

## Setup

For most users the `setup.sh` script at the top is all you need. It
creates `.venv/`, installs deps from `pyproject.toml`, clones the official
Lance code, and applies the `decord` soft-import patch.

```bash
./scripts/setup.sh
```

If you'd rather wire it into an existing env, set `SKIP_VENV=1` or
`PYTHON=/path/to/python`. The `pyproject.toml` pins runtime minimums but
**not** torch / CUDA — install those yourself if you need a specific build:

- **Ampere (sm_86) / Ada (sm_89) / Hopper (sm_90):** `torch>=2.5 cu124+` works,
  and `flash-attn>=2.6` will compile cleanly. The official pins
  (torch 2.5.1+cu124 / transformers 4.49 / flash-attn 2.6.3) are a safe
  starting point.
- **Blackwell (sm_120 / sm_121, e.g. GB10 / DGX Spark):** you need
  **torch 2.9 + cu128** and **flash-attn 2.8.3+**. The official cu124
  pins will fail to compile PTX for `sm_121`. Our `webui/server.py`
  monkey-patches `flex_attention` out of `torch.compile` and forces SDPA
  on the ViT to dodge the remaining Triton compile holes on Blackwell.

## Fetching weights

The web UI **auto-downloads** weights on first launch (`group=video`,
~32 GB). If you want to pre-fetch or grab a different subset:

```bash
# Configs + tokenizers only (~1 MB) — enough to inspect the architecture
python -m lance download --group small

# Image-only: Lance_3B + Qwen2.5-VL ViT + Wan2.2 VAE  (~28 GB)
python -m lance download --group image

# Video-only: Lance_3B_Video + Qwen2.5-VL ViT + Wan2.2 VAE  (~32 GB)
python -m lance download --group video

# Everything (~57 GB)
python -m lance download --group all
```

All resumable. Structurally verify (parses the safetensors header,
no torch needed):

```bash
python -m lance verify --target weights/Lance_hf
```

## Understanding pathway (our HF-compat extractor)

Pre-dating the official release, we built a vanilla
`Qwen2_5_VLForConditionalGeneration` extractor for Lance: (1) remap the
safetensors keys, (2) patch in the missing per-head QK-norms, and you've
got Lance running through stock HF Transformers for text + VQA. The
official code is now the recommended path for everything, but if you want
to play with the extracted standalone understanding head:

```bash
python -m lance extract_understanding \
  --lance weights/Lance_hf/Lance_3B/model.safetensors \
  --vit   weights/Lance_hf/Qwen2.5-VL-ViT/vit.safetensors \
  --config weights/Lance_hf/Lance_3B/llm_config.json \
  --tokenizer-dir weights/Lance_hf/Lance_3B \
  --target weights/lance_3b_understand --dtype bf16

python scripts/chat_text.py --prompt "What is the capital of France?"
python scripts/chat_image.py --image foo.jpg --prompt "What is shown?"
```

### GPU note (sm_121 / GB10)

Text inference works fine on GB10. Image VQA on GPU currently hits a
`torch 2.9 / cu128` NVRTC compile failure for a `prod` reduction kernel
because sm_121 isn't in cu128's known-arch table. CPU works (~2.7 tok/s).

## Roadmap

- [x] Map architecture from safetensors keys (pre-code-release)
- [x] Project scaffold + downloader + structural verifier
- [x] State-dict extractor (Lance → HF Qwen2.5-VL key remap)
- [x] QK-norm monkey-patch
- [x] **Milestone 1**: text chat + image VQA on the understanding pathway
- [x] Wan 2.2 VAE encode/decode port (scratch)
- [x] `LanceForGeneration` scratch MoT implementation (no mrope)
- [x] **Milestone 2**: scratch T2I at 512² (good quality)
- [~] Milestone 3: scratch T2V — **abandoned** (needs mrope + pos_shift +
      flex-attention sparse mask; the official code already has all of it)
- [x] **Milestone 4 (the real one)**: integrate official `bytedance/Lance`
      as the primary inference path with our weights layout
