# lance-assistant

A chat-style multimodal assistant built on **ByteDance Lance**, a 3 B
unified multimodal model that handles text-to-image, text-to-video,
image / video editing, image / video understanding, and text chat from
a single checkpoint.

This repo wraps the official ByteDance inference code in:

- a **FastAPI + Preact chat web UI** with streaming responses, async
  generation jobs, and live progress bubbles,
- an optional **agentic orchestrator layer** (LM Studio / Ollama / vLLM /
  OpenAI — any OpenAI-compatible endpoint) that gives Lance a
  conversational front-end with tool-calling,
- **hot-swapping** between the two Lance variants (image-focused
  `lance_3b` and video-focused `lance_3b_video`) so a single chat can
  produce both crisp-text images and video clips,
- **auto-captioning of uploads** and **automatic post-job reflections**
  so the agent always knows what it's looking at and reacts to what
  Lance just produced — without the user having to nudge it,
- **auto-downloading weights** on first run,
- and a handful of CLI shells for direct one-shot inference and weights
  management.

## Quick start

```bash
git clone https://github.com/SanDiegoDude/lance-assistant.git
cd lance-assistant

./scripts/setup.sh              # creates venv, installs deps, clones official Lance
cp .env.example .env            # optional: configure orchestrator (LM Studio etc.)
./scripts/run_webui.sh          # open http://localhost:7861
```

On first launch the server auto-downloads the Lance weights from
Hugging Face into `weights/Lance_hf/` (~30 GB per variant, resumable).
On a 24 GB card add `--lowvram` so the variants hot-swap instead of
both occupying VRAM (see "Hot-swapping" below).

## How it works

### Web UI

A single-page Preact app served by the FastAPI backend. All Lance tasks
(generation, editing, understanding) run through one ChatGPT-style chat
optionally driven by an **agentic orchestrator** (a VLM running on your
own network that handles conversation and decides when to call Lance).

**Generation tools the orchestrator can call**

| Tool                                          | What it runs                                            |
|---|---|
| `generate_image(prompt, aspect)`              | Lance t2i                                               |
| `generate_video(prompt, resolution, frames)`  | Lance t2v                                               |
| `edit_image(instruction, asset_id?)`          | Lance image_edit on `asset_id` or the most recent image |
| `edit_video(instruction, asset_id?)`          | Lance video_edit on the most recent video               |
| `list_jobs(status?)`                          | inspect background generations                          |
| `get_job(job_id)`                             | live progress + result of one job                       |
| `list_assets(kind?, limit?)`                  | every image/video tracked in this chat                  |
| `get_asset(asset_id)`                         | one asset's URL / source / details                      |
| `cancel_job(job_id)`                          | request cancellation (effective for queued)             |

Image and video **understanding** is handled by the orchestrator VLM
itself — it can already see whatever you attach.

If the orchestrator is unreachable (LM Studio not running, etc.) the
UI shows an "orchestrator: offline" pill and falls back to **Lance-native
dispatch**: the Output: pills (Auto / Text / Image / Video / Understand)
pick a task directly from your prompt + attachment.

### Async generation jobs

Generation runs **asynchronously** in the background. When the
orchestrator calls a tool it gets back `{job_id, status: "queued"}`
instantly and continues the conversation. The result lands in the chat
whenever Lance finishes; meanwhile the user can keep typing, ask
follow-up questions, even start another job in parallel.

Each conversation has a **persistent SSE event channel** at
`/api/conversations/{cid}/events`. All events (text deltas, tool calls,
job lifecycle, per-step progress, errors) flow through that single
channel with monotonic sequence numbers; reconnect with `?from_seq=N+1`
to replay anything missed.

While a job is running the server emits `job_progress` events on every
diffusion step. The UI renders a live progress bar with step count and
ETA; the orchestrator can call `get_job(job_id)` and read `progress`,
`elapsed_seconds`, and `eta_seconds` to give the user a dynamic,
hardware-accurate estimate (no hard-coded "wait 2 min" messages).

### Auto-captioning uploads + post-job reflection

Two pieces of glue make conversations with attached / generated media
feel coherent (only active when `ORCHESTRATOR_EMBED_TOOL_IMAGES=on`,
which signals a VLM orchestrator):

- **Upload captioning** — when you attach an image or video, the server
  fires a one-shot caption job against the orchestrator VLM. The
  caption is silently prepended to your message text as
  `[Auto-caption of the attached <kind> (asset_id=…): …]` so the agent
  has a textual ground truth and stops hallucinating about which
  upload you meant. Videos are sampled at 1 FPS, resized to 512px on
  the long edge, and sent as a frame stack. A small "show
  auto-caption" toggle under the thumbnail lets you see what the agent
  saw. None of this lives in chat history.

- **Post-job reflection** — after a Lance generation job finishes, the
  server queues a brief follow-up orchestrator turn that includes the
  produced asset. The agent responds in plain prose acknowledging what
  came out ("Here's the corgi at sunset — the rim lighting came out
  really well"), without you having to send another message. Reflection
  turns can't call tools, so they never spawn cascades of generations.

### Persistent conversations

Conversations are dumped to `webui/tmp/state/conversations/<conv_id>.json`
on a debounced 500 ms timer, so a restart of the server doesn't lose
chat history, asset registry, or finished job records. In-flight jobs
(queued or running when the server died) are rewritten as `failed` on
restore — the worker thread that owned them is gone and Lance is not
checkpoint-restartable mid-generation.

Generated media itself (under `webui/tmp/results/`) and uploads (under
`webui/tmp/uploads/`) are not deleted by the persistence layer, so any
asset URL referenced by a restored conversation keeps working.

Pass `--nopersist` (or set `LANCE_NOPERSIST=1`) to disable persistence
entirely: nothing is written, anything from previous runs is ignored,
and the conversation graveyard from previous sessions is left untouched
on disk for you to remove yourself if you want.

### Debug tracing

`./scripts/run_webui.sh --debug` (or `LANCE_DEBUG=1`) enables verbose
tracing through the orchestrator and agentic-turn path: every outgoing
request (model, message roles, first-user preview), every SSE chunk
back from the orchestrator, every tool-call name + arguments, every
job submission, plus the branch decisions the turn loop took (final
text vs. tool round, read-only vs. generation, etc.). Output is
written to stdout and mirrored to `webui/tmp/logs/debug.log`
(truncated on each start) so you can `tail -f` it from another
terminal. The `debug()` helper is a no-op when the flag is off, so
production runs pay nothing.

### Context budget

The orchestrator's context window is configured via
`ORCHESTRATOR_CONTEXT_TOKENS` (default 32 000, matching Qwen3-30B's
native window — bump for larger models). Before every request to the
orchestrator the server estimates the total token cost of
`system_prompt + situation_header + history`, reserves headroom for
the assistant's reply, and silently drops the oldest non-system /
non-recent messages if the request would overshoot the budget. The
UI shows a live `used / budget` badge in the header that turns yellow
above 60 %, orange above 85 %, red above 95 %; when a drop happens,
the UI appends a small italic system note ("dropped N older messages
to keep the conversation alive") so it isn't invisible.

Token counts are estimates: `tiktoken` (`cl100k_base`) is used when
available, with a chars/4 fallback. Images are budgeted at a flat
1200 tokens each, videos at ~8× that. These are conservative middle
ground numbers — the goal is "don't overshoot the orchestrator's
window", not "match its billed token count exactly".

### Hot-swapping image and video variants

ByteDance ships two fine-tunes of Lance:

| Variant | Checkpoint dir | Strengths | Weaknesses |
|---|---|---|---|
| `image` | `Lance_3B/`       | Crisp in-image text, sharper detail; used for every image benchmark on the project page (GenEVAL, DPG, GEdit) | No video generation/editing |
| `video` | `Lance_3B_Video/` | Supports `generate_video` / `edit_video` / `x2t_video`                                                         | Image text rendering is noticeably worse — words come out garbled/incoherent |

Pick the mode with `LANCE_MODEL_VARIANT={image,video,auto}` in your `.env`:

- `image` — server locks to image-only. `generate_video` / `edit_video`
  tool calls are rejected.
- `video` — server locks to video-only (the video checkpoint can also
  do image gen, with worse text fidelity).
- `auto` (default) — **both** variants available. The server starts on
  whichever checkpoint is already on disk (prefers image for fresh
  installs) and loads the other one lazily the first time a task needs
  it. Image gen/edit always use the image variant; video gen/edit
  always use the video variant.

In `auto` mode the server keeps both variants in VRAM by default — fine
on big-VRAM hardware (40 GB+) but it won't fit on a 24 GB card. Pass
`--lowvram` (or set `LANCE_LOWVRAM=1`) and the server keeps just one
variant on the GPU at a time, parking the inactive one in system RAM
and hot-swapping when the task changes:

```bash
./scripts/run_webui.sh --lowvram
# or
LANCE_LOWVRAM=1 ./scripts/run_webui.sh
```

A swap moves ~6-8 GB between system RAM and VRAM, so on a 4090 with
DDR5 system RAM it adds ~2-3 s the first time you flip between image
and video tasks. After that the bundle stays cached in CPU RAM, so all
later swaps are the same ~2-3 s with no re-download or re-init from
disk.

You can pre-fetch both checkpoints up front (otherwise the second one
downloads lazily the first time it's needed):

```bash
python -m lance download --group image  # ~30 GB → weights/Lance_hf/Lance_3B/
python -m lance download --group video  # ~30 GB → weights/Lance_hf/Lance_3B_Video/
```

### Orchestrator

The agentic layer is optional but recommended. It's any
OpenAI-compatible chat-completions endpoint that supports the standard
`tools` / `tool_calls` function-calling protocol.

Configure in `.env`:

```bash
ORCHESTRATOR_BASE_URL=http://localhost:1234/v1
ORCHESTRATOR_API_KEY=                       # blank for LM Studio / Ollama
ORCHESTRATOR_MODEL=                         # blank = auto-pick first loaded
```

**Recommended local models** (load one in LM Studio):

- `qwen2.5-vl-7b-instruct` — excellent at function calling + vision, 5-7 GB Q4
- `pixtral-12b` — strong vision, 7-9 GB Q4
- `mistral-small-3.1-24b` — best reasoning if you have VRAM
- `gemma-3-12b-it` — good vision + tool calls
- `llama-3.2-11b-vision` — adequate but weaker at tool calls

If your orchestrator is itself a vision-language model, set
`ORCHESTRATOR_EMBED_TOOL_IMAGES=on` and the server will feed generated
images back to it as `image_url` blocks so it can reason about its own
output ("the butterfly came out a bit blurry — want me to redo it?").

If the orchestrator is unreachable, Lance-native dispatch kicks in as
the fallback. Output: pills pick the task directly.

### Direct CLI

For one-shot inference without the web UI:

```bash
./scripts/run_official.sh t2i           # 768x768 image
./scripts/run_official.sh t2v           # 832x480 / 81 frames / ~5 s clip
./scripts/run_official.sh image_edit
./scripts/run_official.sh video_edit
./scripts/run_official.sh x2t_image     # image understanding
./scripts/run_official.sh x2t_video     # video understanding

# Outputs land in refs/lance_official/results/<task>_sample_*/
# Edit refs/lance_official/config/examples/*.json to change the prompts.
```

## Setup

`./scripts/setup.sh` creates `.venv/`, installs the right
torch + flash-attn for your GPU arch, clones the official Lance code
into `refs/lance_official/`, and applies a soft-import patch for
`decord` (the upstream package is unmaintained and has no Python 3.12
wheel; the web server transparently shims out the small slice of its
API that Lance's video edit path uses, backed by PyAV).

The script auto-detects the GPU arch and installs:

| GPU                                 | Compute cap     | Stack `setup.sh` installs                       |
|---|---|---|
| 3090 / A100 / A40 (Ampere)          | sm_80 / sm_86   | torch 2.5.1+cu124 + flash-attn 2.7.4.post1      |
| 4090 / L40 / RTX 6000 Ada (Ada)     | sm_89           | torch 2.5.1+cu124 + flash-attn 2.7.4.post1      |
| H100 / H200 (Hopper)                | sm_90           | torch 2.5.1+cu124 + flash-attn 2.7.4.post1      |
| GB10 / DGX Spark / 5090 (Blackwell) | sm_120 / sm_121 | torch 2.9.0+cu128 (no flash-attn — uses SDPA)   |
| CPU-only                            | —               | torch 2.5.1+cpu                                 |

Override with env vars when needed:

```bash
GPU_DEPS=cu124 ./scripts/setup.sh      # force Ampere/Ada/Hopper stack
GPU_DEPS=cu121 ./scripts/setup.sh      # official Lance pin (torch 2.5.1+cu121 / flash-attn 2.6.3)
GPU_DEPS=cu128 ./scripts/setup.sh      # Blackwell
GPU_DEPS=cpu   ./scripts/setup.sh
GPU_DEPS=skip  ./scripts/setup.sh      # bring your own torch / flash-attn
FLASH_ATTN_SKIP=1 ./scripts/setup.sh   # install torch, skip flash-attn (SDPA-only)
```

If you'd rather wire it into an existing env, set `SKIP_VENV=1`. You
can also point `PYTHON=/path/to/python` to a specific interpreter when
creating the venv.

### Troubleshooting `flash-attn`

flash-attn is the #1 source of setup pain. `setup.sh` handles the
common case, but if it fails or you're installing manually:

1. **`ModuleNotFoundError: No module named 'torch'` during the build.**
   pip's default *build isolation* creates a clean env without torch,
   which flash-attn's `setup.py` imports. Always:
   ```bash
   pip install flash-attn==... --no-build-isolation
   ```

2. **`pip` starts compiling instead of downloading a wheel.**
   Compilation needs `nvcc` and takes 30+ min. Prebuilt wheels exist on
   the [Dao-AILab releases page](https://github.com/Dao-AILab/flash-attention/releases)
   for **specific** torch + cuda + python combinations. If yours
   doesn't match, pip falls back to source build. Fix by pinning torch
   to a version with wheel coverage:
   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   pip install flash-attn==2.7.4.post1 --no-build-isolation
   ```

3. **`packaging` / `ninja` not found.**
   flash-attn's build script imports them:
   ```bash
   pip install --upgrade pip wheel packaging ninja
   ```
   *before* installing flash-attn.

4. **Blackwell (sm_120 / sm_121) won't build flash-attn.**
   No prebuilt wheels yet and source builds frequently fail. The
   server falls back to SDPA + a monkey-patched `flex_attention` and
   runs fine without flash-attn — just install with
   `FLASH_ATTN_SKIP=1 ./scripts/setup.sh`.

5. **Already installed the wrong torch?** Reinstall:
   ```bash
   .venv/bin/pip uninstall -y torch torchvision torchaudio flash-attn
   .venv/bin/pip install torch==2.5.1 torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu124
   .venv/bin/pip install flash-attn==2.7.4.post1 --no-build-isolation
   ```

### VRAM and precision

By default the server loads Lance in **bf16**: roughly 8 GB per variant
(LLM ~6 GB + ViT ~1.4 GB + VAE ~0.6 GB). The official ByteDance
inference path moves the model to GPU in fp32 first and casts only
afterwards, which needs ~16 GB just for the move and OOMs a 24 GB
card. The webui collapses that into a single bf16 move.

If you have an 80 GB card and want maximum precision:

```bash
export LANCE_DTYPE=float32   # or float16 / bfloat16 (default)
```

`./scripts/run_webui.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
for you (if you haven't picked your own value), which removes a class
of fragmentation-driven OOMs on long-running sessions.

The biggest peak-memory savings in the codebase aren't from precision;
they're from the patches in `webui/server.py:_apply_one_time_patches`:

- The Qwen2.5-VL ViT's SDPA forward upstream builds a dense
  `[1, seq_length, seq_length]` bool attention mask. At video-edit
  sequence lengths (~65k tokens) the mask alone is 4 GB and the math
  kernel SDPA falls back to materializes the same-shape attention
  scores. We replace that with a per-segment SDPA loop that reads
  `cu_seqlens` directly, never materializes the mask, and drops peak
  ViT memory from O(S^2) to O(sum S_i^2).
- `torch.cuda.empty_cache()` runs before every job to recycle any
  fragmented allocator blocks the previous job left behind.

If you're still tight on memory on a 24 GB card, the things that move
the needle next:

- **Drop the edit resolution.** Both `edit_image` and `edit_video`
  accept a `resolution` argument the orchestrator can pass through:
  `edit_image` is `"512"` or `"768"` (default 768), `edit_video` is
  `"192p"`, `"360p"`, or `"480p"` (default 360p). The ViT sequence
  length scales with the square of resolution, so the difference
  between 192p and 480p is ~9x in peak attention memory. On 24 GB
  cards, 480p video-edits are likely to OOM even with the per-segment
  SDPA patch — 360p is the default for that reason.
- **`--lowvram`** keeps only one of the image / video variants resident
  at a time (saves ~8 GB at the cost of a hot-swap reload when the
  task type switches).
- **Shorter clips for `edit_video`.** The job runner already trims
  clips to 6 s before they hit Lance's frame sampler.
- **Offline-quantized weights** (not on-the-fly). The right path here
  is to quantize the LLM (and possibly the ViT) on a high-VRAM box
  once — `bitsandbytes` 8-bit / 4-bit, or fp8 via TransformerEngine
  on Hopper — save the quantized checkpoint, and reuse it. On-the-fly
  quantization at load time means we still need the full fp16/bf16
  copy in RAM for the cast, which gives back most of the savings. The
  diffusion DiT and the WAN VAE are more sensitive to quantization
  noise than the LLM and would need a careful quality eval before
  shipping a quantized variant; the LLM is the obvious first target
  since it's the single largest chunk (~6 GB bf16 → ~3 GB int8).

### Blackwell (sm_121a) compatibility patches

The webui applies three runtime patches that the official ByteDance
code doesn't ship, all required because the cu128 / Triton stack
doesn't yet emit valid PTX for sm_121a:

1. `vit_config._attn_implementation = "sdpa"` — the default
   `flash_attention_2` path inside Qwen2.5-VL ViT calls
   `flash_attn.layers.rotary` whose Triton kernel fails `ptxas`. SDPA
   uses cuDNN, works fine.
2. `qwen2_navit.flex_attention = <eager flex_attention>` — the official
   code wraps `flex_attention` in `torch.compile` at module import;
   Inductor tries to compile for sm_121a and bails. Reverting to the
   eager version is slower but actually runs.
3. `torch._dynamo.config.suppress_errors = True` — belt-and-braces
   fallback for any other `torch.compile` site that might trip on
   sm_121a; degrades to eager instead of failing.

Once cu128 / Triton catch up to sm_121a these will all go away.

## Fetching weights

The web UI auto-downloads weights on first launch. To pre-fetch or
grab a subset manually:

```bash
# Configs + tokenizers only (~1 MB) — enough to inspect the architecture
python -m lance download --group small

# Image-only: Lance_3B + Qwen2.5-VL ViT + Wan2.2 VAE  (~30 GB)
python -m lance download --group image

# Video-only: Lance_3B_Video + Qwen2.5-VL ViT + Wan2.2 VAE  (~32 GB)
python -m lance download --group video

# Everything (~57 GB)
python -m lance download --group all
```

All downloads resume. Verify on-disk integrity (parses the safetensors
header, no torch required):

```bash
python -m lance verify --target weights/Lance_hf
```

## Standalone understanding pathway

For text + VQA without spinning up the full multimodal pipeline, the
repo ships a key remapper that produces a vanilla
`Qwen2_5_VLForConditionalGeneration` checkpoint from Lance's
safetensors. Useful for hooking Lance into anything that expects a
stock HF Transformers model.

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

The full webui already does understanding through the orchestrator
VLM or `x2t_image` / `x2t_video` tasks — this extractor is only useful
if you want a standalone HF checkpoint.

## Repository layout

```
refs/lance_official/       git clone of bytedance/Lance (the model code)
.env / .env.example        configuration: orchestrator endpoint, model variant,
                           low-vram flag, dtype, port, etc.
webui/
  server.py                FastAPI server: variant-aware Lance pipeline,
                           async job runner, SSE event bus, conversation state
  orchestrator.py          OpenAI-compatible client + Lance tool schemas
                           + system prompt
  jobs.py                  JobRecord / EventBus / JobRunner
  static/index.html        Preact + HTM + Tailwind chat UI (single file)
  tmp/                     uploads, request JSONs, generated media (gitignored)
scripts/
  run_webui.sh             launch the chat UI (recommended)
  run_official.sh          wrapper for direct one-shot CLI inference
  setup.sh                 GPU-arch-aware installer
src/lance/
  download.py              hf_hub_download CLI for the weights
  verify.py                structural checker (safetensors header parser)
  extract_understanding.py Lance → HF Qwen2.5-VL state-dict remap
weights/Lance_hf/          downloaded weights (gitignored)
```

## License

The Lance model itself is released by ByteDance Research under their
own terms — see
[bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)
and [bytedance/Lance](https://github.com/bytedance/Lance). This wrapper
is provided as-is, no warranty.
