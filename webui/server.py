"""Lance multimodal chat backend.

FastAPI server that holds the official Lance pipeline in memory (for the six
multimodal tasks) plus our extracted HF understanding checkpoint (for pure
text chat), and dispatches user requests to the right task with a single
chat-style API.

Architecture:

  - Lance jobs run on a single FIFO worker thread (see webui.jobs.JobRunner).
    The orchestrator-driven path submits a job and gets `{job_id, status:
    queued}` back instantly; the worker runs the job and pushes
    job_started / job_completed events into the conversation's EventBus.
  - Each conversation has a persistent SSE event channel; the frontend
    keeps an EventSource open for its lifetime. New events fan out to all
    subscribers; reconnects replay from a cursor so nothing is missed.
  - POST /messages is non-blocking: it kicks off an orchestrator-turn task
    and returns 202; all output flows through the persistent SSE.

Endpoints:

  GET  /                                    serves webui/static/index.html
  GET  /static/<path>                       static files
  GET  /api/health                          health + orchestrator status
  POST /api/orchestrator/probe              re-probe orchestrator
  POST /api/conversations                   create a new conversation
  GET  /api/conversations/{cid}             fetch messages + assets + jobs
  POST /api/conversations/{cid}/messages    enqueue a user message (202)
  GET  /api/conversations/{cid}/events      persistent SSE event channel
  GET  /api/conversations/{cid}/jobs        list jobs in conversation
  POST /api/conversations/{cid}/jobs/{jid}/cancel   request cancellation
  GET  /api/conversations/{cid}/assets      list assets in conversation
  GET  /api/media/{job_id}/{name}           serve generated media
  GET  /api/uploads/{name}                  serve user-uploaded files

Task dispatch (Lance-native fallback when no orchestrator):

  attachment │ mode         │ task
  ───────────┼──────────────┼─────────────
  none       │ image        │ t2i
  none       │ video        │ t2v
  none       │ text / auto  │ text  (HF understanding ckpt)
  image      │ understand   │ x2t_image
  image      │ image        │ image_edit
  image      │ auto         │ x2t_image if prompt looks question-like else image_edit
  video      │ understand   │ x2t_video
  video      │ video        │ video_edit
  video      │ auto         │ x2t_video if prompt looks question-like else video_edit
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

# Make the official lance code importable BEFORE we touch torch.
ROOT = Path(__file__).resolve().parent.parent
LANCE_OFFICIAL = ROOT / "refs" / "lance_official"
sys.path.insert(0, str(LANCE_OFFICIAL))
sys.path.insert(0, str(ROOT / "src"))

# The official lance code uses relative paths like "downloads/Wan2.2_VAE.pth"
# and "config/path_default.yaml" — these only resolve when cwd is
# refs/lance_official/. We chdir there at import time; all our webui paths
# are absolute via WEBUI_DIR so this is safe.
os.chdir(LANCE_OFFICIAL)

# Working directories for this server
WEBUI_DIR = Path(__file__).resolve().parent
TMP_DIR = WEBUI_DIR / "tmp"
UPLOADS_DIR = TMP_DIR / "uploads"
RESULTS_DIR = TMP_DIR / "results"
JSONS_DIR = TMP_DIR / "request_jsons"
for d in (TMP_DIR, UPLOADS_DIR, RESULTS_DIR, JSONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Environment knobs the official sample_env.sh sets — we mirror them here so
# the official code is happy without sourcing the shell file.
os.environ.setdefault("EXP_HW_20250819", "False")
os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

import torch  # noqa: E402

# Blackwell (sm_121a) workaround. The official code wraps flex_attention in
# torch.compile (modeling/lance/qwen2_navit.py:47), which goes through
# Inductor → Triton → ptxas, and ptxas in our cu128/triton refuses sm_121a.
# Telling dynamo to suppress errors makes compile failures fall back to eager;
# we ALSO monkey-patch the wrapped flex_attention back to its uncompiled form
# in the LancePipeline init (slower, but actually runs on Blackwell).
import torch._dynamo  # noqa: E402
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 1024

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import (  # noqa: E402
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[lance-webui {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    task: str                                  # t2i | t2v | image_edit | ... | text
    prompt: str
    attachment_path: Optional[str]
    attachment_kind: Optional[str]             # "image" | "video" | None
    params: Dict[str, Any]
    status: str = "queued"                     # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0                      # 0..1
    progress_note: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    events: "asyncio.Queue[Dict[str, Any]]" = field(default=None)  # type: ignore[assignment]

    def public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "prompt": self.prompt,
            "status": self.status,
            "progress": self.progress,
            "progress_note": self.progress_note,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": (
                (self.finished_at or time.time()) - (self.started_at or self.created_at)
            ),
        }


# ---------------------------------------------------------------------------
# Lance pipeline (multimodal: t2i/t2v/edit/understand)
# ---------------------------------------------------------------------------

class LancePipeline:
    """Wraps the official Lance model and dispatches across the 6 multimodal
    tasks. Loads weights once, then `run(...)` is reusable for any task.

    Heavily adapted from refs/lance_official/lance_gradio_t2v_v2t.py.
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.lock = threading.RLock()
        self.initialized = False
        self.model = None
        self.vae_model = None
        self.vae_config = None
        self.tokenizer = None
        self.new_token_ids = None
        self.image_token_id = None
        self.base_model_args = None
        self.base_data_args = None
        self.base_inference_args = None

    def initialize(self) -> None:
        with self.lock:
            if self.initialized:
                return
            t0 = time.perf_counter()
            log(f"loading Lance model onto GPU {self.device}...")
            self._ensure_weights_and_symlinks()
            self._initialize_inner()
            log(f"Lance model ready in {time.perf_counter() - t0:.1f}s")
            self.initialized = True

    def _ensure_weights_and_symlinks(self) -> None:
        """Make sure Lance weights are present and the official code's
        downloads/ symlinks are set up. Downloads from HF on first run.
        """
        weights_root = ROOT / "weights" / "Lance_hf"
        required_dirs = [
            weights_root / "Lance_3B_Video",
            weights_root / "Qwen2.5-VL-ViT",
        ]
        required_files = [
            weights_root / "Lance_3B_Video" / "model.safetensors",
            weights_root / "Qwen2.5-VL-ViT" / "vit.safetensors",
            weights_root / "Wan2.2_VAE.pth",
        ]
        need_download = not all(p.exists() for p in required_files)
        if need_download:
            log("Lance weights missing — auto-downloading from Hugging Face")
            log(f"  target: {weights_root}")
            log(f"  size:   ~32 GB (video model + ViT + VAE), resumable")
            from lance.download import GROUPS, download_files
            try:
                download_files(weights_root, GROUPS["video"], dry_run=False)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Auto-download failed: {e}\n"
                    f"You can run it manually with:\n"
                    f"  python -m lance download --group video --target {weights_root}"
                )
            log("auto-download complete")
        # Wire up the symlinks the official code expects under refs/lance_official/downloads/
        official_downloads = LANCE_OFFICIAL / "downloads"
        official_downloads.mkdir(parents=True, exist_ok=True)
        for src_rel, dst_name in [
            ("Lance_3B",          "lance_3b"),
            ("Lance_3B_Video",    "lance_3b_video"),
            ("Qwen2.5-VL-ViT",    "Qwen2.5-VL-ViT"),
            ("Wan2.2_VAE.pth",    "Wan2.2_VAE.pth"),
        ]:
            src = weights_root / src_rel
            dst = official_downloads / dst_name
            if not src.exists():
                # lance_3b is image-only — not needed for the video pipeline
                if src_rel == "Lance_3B":
                    continue
                raise RuntimeError(f"required weight missing after download: {src}")
            try:
                if dst.is_symlink() or dst.exists():
                    if not dst.is_symlink() or os.readlink(dst) != str(src.resolve()):
                        dst.unlink(missing_ok=True) if hasattr(dst, "unlink") else None
                        dst.symlink_to(src.resolve())
                else:
                    dst.symlink_to(src.resolve())
            except FileExistsError:
                pass

    def _initialize_inner(self) -> None:
        # Undo the torch.compile wrap of flex_attention from
        # modeling/lance/qwen2_navit.py:47. Without this, the x2t text-gen
        # path triggers Inductor → Triton → ptxas (sm_121a fails). The eager
        # flex_attention is slower but works.
        from torch.nn.attention.flex_attention import flex_attention as _eager_flex_attention
        from modeling.lance import qwen2_navit as _qwen2_navit
        _qwen2_navit.flex_attention = _eager_flex_attention

        from safetensors.torch import load_file
        from transformers import set_seed
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLVisionConfig,
        )
        from common.utils.misc import AutoEncoderParams  # noqa: F401
        from config.config_factory import (
            DataArguments,
            InferenceArguments,
            ModelArguments,
        )
        from data.data_utils import add_special_tokens
        from inference_lance import (
            apply_inference_defaults,
            clean_memory,
            init_from_model_path_if_needed,
        )
        from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.qwen2.modeling_qwen2 import Qwen2Config
        from modeling.vae.wan.model import WanVideoVAE
        from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
        from copy import deepcopy as _deepcopy

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; webui needs a GPU")
        if self.device >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {self.device} unavailable (have {torch.cuda.device_count()})"
            )
        torch.cuda.set_device(self.device)

        # The video checkpoint subsumes the image checkpoint at inference
        # (max_num_latent_frames=31 and T_lat=1 collapses to image gen).
        # Using it for everything keeps a single model in memory.
        model_path = str(LANCE_OFFICIAL / "downloads" / "lance_3b_video")
        vit_path = str(LANCE_OFFICIAL / "downloads" / "Qwen2.5-VL-ViT")

        model_args = ModelArguments(
            model_path=model_path,
            vit_path=vit_path,
            vit_type="qwen_2_5_vl_original",
            llm_qk_norm=True,
            llm_qk_norm_und=True,
            llm_qk_norm_gen=True,
            tie_word_embeddings=False,
            max_num_frames=121,
            max_latent_size=64,
            latent_patch_size=[1, 1, 1],
        )
        data_args = DataArguments()
        inference_args = InferenceArguments(
            validation_num_timesteps=30,
            validation_timestep_shift=3.5,
            copy_init_moe=True,
            visual_und=True,
            visual_gen=True,
            vae_model_type="wan",
            apply_qwen_2_5_vl_pos_emb=True,
            apply_chat_template=False,
            cfg_type=0,
            validation_data_seed=42,
            video_height=480,
            video_width=832,
            num_frames=50,
            task="t2v",
            save_path_gen=str(RESULTS_DIR),
            resolution="video_480p",
            text_template=True,
            use_KVcache=True,
        )
        apply_inference_defaults(model_args, data_args, inference_args)
        inference_args.validation_noise_seed = inference_args.validation_data_seed

        set_seed(inference_args.global_seed)

        log(f"  loading LLM config: {model_path}/llm_config.json")
        llm_config = Qwen2Config.from_json_file(str(Path(model_path) / "llm_config.json"))
        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.qk_norm_und = model_args.llm_qk_norm_und
        llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = inference_args.freeze_und
        llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

        log("  init LLM (Qwen2ForCausalLM)")
        language_model = Qwen2ForCausalLM(llm_config)

        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(vit_path)
        # The default `flash_attention_2` path inside Qwen2.5-VL ViT calls
        # flash_attn's apply_rotary_emb which is implemented with a Triton
        # kernel; on Blackwell (sm_121a) ptxas refuses that arch. SDPA uses
        # cuDNN / native CUDA, no Triton, works fine.
        vit_config._attn_implementation = "sdpa"
        log("  init ViT (Qwen2.5-VL, sdpa attention)")
        vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
        vit_weights = load_file(str(Path(vit_path) / "vit.safetensors"))
        vit_model.load_state_dict(vit_weights, strict=True)
        clean_memory(vit_weights)

        log("  init VAE (Wan 2.2)")
        vae_model = WanVideoVAE()
        vae_config = _deepcopy(vae_model.vae_config)

        config = LanceConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=model_args.latent_patch_size,
            max_num_frames=model_args.max_num_frames,
            max_latent_size=model_args.max_latent_size,
            vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            connector_act=model_args.connector_act,
            interpolate_pos=model_args.interpolate_pos,
            timestep_shift=inference_args.timestep_shift,
        )
        model = Lance(
            language_model=language_model,
            vit_model=vit_model,
            vit_type=model_args.vit_type,
            config=config,
            training_args=inference_args,
        )

        log(f"  move to GPU {self.device}")
        model = model.to(self.device)

        log("  load tokenizer + special tokens")
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

        if inference_args.copy_init_moe:
            language_model.init_moe()

        log(f"  load fine-tuned ckpt: {model_path}")
        init_from_model_path_if_needed(model, model_args)

        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

        image_token_id = language_model.config.video_token_id
        new_token_ids.update({"image_token_id": image_token_id})
        model.update_tokenizer(tokenizer=tokenizer)

        if model_args.tie_word_embeddings:
            model.language_model.untie_lm_head()
            model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)

        model = model.to(device=self.device, dtype=torch.bfloat16)
        model.eval()
        if hasattr(vae_model, "eval"):
            vae_model.eval()

        self.model = model
        self.vae_model = vae_model
        self.vae_config = vae_config
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.image_token_id = image_token_id
        self.base_model_args = model_args
        self.base_data_args = data_args
        self.base_inference_args = inference_args

    # ---------------------------------------------------------------- run --

    def run(self, job: Job) -> Dict[str, Any]:
        """Dispatch a job to the right task. Blocks until done (or raises).

        Returns the public result dict that we store on `job.result`.
        """
        self.initialize()
        with self.lock:
            torch.cuda.set_device(self.device)
            if job.task in ("t2i", "t2v"):
                return self._run_generation(job)
            if job.task in ("image_edit", "video_edit"):
                return self._run_edit(job)
            if job.task in ("x2t_image", "x2t_video"):
                return self._run_understanding(job)
            raise ValueError(f"unknown task for Lance pipeline: {job.task}")

    # ---- common ---------------------------------------------------------

    def _make_request_args(
        self,
        task: str,
        prompt_file: Path,
        save_dir: Path,
        params: Dict[str, Any],
    ):
        request_model_args = deepcopy(self.base_model_args)
        request_model_args.cfg_text_scale = float(params.get("cfg_text_scale", 4.0))

        request_data_args = deepcopy(self.base_data_args)
        request_data_args.val_dataset_config_file = str(prompt_file)

        request_inference_args = deepcopy(self.base_inference_args)
        request_inference_args.validation_num_timesteps = int(params.get("steps", 30))
        request_inference_args.validation_timestep_shift = float(params.get("timestep_shift", 3.5))
        seed = int(params.get("seed", 42))
        request_inference_args.validation_data_seed = seed
        request_inference_args.validation_noise_seed = seed
        request_inference_args.video_height = int(params.get("height", 480))
        request_inference_args.video_width = int(params.get("width", 832))
        request_inference_args.num_frames = int(params.get("num_frames", 1))
        request_inference_args.resolution = params.get("resolution", "video_480p")
        request_inference_args.save_path_gen = str(save_dir)
        request_inference_args.task = task
        request_inference_args.text_template = True
        request_inference_args.prompt_data_dict = {}
        return request_model_args, request_data_args, request_inference_args

    def _build_batch(self, prompt_file, request_model_args, request_data_args, request_inference_args):
        from common.utils.misc import tuple_mul
        from data.data_utils import add_special_tokens  # noqa: F401
        from data.dataset_base import DataConfig, simple_custom_collate
        from data.datasets_custom import ValidationDataset

        dataset_config = DataConfig.from_yaml(str(prompt_file))
        if request_inference_args.visual_und:
            dataset_config.vit_patch_size = request_model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = request_model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = request_model_args.vit_max_num_patch_per_side
        if request_inference_args.visual_gen:
            vae_downsample = tuple_mul(
                tuple(request_model_args.latent_patch_size),
                (self.vae_config.downsample_temporal, self.vae_config.downsample_spatial, self.vae_config.downsample_spatial),
            )
            dataset_config.latent_patch_size = request_model_args.latent_patch_size
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = request_model_args.max_latent_size
            dataset_config.max_num_frames = request_model_args.max_num_frames

        dataset_config.text_cond_dropout_prob = request_model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = request_model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = request_model_args.vit_cond_dropout_prob

        dataset_config.num_frames = request_inference_args.num_frames
        dataset_config.H = request_inference_args.video_height
        dataset_config.W = request_inference_args.video_width
        dataset_config.task = request_inference_args.task
        dataset_config.resolution = request_inference_args.resolution
        dataset_config.text_template = request_inference_args.text_template

        val_dataset = ValidationDataset(
            jsonl_path=str(prompt_file),
            tokenizer=self.tokenizer,
            data_args=request_data_args,
            model_args=request_model_args,
            training_args=request_inference_args,
            new_token_ids=self.new_token_ids,
            dataset_config=dataset_config,
            local_rank=0,
            world_size=1,
        )
        return simple_custom_collate([val_dataset[0]])

    def _validate(self, val_data, request_model_args, request_inference_args, save_dir):
        from inference_lance import validate_on_fixed_batch
        from inference_lance import clean_memory, save_prompt_results

        validate_on_fixed_batch(
            fsdp_model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            val_data_cpu=val_data,
            training_args=request_inference_args,
            model_args=request_model_args,
            inference_args=request_inference_args,
            new_token_ids=self.new_token_ids,
            image_token_id=self.image_token_id,
            device=self.device,
            save_source_video=False,
            save_path_gen=str(save_dir),
            save_path_gt="",
        )
        save_prompt_results(request_inference_args.prompt_data_dict, str(save_dir), log_logger)
        clean_memory()

    # ---- task: t2i / t2v ------------------------------------------------

    def _run_generation(self, job: Job) -> Dict[str, Any]:
        task = job.task
        params = job.params
        ext = "png" if task == "t2i" else "mp4"
        payload = {f"000000.{ext}": job.prompt}

        prompt_file = JSONS_DIR / f"{job.id}_{task}.json"
        with prompt_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        save_dir = RESULTS_DIR / job.id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Pick sensible per-task defaults if caller didn't override
        if task == "t2i":
            params.setdefault("num_frames", 1)
            params.setdefault("resolution", "image_768res")
            params.setdefault("height", 768)
            params.setdefault("width", 768)
        else:
            params.setdefault("num_frames", 81)
            params.setdefault("resolution", params.get("resolution", "video_480p"))
            # 192p ~ 360x192, 360p ~ 640x360, 480p ~ 832x480
            res = params["resolution"]
            if res == "video_192p":
                params.setdefault("height", 192); params.setdefault("width", 320)
            elif res == "video_360p":
                params.setdefault("height", 384); params.setdefault("width", 640)
            else:
                params.setdefault("height", 480); params.setdefault("width", 832)

        request_model_args, request_data_args, request_inference_args = self._make_request_args(
            task, prompt_file, save_dir, params
        )
        val_data = self._build_batch(prompt_file, request_model_args, request_data_args, request_inference_args)
        self._validate(val_data, request_model_args, request_inference_args, save_dir)

        files = sorted(save_dir.glob(f"*.{ext}"))
        if not files:
            files = sorted(save_dir.glob("*.png") if task == "t2i" else save_dir.glob("*.mp4"))
        if not files:
            raise RuntimeError(f"{task} produced no output in {save_dir}")
        out = files[0]
        return {
            "kind": "image" if task == "t2i" else "video",
            "media_url": f"/api/media/{job.id}/{out.name}",
            "media_filename": out.name,
        }

    # ---- task: image_edit / video_edit ----------------------------------

    def _run_edit(self, job: Job) -> Dict[str, Any]:
        task = job.task
        params = job.params
        if not job.attachment_path:
            raise ValueError(f"{task} requires an attachment")

        is_video = task == "video_edit"
        cond_path = Path(job.attachment_path).resolve()
        payload = {
            "0001": {
                "interleave_array": [job.prompt, str(cond_path), str(cond_path)],
                "element_dtype_array": [
                    "text",
                    "video" if is_video else "image",
                    "video" if is_video else "image",
                ],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
        prompt_file = JSONS_DIR / f"{job.id}_{task}.json"
        with prompt_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        save_dir = RESULTS_DIR / job.id
        save_dir.mkdir(parents=True, exist_ok=True)

        if is_video:
            params.setdefault("num_frames", 81)
            params.setdefault("resolution", "video_480p")
            params.setdefault("height", 480); params.setdefault("width", 832)
        else:
            params.setdefault("num_frames", 1)
            params.setdefault("resolution", "image_768res")
            params.setdefault("height", 768); params.setdefault("width", 768)

        request_model_args, request_data_args, request_inference_args = self._make_request_args(
            task, prompt_file, save_dir, params
        )
        val_data = self._build_batch(prompt_file, request_model_args, request_data_args, request_inference_args)
        self._validate(val_data, request_model_args, request_inference_args, save_dir)

        # Edit tasks write filenames like 0001.png / 0001.mp4
        ext_choices = ("mp4",) if is_video else ("png", "jpg")
        outs: List[Path] = []
        for ext in ext_choices:
            outs.extend(sorted(save_dir.glob(f"*.{ext}")))
        if not outs:
            raise RuntimeError(f"{task} produced no output in {save_dir}")
        out = outs[0]
        return {
            "kind": "video" if is_video else "image",
            "media_url": f"/api/media/{job.id}/{out.name}",
            "media_filename": out.name,
        }

    # ---- task: x2t_image / x2t_video ------------------------------------

    def _run_understanding(self, job: Job) -> Dict[str, Any]:
        task = job.task
        if not job.attachment_path:
            raise ValueError(f"{task} requires an attachment")
        is_video = task == "x2t_video"
        cond_path = Path(job.attachment_path).resolve()

        question = (job.prompt or "Describe what you see.").strip()
        system_prompt = (
            "Watch the video carefully and answer the question."
            if is_video else
            "Look at the image carefully and answer the question."
        )

        payload = {
            "0001": {
                "interleave_array": [str(cond_path), [system_prompt, question, ""]],
                "element_dtype_array": ["video" if is_video else "image", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
        prompt_file = JSONS_DIR / f"{job.id}_{task}.json"
        with prompt_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        save_dir = RESULTS_DIR / job.id
        save_dir.mkdir(parents=True, exist_ok=True)

        params = dict(job.params)
        params.setdefault("num_frames", 1)
        params.setdefault("resolution", "image_768res" if not is_video else "video_480p")
        params.setdefault("height", 480 if is_video else 768)
        params.setdefault("width", 832 if is_video else 768)

        request_model_args, request_data_args, request_inference_args = self._make_request_args(
            task, prompt_file, save_dir, params
        )
        val_data = self._build_batch(prompt_file, request_model_args, request_data_args, request_inference_args)
        self._validate(val_data, request_model_args, request_inference_args, save_dir)

        # x2t writes prompt_data.json with the predicted captions
        text_result = self._extract_text_result(save_dir)
        return {
            "kind": "text",
            "text": text_result,
        }

    @staticmethod
    def _extract_text_result(save_dir: Path) -> str:
        """Find the generated caption text in save_dir.

        The official x2t pipeline writes `prompt.json` as
        `{ "<filename>.<ext>": "<answer><|im_end|>" }`.
        """
        for fname in ("prompt.json", "prompt_data.json", "prompt_results.json"):
            candidate = save_dir / fname
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text())
                    if isinstance(data, dict) and data:
                        first = next(iter(data.values()))
                        if isinstance(first, str):
                            return _clean_chat_suffix(first)
                        if isinstance(first, dict):
                            for k in ("caption", "answer", "response", "text"):
                                if k in first and isinstance(first[k], str):
                                    return _clean_chat_suffix(first[k])
                except Exception:
                    pass
        for txt in sorted(save_dir.glob("*.txt")):
            try:
                return _clean_chat_suffix(txt.read_text())
            except Exception:
                pass
        return "(no text returned)"


def log_logger(msg: str = ""):
    """Minimal logger object that satisfies inference_lance.save_prompt_results."""
    log(str(msg))


_CHAT_SUFFIXES = ("<|im_end|>", "<|endoftext|>", "</s>")


def _clean_chat_suffix(s: str) -> str:
    s = (s or "").strip()
    for suf in _CHAT_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].rstrip()
    return s


# ---------------------------------------------------------------------------
# Text chat pipeline (pure-text via HF Qwen2.5-VL on extracted understand ckpt)
# ---------------------------------------------------------------------------

class TextChatPipeline:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.lock = threading.RLock()
        self.initialized = False
        self.model = None
        self.tokenizer = None

    def initialize(self) -> None:
        with self.lock:
            if self.initialized:
                return
            ckpt_dir = ROOT / "weights" / "lance_3b_understand"
            if not ckpt_dir.exists():
                raise RuntimeError(
                    f"text chat needs the extracted understanding ckpt at {ckpt_dir}. "
                    "Run: python -m lance extract_understanding ..."
                )
            log(f"loading text-chat model: {ckpt_dir}")
            from lance.qknorm_patch import patch_qwen2_5_vl_qknorm
            patch_qwen2_5_vl_qknorm()
            from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(ckpt_dir), dtype=torch.bfloat16
            )
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            log("text-chat model ready")

    @torch.no_grad()
    def chat(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.initialize()
        # The extracted understanding tokenizer didn't ship with a chat
        # template, so build the ChatML prompt manually. Qwen 2.5 format.
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.lock:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Top-level server
# ---------------------------------------------------------------------------

QUESTION_HEADS = (
    "what", "how", "why", "when", "where", "who", "which",
    "describe", "explain", "tell me", "is there", "are there",
    "does", "do you", "can you", "could you", "is this", "are these",
)


def looks_like_question(prompt: str) -> bool:
    p = (prompt or "").strip().lower()
    if not p:
        return True  # empty prompt with attached media → describe
    if p.endswith("?"):
        return True
    return any(p.startswith(h) for h in QUESTION_HEADS)


def decide_task(prompt: str, mode: str, attachment_kind: Optional[str]) -> str:
    mode = (mode or "auto").lower()
    if attachment_kind is None:
        if mode == "image":
            return "t2i"
        if mode == "video":
            return "t2v"
        return "text"
    if attachment_kind == "image":
        if mode == "understand":
            return "x2t_image"
        if mode == "image":
            return "image_edit"
        if mode == "auto":
            return "x2t_image" if looks_like_question(prompt) else "image_edit"
        # mode="video" with image attachment: treat as ambient — fall back to t2v? no, that doesn't use the image. Reject:
        return "x2t_image"
    if attachment_kind == "video":
        if mode == "understand":
            return "x2t_video"
        if mode == "video":
            return "video_edit"
        if mode == "auto":
            return "x2t_video" if looks_like_question(prompt) else "video_edit"
        return "x2t_video"
    return "text"


def detect_attachment_kind(filename: str, content_type: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    suffix = Path(filename).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return "image"
    if suffix in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        return "video"
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("video/"):
            return "video"
    return None


# ---------------------------------------------------------------------------
# Conversation state + agentic message handling
# ---------------------------------------------------------------------------

from webui.orchestrator import (  # noqa: E402
    LANCE_TOOLS,
    OrchestratorClient,
    Settings as OrchSettings,
    SYSTEM_PROMPT,
    file_to_data_url,
)
from webui.jobs import (  # noqa: E402
    Asset,
    EventBus,
    JobRecord,
    JobRunner,
    new_id,
)


@dataclass
class ConvMessage:
    """One message in a conversation (server-side representation).

    `oai_message` is the dict we hand to the orchestrator. `display_blocks`
    is what the UI shows: a list of bubbles (text, image, video, tool_call,
    tool_result). They diverge because the orchestrator sees data URLs for
    images while the UI shows server media URLs.

    `tool_call_id` is set on role="tool" messages so we can locate and
    back-fill them when an async job completes.
    """
    role: str                       # "system" | "user" | "assistant" | "tool"
    oai_message: Dict[str, Any]     # serializable for orchestrator
    display_blocks: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    tool_call_id: Optional[str] = None
    job_id: Optional[str] = None    # set when this message is the orchestrator's queue-placeholder


@dataclass
class Conversation:
    id: str
    messages: List[ConvMessage] = field(default_factory=list)
    assets: List[Asset] = field(default_factory=list)
    jobs: Dict[str, JobRecord] = field(default_factory=dict)
    events: EventBus = field(default_factory=EventBus)
    # Per-conversation lock so orchestrator turns are serialized
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Legacy "most-recent" pointers (still used by the Lance-native path
    # and as the default target for edit_* when no asset_id is given).
    last_image_path: Optional[str] = None
    last_video_path: Optional[str] = None

    def append(self, m: ConvMessage) -> None:
        self.messages.append(m)
        for blk in m.display_blocks:
            if blk.get("kind") == "image":
                self.last_image_path = blk.get("path") or self.last_image_path
            elif blk.get("kind") == "video":
                self.last_video_path = blk.get("path") or self.last_video_path

    def oai_messages(self) -> List[Dict[str, Any]]:
        return [m.oai_message for m in self.messages]

    # ---- asset helpers -------------------------------------------------

    def add_asset(self, asset: Asset) -> None:
        self.assets.append(asset)
        if asset.local_path:
            if asset.kind == "image":
                self.last_image_path = asset.local_path
            elif asset.kind == "video":
                self.last_video_path = asset.local_path

    def find_asset(self, asset_id: str) -> Optional[Asset]:
        for a in self.assets:
            if a.id == asset_id:
                return a
        return None

    def latest_asset(self, kind: str) -> Optional[Asset]:
        for a in reversed(self.assets):
            if a.kind == kind:
                return a
        return None

    # ---- tool-message back-fill ----------------------------------------

    def find_tool_message(self, tool_call_id: str) -> Optional[ConvMessage]:
        for m in self.messages:
            if m.role == "tool" and m.tool_call_id == tool_call_id:
                return m
        return None


# ---------- App state -------------------------------------------------------

class AppState:
    def __init__(self):
        self.pipeline = LancePipeline(device=0)
        self.text = TextChatPipeline(device="cuda:0")
        # Orchestrator
        self.orch_settings = OrchSettings.from_env()
        self.orch_client = OrchestratorClient(self.orch_settings)
        self.orch_probe: Dict[str, Any] = {"reachable": False, "model": "", "models": [], "error": "not probed yet"}
        # Conversations
        self.conversations: Dict[str, Conversation] = {}
        # Async job runtime (single FIFO worker, started in app startup)
        self.runner: JobRunner = JobRunner(
            get_conversation=self.conversations.get,
            execute=self._lance_execute,
            build_asset=self._build_asset,
            backfill_tool_message=self._backfill_tool_message,
        )

    # ---- callbacks invoked by JobRunner --------------------------------

    def _lance_execute(self, job: JobRecord) -> Dict[str, Any]:
        """Run a single Lance task synchronously inside the worker thread.

        Bridges the JobRecord shape used by JobRunner to the legacy `Job`
        shape that LancePipeline.run() expects.
        """
        if job.lance_task == "text":
            text = self.text.chat(
                job.prompt, max_new_tokens=int(job.lance_params.get("max_new_tokens", 512))
            )
            return {"kind": "text", "text": text}
        legacy = Job(
            id=job.id,
            task=job.lance_task,
            prompt=job.prompt,
            attachment_path=job.attachment_path,
            attachment_kind=job.attachment_kind,
            params=dict(job.lance_params),
        )
        return self.pipeline.run(legacy)

    def _build_asset(self, job: JobRecord, result: Dict[str, Any]) -> Optional[Asset]:
        """Promote a finished media job to an Asset on its conversation."""
        kind = result.get("kind")
        if kind not in ("image", "video"):
            return None
        conv = self.conversations.get(job.conversation_id)
        if conv is None:
            return None
        media_url = result.get("media_url", "")
        local_path = None
        m = re.match(r"^/api/media/([^/]+)/(.+)$", media_url)
        if m:
            local_path = str(RESULTS_DIR / m.group(1) / m.group(2))
        asset = Asset(
            id=new_id("a"),
            conversation_id=conv.id,
            kind=kind,
            url=media_url,
            filename=result.get("media_filename", ""),
            caption=job.prompt,
            source=job.tool,
            job_id=job.id,
            local_path=local_path,
        )
        conv.add_asset(asset)
        return asset

    def _backfill_tool_message(
        self,
        job: JobRecord,
        result: Dict[str, Any],
        asset: Optional[Asset],
    ) -> None:
        """When an async job completes, rewrite the placeholder tool
        message that originally said `{status: queued}` so that the next
        orchestrator turn sees the real result in conversation history.
        """
        if not job.tool_call_id:
            return
        conv = self.conversations.get(job.conversation_id)
        if conv is None:
            return
        msg = conv.find_tool_message(job.tool_call_id)
        if msg is None:
            return
        kind = result.get("kind")
        if kind == "image":
            text = (
                f"Job {job.id} completed: image generated successfully "
                f"(filename={result.get('media_filename', '')}). "
                f"asset_id={asset.id if asset else 'unknown'}. "
                f"The image is now visible in the chat."
            )
        elif kind == "video":
            text = (
                f"Job {job.id} completed: video generated successfully "
                f"(filename={result.get('media_filename', '')}). "
                f"asset_id={asset.id if asset else 'unknown'}. "
                f"The video is now visible in the chat."
            )
        elif kind == "text":
            text = result.get("text", "")
        elif kind == "error":
            text = f"Job {job.id} FAILED: {result.get('error', 'unknown error')}"
        else:
            text = f"Job {job.id} completed: {json.dumps(result, default=str)[:500]}"
        msg.oai_message["content"] = text

        # Patch the display block too so subsequent /api/conversations/{cid}
        # reads (including page reloads) reflect the completion.
        for blk in msg.display_blocks:
            if blk.get("kind") == "tool_result" and blk.get("job_id") == job.id:
                blk["status"] = "done" if kind != "error" else "error"
                blk["elapsed"] = (job.finished_at or time.time()) - (job.started_at or job.created_at)
                if kind in ("image", "video") and asset is not None:
                    blk["media_url"] = asset.url
                    blk["media_filename"] = asset.filename
                    blk["kind_inner"] = asset.kind
                    blk["asset_id"] = asset.id
                if kind == "error":
                    blk["error"] = result.get("error", "")
                if kind == "text":
                    blk["text"] = result.get("text", "")

    # ---- factories used by message-handling code ----------------------

    def make_job(
        self,
        conv: Conversation,
        tool: str,
        args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> JobRecord:
        """Translate an orchestrator tool call into a JobRecord ready for
        the JobRunner. Validates references like asset_id for edit_*.
        """
        if tool == "generate_image":
            aspect = args.get("aspect", "square")
            sizes = {"square": (768, 768), "landscape": (1024, 576), "portrait": (576, 1024)}
            w, h = sizes.get(aspect, (768, 768))
            return JobRecord(
                id=new_id("j"), conversation_id=conv.id, tool=tool, args=dict(args),
                lance_task="t2i",
                lance_params={"steps": 30, "cfg_text_scale": 4.0, "seed": 42,
                              "height": h, "width": w, "resolution": "image_768res",
                              "num_frames": 1, "timestep_shift": 3.5},
                prompt=args.get("prompt", ""),
                tool_call_id=tool_call_id,
            )
        if tool == "generate_video":
            res = args.get("resolution", "192p")
            res_map = {"192p": ("video_192p", 192, 320),
                       "360p": ("video_360p", 384, 640),
                       "480p": ("video_480p", 480, 832)}
            res_key, h, w = res_map.get(res, res_map["192p"])
            nf = int(args.get("num_frames", 49))
            return JobRecord(
                id=new_id("j"), conversation_id=conv.id, tool=tool, args=dict(args),
                lance_task="t2v",
                lance_params={"steps": 30, "cfg_text_scale": 4.0, "seed": 42,
                              "height": h, "width": w, "resolution": res_key,
                              "num_frames": nf, "timestep_shift": 3.5},
                prompt=args.get("prompt", ""),
                tool_call_id=tool_call_id,
            )
        if tool == "edit_image":
            target_asset = None
            asset_id = args.get("asset_id")
            if asset_id:
                target_asset = conv.find_asset(asset_id)
                if target_asset is None or target_asset.kind != "image":
                    raise ValueError(f"unknown image asset_id={asset_id}")
            else:
                target_asset = conv.latest_asset("image")
            attachment_path = (target_asset.local_path
                               if target_asset is not None
                               else conv.last_image_path)
            if not attachment_path:
                raise ValueError("no image available to edit; ask the user to attach one or generate one first")
            return JobRecord(
                id=new_id("j"), conversation_id=conv.id, tool=tool, args=dict(args),
                lance_task="image_edit",
                lance_params={"steps": 30, "cfg_text_scale": 4.0, "seed": 42,
                              "height": 768, "width": 768, "resolution": "image_768res",
                              "num_frames": 1, "timestep_shift": 3.5},
                prompt=args.get("instruction", ""),
                attachment_path=attachment_path, attachment_kind="image",
                tool_call_id=tool_call_id,
            )
        if tool == "edit_video":
            target_asset = None
            asset_id = args.get("asset_id")
            if asset_id:
                target_asset = conv.find_asset(asset_id)
                if target_asset is None or target_asset.kind != "video":
                    raise ValueError(f"unknown video asset_id={asset_id}")
            else:
                target_asset = conv.latest_asset("video")
            attachment_path = (target_asset.local_path
                               if target_asset is not None
                               else conv.last_video_path)
            if not attachment_path:
                raise ValueError("no video available to edit")
            return JobRecord(
                id=new_id("j"), conversation_id=conv.id, tool=tool, args=dict(args),
                lance_task="video_edit",
                lance_params={"steps": 30, "cfg_text_scale": 4.0, "seed": 42,
                              "height": 480, "width": 832, "resolution": "video_480p",
                              "num_frames": 81, "timestep_shift": 3.5},
                prompt=args.get("instruction", ""),
                attachment_path=attachment_path, attachment_kind="video",
                tool_call_id=tool_call_id,
            )
        raise ValueError(f"unknown generation tool: {tool}")


state = AppState()


# ---------------------------------------------------------------------------
# Read-only orchestrator tools (synchronous state queries)
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS = {"list_jobs", "get_job", "list_assets", "get_asset", "cancel_job"}
GENERATION_TOOLS = {"generate_image", "generate_video", "edit_image", "edit_video"}


def execute_read_only_tool(name: str, args: Dict[str, Any], conv: Conversation) -> Dict[str, Any]:
    """Synchronous (non-Lance) tools the orchestrator can call to inspect
    or manipulate conversation state. Fast, no GPU.
    """
    if name == "list_jobs":
        wanted = (args.get("status") or "all").lower()
        limit = int(args.get("limit") or 20)
        items = list(conv.jobs.values())
        items.sort(key=lambda j: j.created_at, reverse=True)
        if wanted == "active":
            items = [j for j in items if j.status in ("queued", "running")]
        elif wanted != "all":
            items = [j for j in items if j.status == wanted]
        items = items[:limit]
        return {"count": len(items), "jobs": [j.short_dict() for j in items]}
    if name == "get_job":
        job_id = args.get("job_id", "")
        job = conv.jobs.get(job_id)
        if job is None:
            return {"error": f"unknown job_id={job_id}"}
        return {"job": job.public_dict()}
    if name == "list_assets":
        wanted = (args.get("kind") or "all").lower()
        limit = int(args.get("limit") or 20)
        items = list(conv.assets)
        items.sort(key=lambda a: a.created_at, reverse=True)
        if wanted != "all":
            items = [a for a in items if a.kind == wanted]
        items = items[:limit]
        return {"count": len(items), "assets": [a.public_dict() for a in items]}
    if name == "get_asset":
        asset_id = args.get("asset_id", "")
        asset = conv.find_asset(asset_id)
        if asset is None:
            return {"error": f"unknown asset_id={asset_id}"}
        return {"asset": asset.public_dict()}
    if name == "cancel_job":
        job_id = args.get("job_id", "")
        ok = state.runner.cancel(job_id, conversation_id=conv.id)
        return {"cancelled": ok, "job_id": job_id}
    raise ValueError(f"unknown read-only tool: {name}")


# ---------------------------------------------------------------------------
# User-message ingestion (shared by both paths)
# ---------------------------------------------------------------------------

def _record_user_message(
    conv: Conversation,
    user_prompt: str,
    user_attachment: Optional[Dict[str, Any]],
    output_mode: str,
) -> None:
    """Append the user's message to the conversation history and emit it
    to the event bus. Handles attachments (image/video) — they become
    user-uploaded assets so the orchestrator can list them too.
    """
    display_blocks: List[Dict[str, Any]] = []
    image_blocks: List[Dict[str, Any]] = []
    extra_text_notes: List[str] = []
    embed_user_images = state.orch_settings.embed_tool_images

    if user_prompt:
        display_blocks.append({"kind": "text", "text": user_prompt})

    if user_attachment:
        kind = user_attachment["kind"]
        path = Path(user_attachment["path"])
        media_url = user_attachment.get("media_url", "")
        display_blocks.append({
            "kind": kind, "path": str(path), "media_url": media_url,
            "filename": user_attachment.get("filename", path.name),
        })
        # Register the upload as an asset so the orchestrator can reference it
        upload_asset = Asset(
            id=new_id("a"),
            conversation_id=conv.id,
            kind=kind,
            url=media_url,
            filename=user_attachment.get("filename", path.name),
            caption="(uploaded by user)",
            source="user_upload",
            local_path=str(path),
        )
        conv.add_asset(upload_asset)
        if kind == "image":
            if embed_user_images:
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": file_to_data_url(path)},
                })
            else:
                extra_text_notes.append(
                    f"[The user attached an image (asset_id={upload_asset.id}). "
                    "Your orchestrator may not be vision-capable; the image is available as the most recent image for `edit_image`.]"
                )
        elif kind == "video":
            extra_text_notes.append(
                f"[The user attached a video clip (asset_id={upload_asset.id}). "
                "It's available for `edit_video`.]"
            )

    text_parts: List[str] = []
    if user_prompt:
        text_parts.append(user_prompt)
    text_parts.extend(extra_text_notes)
    text_str = "\n\n".join(p for p in text_parts if p)

    if image_blocks:
        user_oai_content: Any = [{"type": "text", "text": text_str}, *image_blocks]
    else:
        user_oai_content = text_str

    if output_mode and output_mode not in ("auto", "text"):
        mode_hint = {
            "image":      "The user prefers an image as the output of this turn — call generate_image or edit_image.",
            "video":      "The user prefers a video as the output of this turn — call generate_video or edit_video.",
            "understand": "The user wants a description / analysis of the attached media; do not call any tools.",
        }.get(output_mode)
        if mode_hint:
            if isinstance(user_oai_content, list):
                user_oai_content[0]["text"] += f"\n\n[output mode: {mode_hint}]"
            else:
                user_oai_content = f"{user_oai_content}\n\n[output mode: {mode_hint}]"

    conv.append(ConvMessage(
        role="user",
        oai_message={"role": "user", "content": user_oai_content},
        display_blocks=display_blocks,
    ))
    conv.events.emit("message_appended", {
        "role": "user",
        "index": len(conv.messages) - 1,
        "display_blocks": display_blocks,
        "created_at": conv.messages[-1].created_at,
    })


# ---------------------------------------------------------------------------
# Agentic conversation runner — emits to event bus, async-submits Lance jobs
# ---------------------------------------------------------------------------

async def run_agentic_turn(conv: Conversation) -> None:
    """Run a single orchestrator turn on `conv`. Streams text deltas and
    issues tool calls; generation tools submit async jobs and immediately
    feed `{job_id, status: queued}` back to the orchestrator. The turn
    finishes when the orchestrator stops emitting tool calls.

    Output is pushed to `conv.events` (the persistent SSE bus). Errors
    are caught and emitted as `error` events; this coroutine never
    raises out.
    """
    turn_id = new_id("t")
    conv.events.emit("turn_started", {"turn_id": turn_id, "mode": "agentic"})

    # Build messages: system prompt + brief situation header + history.
    sit_lines = _situation_header(conv)
    system_content = SYSTEM_PROMPT
    if sit_lines:
        system_content = SYSTEM_PROMPT + "\n\n# Current context\n" + sit_lines
    base_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
    base_messages.extend(conv.oai_messages())

    try:
        max_tool_rounds = 6
        for _round_idx in range(max_tool_rounds):
            accumulated_text = ""
            accumulated_tool_calls: List[Dict[str, Any]] = []
            try:
                async for evt in state.orch_client.stream_chat(base_messages, tools=LANCE_TOOLS):
                    if evt["type"] == "text":
                        accumulated_text += evt["delta"]
                        conv.events.emit("text_delta", {"turn_id": turn_id, "delta": evt["delta"]})
                    elif evt["type"] == "tool_call_done":
                        accumulated_tool_calls = evt["calls"]
                    elif evt["type"] == "stop":
                        break
            except Exception as e:  # noqa: BLE001
                err = f"orchestrator failure: {type(e).__name__}: {e}"
                log(err)
                conv.events.emit("error", {"turn_id": turn_id, "message": err})
                conv.events.emit("turn_ended", {"turn_id": turn_id, "status": "error"})
                return

            # Record the assistant message (text only at first).
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            assistant_msg["content"] = accumulated_text if accumulated_text else None
            if accumulated_tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in accumulated_tool_calls
                ]

            ad_blocks: List[Dict[str, Any]] = []
            if accumulated_text:
                ad_blocks.append({"kind": "text", "text": accumulated_text})
            conv.append(ConvMessage(
                role="assistant",
                oai_message=assistant_msg,
                display_blocks=ad_blocks,
            ))
            base_messages.append(assistant_msg)
            conv.events.emit("message_appended", {
                "role": "assistant",
                "index": len(conv.messages) - 1,
                "display_blocks": ad_blocks,
                "created_at": conv.messages[-1].created_at,
            })

            if not accumulated_tool_calls:
                break

            # Process tool calls.
            for tc in accumulated_tool_calls:
                name = tc["name"]
                args = tc.get("arguments") or {}
                conv.events.emit("tool_call", {
                    "tool_call_id": tc["id"],
                    "name": name,
                    "args": args,
                })

                # --- Read-only / state-query tools: execute synchronously
                if name in READ_ONLY_TOOLS:
                    ro_result: Dict[str, Any] = {}
                    try:
                        ro_result = execute_read_only_tool(name, args, conv)
                        tool_text = json.dumps(ro_result, default=str)
                        kind = "info"
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"
                        ro_result = {"error": err}
                        tool_text = json.dumps(ro_result)
                        kind = "error"
                    tool_oai = {"role": "tool", "tool_call_id": tc["id"], "content": tool_text}
                    conv.append(ConvMessage(
                        role="tool",
                        oai_message=tool_oai,
                        display_blocks=[{"kind": "tool_result", "name": name,
                                         "text": tool_text[:300],
                                         "status": "done" if kind == "info" else "error"}],
                        tool_call_id=tc["id"],
                    ))
                    base_messages.append(tool_oai)
                    conv.events.emit("tool_result", {
                        "tool_call_id": tc["id"],
                        "name": name,
                        "result": ro_result,
                        "kind": kind,
                    })
                    continue

                # --- Generation tools: submit a Job, return queued envelope
                if name in GENERATION_TOOLS:
                    try:
                        job = state.make_job(conv, name, args, tool_call_id=tc["id"])
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"
                        log(f"submit {name} failed: {err}")
                        tool_oai = {"role": "tool", "tool_call_id": tc["id"],
                                    "content": json.dumps({"error": err})}
                        conv.append(ConvMessage(
                            role="tool",
                            oai_message=tool_oai,
                            display_blocks=[{"kind": "tool_error", "name": name, "error": err}],
                            tool_call_id=tc["id"],
                        ))
                        base_messages.append(tool_oai)
                        conv.events.emit("tool_result", {
                            "tool_call_id": tc["id"],
                            "name": name,
                            "result": {"error": err},
                            "kind": "error",
                        })
                        continue

                    # Submit (non-blocking)
                    state.runner.submit(job)
                    queued_payload = {
                        "job_id": job.id,
                        "status": "queued",
                        "message": (
                            f"Job {job.id} ({name}) has been queued. "
                            f"It will run in the background and the result will appear "
                            f"in the chat automatically when complete (~{_eta_hint(name, args)}). "
                            f"You can call list_jobs() to check progress, or simply respond to the user "
                            f"with a brief acknowledgement now — do NOT call this tool again."
                        ),
                    }
                    tool_text = json.dumps(queued_payload)
                    tool_oai = {"role": "tool", "tool_call_id": tc["id"], "content": tool_text}
                    # Placeholder display block — JobRunner back-fills it when done.
                    placeholder = {
                        "kind": "tool_result",
                        "name": name,
                        "job_id": job.id,
                        "tool_call_id": tc["id"],
                        "args": args,
                        "status": "queued",
                        "elapsed": 0.0,
                    }
                    conv.append(ConvMessage(
                        role="tool",
                        oai_message=tool_oai,
                        display_blocks=[placeholder],
                        tool_call_id=tc["id"],
                        job_id=job.id,
                    ))
                    base_messages.append(tool_oai)
                    conv.events.emit("tool_result", {
                        "tool_call_id": tc["id"],
                        "name": name,
                        "job_id": job.id,
                        "result": queued_payload,
                        "kind": "queued",
                    })
                    continue

                # Unknown tool
                err = f"unknown tool {name!r}"
                tool_oai = {"role": "tool", "tool_call_id": tc["id"],
                            "content": json.dumps({"error": err})}
                conv.append(ConvMessage(
                    role="tool", oai_message=tool_oai,
                    display_blocks=[{"kind": "tool_error", "name": name, "error": err}],
                    tool_call_id=tc["id"],
                ))
                base_messages.append(tool_oai)
                conv.events.emit("tool_result", {
                    "tool_call_id": tc["id"], "name": name,
                    "result": {"error": err}, "kind": "error",
                })

        conv.events.emit("turn_ended", {"turn_id": turn_id, "status": "ok"})
    except Exception as e:  # noqa: BLE001
        log(f"agentic turn crashed: {e}\n{traceback.format_exc()}")
        conv.events.emit("error", {"turn_id": turn_id, "message": str(e)})
        conv.events.emit("turn_ended", {"turn_id": turn_id, "status": "error"})


def _situation_header(conv: Conversation) -> str:
    """A compact situational summary prepended to the system prompt so the
    orchestrator is aware of pending jobs and prior assets without having
    to call list_jobs() every turn."""
    lines: List[str] = []
    active = [j for j in conv.jobs.values() if j.status in ("queued", "running")]
    done = [j for j in conv.jobs.values() if j.status == "done"]
    if active:
        lines.append(f"- {len(active)} job(s) currently {','.join(sorted({j.status for j in active}))}. "
                     f"Oldest started {_relative_ago(min(j.started_at or j.created_at for j in active))} ago.")
        for j in active[:4]:
            lines.append(f"    job {j.id} {j.tool} \"{j.prompt[:80]}\"")
    if conv.assets:
        kinds = {}
        for a in conv.assets:
            kinds[a.kind] = kinds.get(a.kind, 0) + 1
        summary = ", ".join(f"{n} {k}{'s' if n != 1 else ''}" for k, n in kinds.items())
        lines.append(f"- {len(conv.assets)} asset(s) in chat ({summary}). Use list_assets() to inspect.")
    if done:
        lines.append(f"- {len(done)} previous generation job(s) completed.")
    return "\n".join(lines)


def _relative_ago(ts: float) -> str:
    s = max(0.0, time.time() - ts)
    if s < 60: return f"{int(s)}s"
    if s < 3600: return f"{int(s/60)}m"
    return f"{int(s/3600)}h"


def _eta_hint(tool: str, args: Dict[str, Any]) -> str:
    if tool == "generate_image": return "2 min"
    if tool == "edit_image":     return "1 min"
    if tool == "edit_video":     return "26 min"
    if tool == "generate_video":
        res = args.get("resolution", "192p")
        return {"192p": "3 min", "360p": "8 min", "480p": "26 min"}.get(res, "5 min")
    return "a moment"


# ---------------------------------------------------------------------------
# Native-Lance turn runner (fallback when orchestrator is unreachable)
# ---------------------------------------------------------------------------

async def run_lance_native_turn(conv: Conversation, user_prompt: str,
                                user_attachment: Optional[Dict[str, Any]],
                                output_mode: str) -> None:
    """Pre-orchestrator behavior: use decide_task() to pick a task, submit
    a job (async, like the agentic path), then return. The frontend sees
    the result land via job_completed on the event bus.
    """
    turn_id = new_id("t")
    conv.events.emit("turn_started", {"turn_id": turn_id, "mode": "lance_native"})

    att_kind = user_attachment["kind"] if user_attachment else None
    task = decide_task(user_prompt, output_mode, att_kind)

    if task == "text":
        # Text fallback runs through the same job queue so the UI sees a
        # consistent job lifecycle.
        job = JobRecord(
            id=new_id("j"), conversation_id=conv.id, tool="text_chat",
            args={"prompt": user_prompt}, lance_task="text",
            lance_params={"max_new_tokens": 512},
            prompt=user_prompt,
        )
        # Add a placeholder tool_result so the UI shows progress and the
        # back-fill replaces it with the text answer.
        placeholder_tc_id = new_id("tc")
        job.tool_call_id = placeholder_tc_id
        tool_text = json.dumps({"job_id": job.id, "status": "queued"})
        conv.append(ConvMessage(
            role="tool",
            oai_message={"role": "tool", "tool_call_id": placeholder_tc_id, "content": tool_text},
            display_blocks=[{
                "kind": "tool_result", "name": "text_chat", "job_id": job.id,
                "tool_call_id": placeholder_tc_id, "args": {"prompt": user_prompt},
                "status": "queued", "elapsed": 0.0,
            }],
            tool_call_id=placeholder_tc_id, job_id=job.id,
        ))
        state.runner.submit(job)
        conv.events.emit("turn_ended", {"turn_id": turn_id, "status": "ok"})
        return

    # Media task — t2i / t2v / image_edit / video_edit / x2t_*
    job_params = {
        "steps": 30, "cfg_text_scale": 4.0, "seed": 42,
        "height": 768 if task in ("t2i", "image_edit") else 480,
        "width":  768 if task in ("t2i", "image_edit") else 832,
        "resolution": "image_768res" if task in ("t2i", "image_edit", "x2t_image") else "video_480p",
        "num_frames": 1 if task in ("t2i", "image_edit", "x2t_image") else 81,
        "timestep_shift": 3.5,
    }
    tool_name = {
        "t2i": "generate_image", "t2v": "generate_video",
        "image_edit": "edit_image", "video_edit": "edit_video",
        "x2t_image": "describe_image", "x2t_video": "describe_video",
    }.get(task, task)
    placeholder_tc_id = new_id("tc")
    job = JobRecord(
        id=new_id("j"), conversation_id=conv.id, tool=tool_name,
        args={"prompt": user_prompt} if task in ("t2i", "t2v") else {"instruction": user_prompt},
        lance_task=task, lance_params=job_params,
        prompt=user_prompt,
        attachment_path=user_attachment["path"] if user_attachment else None,
        attachment_kind=att_kind,
        tool_call_id=placeholder_tc_id,
    )
    # Synthetic placeholder tool message so UI/back-fill works the same way
    tool_text = json.dumps({"job_id": job.id, "status": "queued"})
    conv.append(ConvMessage(
        role="tool",
        oai_message={"role": "tool", "tool_call_id": placeholder_tc_id, "content": tool_text},
        display_blocks=[{
            "kind": "tool_result", "name": tool_name, "job_id": job.id,
            "tool_call_id": placeholder_tc_id, "args": {"prompt": user_prompt},
            "status": "queued", "elapsed": 0.0,
        }],
        tool_call_id=placeholder_tc_id, job_id=job.id,
    ))
    state.runner.submit(job)
    conv.events.emit("turn_ended", {"turn_id": turn_id, "status": "ok"})


# ---------- FastAPI app -----------------------------------------------------

app = FastAPI(title="Lance Assistant")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((WEBUI_DIR / "static" / "index.html").read_text())


app.mount("/static", StaticFiles(directory=str(WEBUI_DIR / "static")), name="static")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    snap = state.runner.queue_snapshot()
    total_jobs = sum(len(c.jobs) for c in state.conversations.values())
    active_jobs = sum(1 for c in state.conversations.values() for j in c.jobs.values()
                      if j.status in ("queued", "running"))
    return {
        "ready": state.pipeline.initialized,
        "text_ready": state.text.initialized,
        "device": str(state.pipeline.device),
        "queue_depth": snap["depth"],
        "current_job": snap["current"],
        "active_jobs": active_jobs,
        "total_jobs": total_jobs,
        "conversations": len(state.conversations),
        "orchestrator": {
            "configured": bool(state.orch_settings.base_url),
            "base_url": state.orch_settings.base_url,
            "reachable": state.orch_probe.get("reachable", False),
            "model": state.orch_probe.get("model", ""),
            "models": state.orch_probe.get("models", []),
            "error": state.orch_probe.get("error", ""),
        },
    }


@app.post("/api/orchestrator/probe")
async def orchestrator_probe() -> Dict[str, Any]:
    """Re-probe the orchestrator. Useful when LM Studio starts up after
    the webui."""
    state.orch_probe = await state.orch_client.probe()
    return state.orch_probe


@app.get("/api/media/{job_id}/{name}")
async def media(job_id: str, name: str) -> FileResponse:
    path = RESULTS_DIR / job_id / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="media not found")
    mt, _ = mimetypes.guess_type(name)
    return FileResponse(path, media_type=mt or "application/octet-stream")


@app.get("/api/uploads/{name}")
async def serve_upload(name: str) -> FileResponse:
    """Serve a user-uploaded file (also used to render user-attached images
    in the chat history)."""
    path = UPLOADS_DIR / name
    if not path.exists():
        raise HTTPException(404, "upload not found")
    mt, _ = mimetypes.guess_type(name)
    return FileResponse(path, media_type=mt or "application/octet-stream")


# ---------------------------------------------------------------------------
# Conversations API
# ---------------------------------------------------------------------------

@app.post("/api/conversations")
async def create_conversation() -> Dict[str, Any]:
    cid = uuid.uuid4().hex
    conv = Conversation(id=cid)
    # Bind the bus to the running loop now so JobRunner emissions schedule
    # correctly.
    conv.events.bind_loop(asyncio.get_running_loop())
    state.conversations[cid] = conv
    log(f"created conversation {cid}")
    return {"conversation_id": cid}


@app.get("/api/conversations/{cid}")
async def get_conversation(cid: str) -> Dict[str, Any]:
    conv = state.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    return {
        "id": conv.id,
        "messages": [
            {"role": m.role, "display_blocks": m.display_blocks,
             "created_at": m.created_at,
             "tool_call_id": m.tool_call_id, "job_id": m.job_id}
            for m in conv.messages
        ],
        "assets": [a.public_dict() for a in conv.assets],
        "jobs":   [j.public_dict() for j in conv.jobs.values()],
        "latest_event_seq": conv.events.latest_seq,
    }


@app.post("/api/conversations/{cid}/messages")
async def post_message(
    cid: str,
    prompt: str = Form(""),
    mode: str = Form("auto"),
    use_orchestrator: bool = Form(True),
    attachment: Optional[UploadFile] = File(None),
) -> JSONResponse:
    """Enqueue a user message. Non-blocking: returns 202 with `{queued: true}`
    and the actual orchestrator/Lance work happens off the request thread.
    All output flows through the conversation's persistent SSE channel
    (/api/conversations/{cid}/events).
    """
    conv = state.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")

    user_att: Optional[Dict[str, Any]] = None
    if attachment is not None and attachment.filename:
        suffix = Path(attachment.filename).suffix or ".bin"
        upid = uuid.uuid4().hex
        save_to = UPLOADS_DIR / f"{upid}{suffix}"
        with save_to.open("wb") as f:
            shutil.copyfileobj(attachment.file, f)
        kind = detect_attachment_kind(attachment.filename, attachment.content_type)
        if kind is None:
            raise HTTPException(400, "unsupported attachment type")
        user_att = {
            "path": str(save_to),
            "kind": kind,
            "media_url": f"/api/uploads/{upid}{suffix}",
            "filename": attachment.filename,
        }

    use_agentic = use_orchestrator and state.orch_probe.get("reachable", False)
    log(f"conv {cid} new msg (orchestrator={'agentic' if use_agentic else 'native'}, "
        f"attach={user_att['kind'] if user_att else None})")

    # Record the user message synchronously (so the event order is sane)
    _record_user_message(conv, prompt, user_att, mode)

    # Kick off the turn task (non-blocking).
    asyncio.create_task(_run_turn_serialized(conv, prompt, user_att, mode, use_agentic))

    return JSONResponse({"queued": True, "conversation_id": cid}, status_code=202)


async def _run_turn_serialized(conv: Conversation, prompt: str,
                               user_att: Optional[Dict[str, Any]], mode: str,
                               use_agentic: bool) -> None:
    """Wrap a turn in the conversation's turn lock so concurrent user
    messages serialize cleanly. The lock has no effect on long-running
    Lance jobs because those are now async / detached."""
    async with conv.turn_lock:
        try:
            if use_agentic:
                await run_agentic_turn(conv)
            else:
                await run_lance_native_turn(conv, prompt, user_att, mode)
        except Exception as e:  # noqa: BLE001
            log(f"turn crashed for conv {conv.id}: {e}\n{traceback.format_exc()}")
            conv.events.emit("error", {"message": f"turn crashed: {e}"})


@app.get("/api/conversations/{cid}/events")
async def conversation_events(cid: str, request: Request, from_seq: int = 0) -> StreamingResponse:
    """Persistent SSE stream of all events for a conversation.

    The client should open this once per page load and keep it open. On
    reconnect, pass `?from_seq=N+1` where N is the last seq seen so the
    server replays anything missed.
    """
    conv = state.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    conv.events.bind_loop(asyncio.get_running_loop())

    async def stream() -> AsyncIterator[bytes]:
        # 1) Replay any events from cursor onward (this catches up reconnects
        #    or first-load).
        for ev in conv.events.replay(from_seq):
            yield _format_sse(ev.seq, ev.type, ev.payload)
        # 2) Subscribe to live events.
        q = conv.events.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue
                yield _format_sse(ev.seq, ev.type, ev.payload)
        finally:
            conv.events.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",       # disable nginx buffering if behind one
    })


def _format_sse(seq: int, type: str, payload: Dict[str, Any]) -> bytes:
    body = {"seq": seq, "type": type, "payload": payload}
    return f"data: {json.dumps(body, default=str)}\n\n".encode()


# ---------- Conversation-scoped jobs / assets endpoints ----------------------

@app.get("/api/conversations/{cid}/jobs")
async def list_conversation_jobs(cid: str, status: str = "all", limit: int = 50) -> Dict[str, Any]:
    conv = state.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    items = list(conv.jobs.values())
    items.sort(key=lambda j: j.created_at, reverse=True)
    if status != "all":
        items = [j for j in items if j.status == status]
    return {"count": len(items), "jobs": [j.public_dict() for j in items[:limit]]}


@app.post("/api/conversations/{cid}/jobs/{job_id}/cancel")
async def cancel_conversation_job(cid: str, job_id: str) -> Dict[str, Any]:
    conv = state.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    ok = state.runner.cancel(job_id, conversation_id=cid)
    return {"cancelled": ok, "job_id": job_id}


@app.get("/api/conversations/{cid}/assets")
async def list_conversation_assets(cid: str, kind: str = "all", limit: int = 50) -> Dict[str, Any]:
    conv = state.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    items = list(conv.assets)
    items.sort(key=lambda a: a.created_at, reverse=True)
    if kind != "all":
        items = [a for a in items if a.kind == kind]
    return {"count": len(items), "assets": [a.public_dict() for a in items[:limit]]}


# ---------- Startup ---------------------------------------------------------

@app.on_event("startup")
async def warmup() -> None:
    # Bind the runner's event-loop reference and start its worker thread.
    state.runner.start()
    log("startup: deferring Lance model load until first request")
    state.orch_probe = await state.orch_client.probe()
    if state.orch_probe.get("reachable"):
        log(f"orchestrator OK: {state.orch_settings.base_url} model={state.orch_probe.get('model')!r}")
    else:
        log(f"orchestrator UNREACHABLE ({state.orch_settings.base_url}) — falling back to native Lance dispatch. error={state.orch_probe.get('error')}")
