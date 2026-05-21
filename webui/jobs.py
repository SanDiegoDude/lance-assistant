"""Async job + asset + event-bus runtime for the Lance Assistant.

This module gives the chat backend three capabilities that the original
synchronous design lacked:

  1. **Async jobs**: Lance generation runs on a single FIFO worker thread
     so the HTTP/orchestrator path never blocks. The orchestrator can issue
     a `generate_image` tool call and we hand it back a `{job_id, status:
     queued}` envelope immediately. The user is free to chat again while
     Lance grinds on the GPU.

  2. **Asset registry**: every completed media job (plus user uploads) is
     tracked as an `Asset` with a stable ID. The orchestrator can list /
     fetch them as a tool, and `edit_image` can target a specific asset
     instead of only the most recent one.

  3. **Per-conversation event bus**: an append-only log of typed events
     (text_delta, job_started, job_completed, ...) fanned out to any
     number of subscriber queues. The frontend keeps a persistent SSE on
     this stream, so background-job completions arrive in the UI even
     when no message is being actively streamed. Replay-from-cursor lets
     reconnecting clients catch up without dropping anything.

The module is intentionally self-contained and doesn't import from
`server.py`, so cyclic deps are avoided.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import asyncio


# ---------------------------------------------------------------------------
# Logging (lightweight; the server has its own log() but we don't want to
# import from there)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[lance-jobs {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------

@dataclass
class Asset:
    """A piece of media tracked inside a conversation.

    Assets can come from three places:
      - Lance-generated output of a completed Job (kind="image" or "video",
        source="generate_image"/"generate_video"/"edit_image"/"edit_video")
      - A file the user uploaded (source="user_upload")
      - The result of a re-attached prior output (source="reattach")
    """
    id: str
    conversation_id: str
    kind: str                       # "image" | "video"
    url: str                        # /api/media/... or /api/uploads/...
    filename: str = ""
    caption: str = ""               # the prompt/instruction that produced it
    source: str = "unknown"         # generate_image | edit_image | user_upload | ...
    job_id: Optional[str] = None
    local_path: Optional[str] = None  # absolute path on disk (for re-feeding to Lance)
    created_at: float = field(default_factory=time.time)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "url": self.url,
            "filename": self.filename,
            "caption": self.caption,
            "source": self.source,
            "job_id": self.job_id,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# JobRecord — a single async Lance task tracked inside a conversation
# ---------------------------------------------------------------------------

@dataclass
class JobRecord:
    """A unit of Lance work submitted to the JobRunner.

    `lance_task` / `lance_params` / `prompt` / `attachment_path` are what
    the existing LancePipeline.run() consumes. The other fields are
    runtime tracking.
    """
    id: str
    conversation_id: str
    tool: str                       # generate_image | generate_video | edit_image | edit_video | text
    args: Dict[str, Any]            # original tool arguments from the orchestrator
    lance_task: str                 # t2i | t2v | image_edit | video_edit | x2t_image | x2t_video | text
    lance_params: Dict[str, Any]
    prompt: str
    attachment_path: Optional[str] = None
    attachment_kind: Optional[str] = None
    status: str = "queued"          # queued | running | done | failed | cancelled
    progress: float = 0.0
    progress_note: str = ""
    # Live progress reporting (populated by the executor via on_progress callback).
    # `step` / `total_steps` come from the diffusion loop (NaN-equivalent 0 means
    # not yet started, or a task that doesn't have a denoising loop like x2t).
    step: int = 0
    total_steps: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    tool_call_id: Optional[str] = None       # which orchestrator tool_call this came from
    asset_id: Optional[str] = None           # filled when result is media
    cancel_requested: bool = False

    def _elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def _eta_seconds(self) -> Optional[float]:
        """Linear extrapolation from elapsed step time. Only meaningful
        while running with at least one step recorded; otherwise None.
        """
        if self.status != "running":
            return None
        elapsed = self._elapsed()
        if elapsed is None or self.step <= 0 or self.total_steps <= 0:
            return None
        if self.step >= self.total_steps:
            return 0.0
        # Assume per-step time is constant; remaining = elapsed * (total/step - 1).
        return max(0.0, elapsed * (self.total_steps - self.step) / float(self.step))

    def public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "tool": self.tool,
            "args": self.args,
            "lance_task": self.lance_task,
            "prompt": self.prompt,
            "status": self.status,
            "progress": self.progress,
            "progress_note": self.progress_note,
            "step": self.step,
            "total_steps": self.total_steps,
            "result": self.result,
            "error": self.error,
            "asset_id": self.asset_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self._elapsed(),
            "eta_seconds": self._eta_seconds(),
        }

    def short_dict(self) -> Dict[str, Any]:
        """Compact view suitable for embedding in tool responses."""
        return {
            "id": self.id,
            "tool": self.tool,
            "status": self.status,
            "progress": round(self.progress, 3),
            "progress_note": self.progress_note,
            "elapsed_seconds": round(self._elapsed() or 0.0, 1),
            "eta_seconds": (round(self._eta_seconds(), 1)
                            if self._eta_seconds() is not None else None),
            "prompt": self.prompt[:120],
            "asset_id": self.asset_id,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# EventBus — per-conversation, append-only, fan-out, replay-from-cursor
# ---------------------------------------------------------------------------

@dataclass
class ConvEvent:
    seq: int                # monotonic per conversation, 0-indexed
    type: str               # text_delta, tool_call_issued, job_queued, job_started, ...
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe, append-only event log with async fan-out.

    `emit()` is callable from any thread (including the JobRunner worker).
    Subscribers receive a per-subscription `asyncio.Queue` that holds new
    events; they can also `replay(from_seq)` to pull historical events.

    The bus captures the running asyncio loop the first time `emit()` is
    called from a non-loop context, so that thread-safe puts can be
    scheduled correctly.
    """

    MAX_LOG = 4096            # cap memory; older events get truncated
    MAX_QUEUE = 4096

    def __init__(self) -> None:
        self._log: List[ConvEvent] = []
        self._log_lock = threading.Lock()
        self._subscribers: Set[asyncio.Queue] = set()
        self._subscribers_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -- loop binding -----------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the running event loop. Idempotent."""
        self._loop = loop

    def _ensure_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        if self._loop is not None:
            return self._loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return self._loop

    # -- producer side ----------------------------------------------------

    def emit(self, type: str, payload: Dict[str, Any]) -> ConvEvent:
        """Append an event and fan it out to all subscribers.

        Thread-safe — callable from the JobRunner worker thread.
        """
        with self._log_lock:
            seq = (self._log[-1].seq + 1) if self._log else 0
            ev = ConvEvent(seq=seq, type=type, payload=payload, ts=time.time())
            self._log.append(ev)
            # Truncate older events (but preserve seq numbers — clients
            # holding a high cursor won't get older events; that's OK).
            if len(self._log) > self.MAX_LOG:
                self._log = self._log[-self.MAX_LOG:]

        with self._subscribers_lock:
            subs = list(self._subscribers)

        loop = self._ensure_loop()
        if loop is not None:
            for q in subs:
                try:
                    loop.call_soon_threadsafe(self._safe_put, q, ev)
                except RuntimeError:
                    # loop is closing
                    pass
        return ev

    @staticmethod
    def _safe_put(q: asyncio.Queue, ev: ConvEvent) -> None:
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            # drop oldest, then push (best-effort)
            try:
                _ = q.get_nowait()
                q.put_nowait(ev)
            except Exception:
                pass

    # -- consumer side ----------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_QUEUE)
        with self._subscribers_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._subscribers_lock:
            self._subscribers.discard(q)

    def replay(self, from_seq: int = 0) -> List[ConvEvent]:
        """Return all log entries with seq >= from_seq."""
        with self._log_lock:
            # binary-search would be O(log n) but the list is small; linear is fine
            return [e for e in self._log if e.seq >= from_seq]

    @property
    def latest_seq(self) -> int:
        with self._log_lock:
            return self._log[-1].seq if self._log else -1


# ---------------------------------------------------------------------------
# JobRunner — single FIFO worker that owns the GPU
# ---------------------------------------------------------------------------

# Type aliases for the callbacks the runner invokes back into the server.
GetConvFn = Callable[[str], Any]           # conv_id -> Conversation-like (must have .events, .jobs)
# Progress callback handed to the executor; the executor invokes it from
# inside the diffusion loop (or wherever progress is observable). Signature:
#   on_progress(step:int, total_steps:int, note:str="")
ProgressFn = Callable[[int, int, str], None]
# The executor is called with (job, on_progress). on_progress is optional —
# executors that don't have step granularity may simply ignore it.
ExecuteFn = Callable[[JobRecord, ProgressFn], Dict[str, Any]]
PostProcessFn = Callable[[JobRecord, Dict[str, Any]], Optional[Asset]]  # build Asset from result
BackfillFn = Callable[[JobRecord, Dict[str, Any], Optional[Asset]], None]  # update ConvMessage


class JobRunner:
    """Background single-threaded worker for Lance jobs.

    The runner owns one Python thread, pulls JobRecords off a FIFO queue,
    and dispatches them via the provided `execute_fn` callback. It emits
    `job_started` / `job_progress` / `job_completed` / `job_failed` events
    onto the owning conversation's EventBus.

    We keep this single-worker because Lance lives on one GPU and isn't
    re-entrant.
    """

    def __init__(
        self,
        get_conversation: GetConvFn,
        execute: ExecuteFn,
        build_asset: PostProcessFn,
        backfill_tool_message: BackfillFn,
    ) -> None:
        self._get_conv = get_conversation
        self._execute = execute
        self._build_asset = build_asset
        self._backfill = backfill_tool_message
        self._queue: "queue.Queue[JobRecord]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="lance-job-runner")
        self._started = False
        self._current: Optional[JobRecord] = None
        self._current_lock = threading.Lock()

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    # ------------------------------------------------------------ submit

    def submit(self, job: JobRecord) -> None:
        """Queue a job. Thread-safe."""
        conv = self._get_conv(job.conversation_id)
        if conv is None:
            raise RuntimeError(f"unknown conversation {job.conversation_id}")
        conv.jobs[job.id] = job
        position = self._queue.qsize() + (1 if self._current is not None else 0)
        conv.events.emit("job_queued", {
            "job_id": job.id,
            "tool": job.tool,
            "lance_task": job.lance_task,
            "prompt": job.prompt,
            "args": job.args,
            "position": position,
            "tool_call_id": job.tool_call_id,
        })
        self._queue.put(job)

    def cancel(self, job_id: str, conversation_id: Optional[str] = None) -> bool:
        """Best-effort cancel.

        If the job is queued, mark it cancelled and the worker will skip
        it. If it's currently running, we have no way to preempt Lance
        cleanly; we set `cancel_requested` so the caller knows we tried
        but the GPU will keep going. Returns True if status changed.
        """
        # We don't have direct queue mutation, but the worker checks status
        # before running. Find the job via the conversation.
        if conversation_id:
            conv = self._get_conv(conversation_id)
            job = conv.jobs.get(job_id) if conv else None
        else:
            job = None
            # caller didn't tell us which conv; scan via current
            with self._current_lock:
                if self._current and self._current.id == job_id:
                    job = self._current
        if job is None:
            return False
        if job.status in ("done", "failed", "cancelled"):
            return False
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            conv = self._get_conv(job.conversation_id)
            if conv is not None:
                conv.events.emit("job_cancelled", {"job_id": job.id, "tool": job.tool})
            return True
        # running — can't preempt, but caller knows
        return True

    def queue_snapshot(self) -> Dict[str, Any]:
        with self._current_lock:
            cur = self._current.short_dict() if self._current else None
        return {"current": cur, "depth": self._queue.qsize()}

    # ------------------------------------------------------------ worker

    def _loop(self) -> None:
        _log("worker thread started")
        while True:
            try:
                job = self._queue.get()
            except Exception:
                continue
            if job is None:
                continue
            if job.cancel_requested or job.status == "cancelled":
                continue
            self._run_one(job)

    def _run_one(self, job: JobRecord) -> None:
        conv = self._get_conv(job.conversation_id)
        if conv is None:
            _log(f"job {job.id} skipped: conversation {job.conversation_id} gone")
            return

        with self._current_lock:
            self._current = job
        job.status = "running"
        job.started_at = time.time()
        conv.events.emit("job_started", {
            "job_id": job.id,
            "tool": job.tool,
            "lance_task": job.lance_task,
            "prompt": job.prompt,
            "tool_call_id": job.tool_call_id,
        })

        # Throttle progress event emission so we don't flood the SSE stream
        # — 50 denoising steps over a 2-minute generation works out to one
        # step every ~2.4 s, so we'd never naturally exceed this, but a
        # faster machine could. Always emit the final step (step == total).
        last_emit = [0.0]
        EMIT_MIN_INTERVAL = 0.4

        def on_progress(step: int, total_steps: int, note: str = "") -> None:
            try:
                job.step = int(step)
                job.total_steps = int(total_steps)
                if total_steps > 0:
                    job.progress = max(0.0, min(1.0, step / float(total_steps)))
                if note:
                    job.progress_note = note
                else:
                    job.progress_note = f"step {step}/{total_steps}" if total_steps else ""
                now = time.time()
                is_final_step = (total_steps > 0 and step >= total_steps)
                if is_final_step or (now - last_emit[0]) >= EMIT_MIN_INTERVAL:
                    last_emit[0] = now
                    elapsed = job._elapsed()
                    eta = job._eta_seconds()
                    conv.events.emit("job_progress", {
                        "job_id": job.id,
                        "tool": job.tool,
                        "lance_task": job.lance_task,
                        "step": job.step,
                        "total_steps": job.total_steps,
                        "progress": job.progress,
                        "progress_note": job.progress_note,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                        "tool_call_id": job.tool_call_id,
                    })
            except Exception as e:  # noqa: BLE001
                # Never let a progress hook break the actual generation.
                _log(f"progress callback error for job {job.id}: {e}")

        try:
            result = self._execute(job, on_progress)
            job.result = result
            job.status = "done"
            job.finished_at = time.time()
            asset = self._build_asset(job, result)
            if asset is not None:
                job.asset_id = asset.id
            elapsed = job.finished_at - (job.started_at or job.created_at)
            payload = {
                "job_id": job.id,
                "tool": job.tool,
                "lance_task": job.lance_task,
                "result": result,
                "elapsed": elapsed,
                "asset_id": asset.id if asset else None,
                "asset": asset.public_dict() if asset else None,
                "tool_call_id": job.tool_call_id,
            }
            # Back-fill the tool message in the conversation BEFORE emitting
            # the event, so anyone reacting to `job_completed` (including a
            # future orchestrator turn) sees the updated history.
            try:
                self._backfill(job, result, asset)
            except Exception as e:  # noqa: BLE001
                _log(f"backfill error for job {job.id}: {e}")
            conv.events.emit("job_completed", payload)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            _log(f"job {job.id} failed: {err}\n{tb}")
            job.status = "failed"
            job.error = err
            job.finished_at = time.time()
            try:
                self._backfill(job, {"kind": "error", "error": err}, None)
            except Exception as be:  # noqa: BLE001
                _log(f"backfill error for failed job {job.id}: {be}")
            conv.events.emit("job_failed", {
                "job_id": job.id,
                "tool": job.tool,
                "error": err,
                "tool_call_id": job.tool_call_id,
            })
        finally:
            with self._current_lock:
                self._current = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex
    return f"{prefix}{raw[:16]}" if prefix else raw
