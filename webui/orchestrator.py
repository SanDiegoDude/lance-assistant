"""OpenAI-compatible orchestrator client + Lance tool definitions.

The orchestrator is whatever VLM the user has running behind an
OpenAI-compatible API (LM Studio, Ollama, vLLM, llama.cpp server,
OpenAI itself). It drives the chat and emits function calls that we
translate into calls against the local Lance pipeline.

Public surface:

  Settings.from_env()                         -> Settings
  OrchestratorClient(settings)                -> client
    .probe()             -> {"reachable", "model", "models"}
    .stream_chat(messages, tools=...) -> async iterator over events:
        {"type": "text", "delta": str}
        {"type": "tool_call_partial", "index": int, "id": str|None,
            "name": str|None, "arguments_delta": str}
        {"type": "tool_call_done", "calls": [
            {"id": str, "name": str, "arguments": dict}, ...]}
        {"type": "stop", "finish_reason": str}

  LANCE_TOOLS              -> list of OpenAI tool descriptors
  SYSTEM_PROMPT            -> the base prompt prepended to every chat
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _truthy(s: str) -> bool:
    return (s or "").strip().lower() in {"1", "on", "true", "yes", "y", "enable", "enabled"}


@dataclass
class Settings:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    request_timeout: float = 600.0
    enabled: bool = True
    # If your orchestrator is a vision-language model, set this to True so
    # the server appends a user-role message containing the just-generated
    # image as an image_url block, letting the model actually see what was
    # made and respond accordingly. Off by default for broad compatibility
    # with text-only orchestrators (most local models don't accept image
    # content blocks).
    embed_tool_images: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv  # type: ignore
            # Search upward from this file for a .env
            here = Path(__file__).resolve().parent
            for d in (here, *here.parents):
                cand = d / ".env"
                if cand.exists():
                    load_dotenv(cand, override=False)
                    break
        except Exception:
            pass

        base_url = (os.getenv("ORCHESTRATOR_BASE_URL") or "http://localhost:1234/v1").strip()
        # Explicit disable: set ORCHESTRATOR_BASE_URL=off / none / disable / "".
        disabled_tokens = {"off", "none", "disable", "disabled", "no", "false", ""}
        enabled = base_url.lower() not in disabled_tokens
        return cls(
            base_url=base_url if enabled else "",
            api_key=(os.getenv("ORCHESTRATOR_API_KEY") or "").strip(),
            model=(os.getenv("ORCHESTRATOR_MODEL") or "").strip(),
            temperature=float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("ORCHESTRATOR_MAX_TOKENS", "2048")),
            request_timeout=float(os.getenv("ORCHESTRATOR_REQUEST_TIMEOUT", "600")),
            enabled=enabled,
            embed_tool_images=_truthy(os.getenv("ORCHESTRATOR_EMBED_TOOL_IMAGES", "off")),
        )


# ---------------------------------------------------------------------------
# System prompt + tool schemas
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Lance Assistant, a helpful multimodal AI. The user is talking to you through a chat UI that lets them ask questions, request images and videos, and edit media.

You have access to ByteDance Lance, a unified multimodal model. You yourself handle conversation, reasoning, and visual understanding (you can already see any images the user attaches). When the user asks for new pixels — a new image, a new video, an edit to an existing one — you call one of the tools below. Otherwise just answer directly.

Tools you can call:

- `generate_image(prompt, aspect)` — make a brand-new image from a text prompt. Use when the user asks to "draw", "create", "make", "generate", "render", "paint", "show me" an image / picture / photo / illustration.
- `generate_video(prompt, resolution, num_frames)` — make a short video clip. Use sparingly: video gen is SLOW (3 minutes at 192p, up to 26 minutes at 480p). If the user just says "make a video" without specifying quality, default to 192p and 49 frames; warn them in your text response that higher quality takes longer.
- `edit_image(instruction)` — apply an edit ("make it night-time", "add a hat", "change the hair color") to the most recently shown image in this conversation. Identity is preserved across the edit. Only available when there is at least one image in the chat history.
- `edit_video(instruction)` — same idea for the most recently shown video. Also slow (~26 min for 480p).

Rules:

- Be a normal helpful assistant for everything that isn't a media generation request. Answer questions, reason, code, explain — directly, without tools.
- When the user attaches media and asks about it, describe / analyze it yourself. You don't need a tool for that.
- When you DO call a tool, the system will execute it and show the result in the chat. Then you'll get a follow-up message containing what was produced. Respond with a brief, friendly confirmation ("Here's the image of …", "Done — what do you think?"). Don't repeat the whole prompt.
- Never invent images / videos in your text response. Either use a tool to produce them or don't claim they exist.
- Prefer concise responses with proper markdown. Use code fences (```language) for any code."""


def _tool(name: str, description: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


LANCE_TOOLS: List[Dict[str, Any]] = [
    _tool(
        "generate_image",
        "Generate a new image from a text prompt using the Lance image model. "
        "Use whenever the user asks to draw, create, make, generate, render, paint, "
        "or show an image / picture / photo / illustration. Takes about 2 minutes.",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed visual description of the image. Be specific about subject, style, lighting, composition, and any text to render. The model is great at both photorealistic and stylized images.",
                },
                "aspect": {
                    "type": "string",
                    "enum": ["square", "landscape", "portrait"],
                    "description": "Output aspect ratio. square=768x768, landscape=1024x576, portrait=576x1024.",
                    "default": "square",
                },
            },
            "required": ["prompt"],
        },
    ),
    _tool(
        "generate_video",
        "Generate a short video clip from a text prompt using the Lance video model. "
        "SLOW: ~3 min at 192p, ~8 min at 360p, ~26 min at 480p. Confirm before invoking unless the user explicitly asked for a video. "
        "Default to 192p / 49 frames unless the user requested better quality.",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed visual description for the video. Include camera motion, scene action, and style.",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["192p", "360p", "480p"],
                    "default": "192p",
                    "description": "192p=320x192 (fastest, ~3 min), 360p=640x384 (~8 min), 480p=832x480 (best, ~26 min).",
                },
                "num_frames": {
                    "type": "integer",
                    "default": 49,
                    "description": "Number of frames; must be 4*k+1. Valid: 13, 25, 49, 81, 121. At 16 fps: 49 frames = ~3 sec, 81 = ~5 sec, 121 = ~7.5 sec.",
                },
            },
            "required": ["prompt"],
        },
    ),
    _tool(
        "edit_image",
        "Apply an edit instruction to the most recent image shown in this conversation. "
        "Lance's image-edit model preserves identity (faces, objects, scene structure) while applying the requested change. "
        "Use when the user wants to modify, change, alter, or edit a picture. Takes about 1 minute.",
        {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "What to change about the image. Examples: 'make it night-time with stars', 'change the hair color to dark green', 'add a wizard hat', 'make it black and white'.",
                },
            },
            "required": ["instruction"],
        },
    ),
    _tool(
        "edit_video",
        "Apply an edit instruction to the most recent video shown in this conversation. "
        "SLOW (~26 min for 480p). Confirm with the user first.",
        {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "What to change about the video."},
            },
            "required": ["instruction"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Helpers — image embedding for VLM input
# ---------------------------------------------------------------------------

def file_to_data_url(path: Path) -> str:
    """Encode an image as a data: URL the orchestrator can consume."""
    suffix = path.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }.get(suffix, "image/png")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Orchestrator client
# ---------------------------------------------------------------------------

class OrchestratorClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._headers = {"Content-Type": "application/json"}
        if settings.api_key:
            self._headers["Authorization"] = f"Bearer {settings.api_key}"

    async def probe(self) -> Dict[str, Any]:
        """Try to talk to the orchestrator. Returns:
            {"reachable": True/False, "model": str, "models": [str, ...], "error": str}
        """
        if not self.settings.enabled:
            return {"reachable": False, "error": "disabled by config"}
        url = self.settings.base_url.rstrip("/") + "/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, headers=self._headers)
                r.raise_for_status()
                data = r.json()
            ids: List[str] = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                ids = [m.get("id") for m in data["data"] if isinstance(m, dict) and "id" in m]
            elif isinstance(data, list):
                ids = [m.get("id") for m in data if isinstance(m, dict) and "id" in m]
            picked = self.settings.model or (ids[0] if ids else "")
            return {"reachable": True, "model": picked, "models": ids}
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "error": f"{type(e).__name__}: {e}"}

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a chat completion. Yields normalized events (see module docstring)."""
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Accumulate per-index tool call state so we can emit a fully formed
        # `tool_call_done` event when streaming concludes.
        partial_tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        timeout = httpx.Timeout(self.settings.request_timeout, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=self._headers, json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")
                    raise RuntimeError(f"orchestrator HTTP {r.status_code}: {body[:400]}")

                async for raw_line in r.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                    except Exception:
                        continue
                    choices = evt.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if "content" in delta and delta["content"]:
                        yield {"type": "text", "delta": delta["content"]}
                    for tc in delta.get("tool_calls", []) or []:
                        idx = tc.get("index", 0)
                        slot = partial_tool_calls.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        func = tc.get("function") or {}
                        if func.get("name"):
                            slot["name"] = func["name"]
                        if func.get("arguments"):
                            slot["arguments"] += func["arguments"]
                        yield {
                            "type": "tool_call_partial",
                            "index": idx,
                            "id": slot["id"],
                            "name": slot["name"],
                            "arguments_delta": func.get("arguments", ""),
                        }
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        # Emit done event with any tool calls
        if partial_tool_calls:
            final_calls = []
            for idx in sorted(partial_tool_calls):
                slot = partial_tool_calls[idx]
                args_str = slot["arguments"] or "{}"
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"_raw": args_str}
                final_calls.append({
                    "id": slot["id"] or f"call_{idx}",
                    "name": slot["name"] or "",
                    "arguments": args,
                })
            yield {"type": "tool_call_done", "calls": final_calls}

        yield {"type": "stop", "finish_reason": finish_reason or "stop"}
