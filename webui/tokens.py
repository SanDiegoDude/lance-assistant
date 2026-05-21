"""Lightweight token budget estimation + history truncation.

The orchestrator can be anything OpenAI-compatible (LM Studio, vLLM,
Ollama, OpenAI itself, …) so we can't ask the *exact* tokenizer for an
authoritative count. What we want is a *cheap, conservative* estimate
that lets us:

  1. Show the user (in the UI) how much of the configured context
     window they've spent on this conversation.
  2. Trim the oldest non-system messages before they push the chat
     over the configured budget, so a long session never explodes
     on the orchestrator side.

We try `tiktoken` first (its `cl100k_base` is a reasonable proxy for
most modern transformers tokenizers and overcounts on average vs.
Qwen / Llama BPEs — which is exactly the side we want to err on for
a budget check). If `tiktoken` isn't importable we fall back to a
4-chars-per-token approximation.

Images and videos are counted with flat constants — VLM token costs
for vision content vary wildly between models (OpenAI charges ~85
base + ~170 per 512-px tile; Qwen2.5-VL is dynamic; some local builds
inline thousands). The constants here are a conservative middle
ground; the goal is "don't accidentally overshoot the orchestrator's
window", not "match the orchestrator's billed count exactly".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

try:
    import tiktoken  # type: ignore
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None

# Per-image budget. Most multimodal models we target spend somewhere
# between 600 and 1500 tokens on a single image; 1200 is a sensible
# default. Bump this if your orchestrator embeds very-high-res images
# at high detail.
IMAGE_TOKEN_COST = 1200

# Per video, we count IMAGE_TOKEN_COST * VIDEO_FRAMES_BUDGET — videos
# in our flow are typically passed as a stack of sampled frames.
VIDEO_FRAMES_BUDGET = 8


def _count_text(s: str) -> int:
    if not s:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(s, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(s) // 4)


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Conservative token count for one OpenAI-format chat message."""
    n = 4  # role + delimiter overhead
    content = msg.get("content")

    if isinstance(content, str):
        n += _count_text(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                n += _count_text(block.get("text", ""))
            elif kind == "image_url":
                n += IMAGE_TOKEN_COST
            elif kind == "video_url":
                n += IMAGE_TOKEN_COST * VIDEO_FRAMES_BUDGET
            else:
                # Unknown block — fall back to a JSON dump count so we
                # don't silently ignore something heavy.
                try:
                    n += _count_text(json.dumps(block, default=str))
                except Exception:
                    pass
    elif content is not None:
        try:
            n += _count_text(json.dumps(content, default=str))
        except Exception:
            pass

    # tool_calls / tool_call_id payloads are part of the wire format
    if msg.get("tool_calls"):
        try:
            n += _count_text(json.dumps(msg["tool_calls"], default=str))
        except Exception:
            pass
    if msg.get("name"):
        n += _count_text(str(msg["name"]))

    return n


def estimate_messages_tokens(msgs: List[Dict[str, Any]]) -> int:
    """Estimated total tokens for a chat completion request body."""
    return sum(estimate_message_tokens(m) for m in msgs) + 4  # priming


def truncate_to_budget(
    messages: List[Dict[str, Any]],
    budget: int,
    keep_recent: int = 12,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Drop oldest non-system / non-recent messages until under budget.

    Always preserves:
      * the very first message if it is ``role="system"``
      * the last ``keep_recent`` messages

    Returns ``(messages_out, before_tokens, after_tokens, dropped_count)``.
    The before/after numbers are useful for telemetry / UI display.
    """
    before = estimate_messages_tokens(messages)
    if before <= budget:
        return messages, before, before, 0

    head: List[Dict[str, Any]] = []
    body = messages
    if messages and messages[0].get("role") == "system":
        head = [messages[0]]
        body = messages[1:]

    if len(body) <= keep_recent:
        # Can't drop anything; the recent tail is already everything.
        return messages, before, before, 0

    tail = body[-keep_recent:]
    droppable = body[: len(body) - keep_recent]

    while droppable:
        # Avoid orphaning an assistant tool_calls message from its
        # matching tool result(s); we always drop in matched pairs.
        first = droppable[0]
        n_drop = 1
        if first.get("role") == "assistant" and first.get("tool_calls"):
            n_drop = 1 + sum(
                1 for m in droppable[1:]
                if m.get("role") == "tool"
            ) if len(droppable) > 1 else 1
            # be conservative: only orphan-protect immediately adjacent tools
            n_drop = 1
            for m in droppable[1:]:
                if m.get("role") == "tool":
                    n_drop += 1
                else:
                    break

        del droppable[:n_drop]
        candidate = head + droppable + tail
        if estimate_messages_tokens(candidate) <= budget:
            after = estimate_messages_tokens(candidate)
            return candidate, before, after, len(messages) - len(candidate)

    # Even head + tail exceed budget — accept it, the model will
    # complain about its own limit but at least we didn't push junk.
    final = head + tail
    after = estimate_messages_tokens(final)
    return final, before, after, len(messages) - len(final)
