"""One-shot captioning of user-uploaded media via the orchestrator VLM.

When the user attaches an image or a video to a message we kick off a
self-contained captioning call against the orchestrator. The result is
written to the upload `Asset.caption` (and to a small dict the caller
keeps around so the live turn can inject it into the OAI history).

Design constraints driving this module:
  * The caption is **not** part of the user-visible chat. The user
    didn't ask "describe this to me", so we don't want it polluting the
    conversation. It only shows up as a brief status pill in the UI
    ("captioning attachment…" -> done) and gets silently prepended to
    the user's OAI message text so the orchestrator has a textual
    handle on what it's looking at.
  * Captioning must work for both vision-capable and text-only
    orchestrators. If the orchestrator can't see images we skip
    captioning (and the orchestrator just operates blind, same as
    before this module existed).
  * Videos are too expensive to send as a single blob. We sample at
    1 FPS, resize to 512px (longest edge), and send the frame stack to
    the orchestrator as a sequence of `image_url` blocks — the format
    Qwen2.5-VL and most other multimodal models expect.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image  # type: ignore

from .orchestrator import (
    OrchestratorClient,
    bytes_to_data_url,
    file_to_data_url,
)


_DEFAULT_MAX_FRAMES = 24
_DEFAULT_FRAME_LONG_EDGE = 512
_CAPTION_TIMEOUT = 90.0  # generous: video captioning can be slower than image
_CAPTION_MAX_TOKENS = 256

_IMAGE_CAPTION_SYSTEM = (
    "You are an assistant that describes images concisely and accurately. "
    "When given an image, reply with one or two sentences in plain prose, "
    "no preamble, no markdown. Mention the main subject(s), what they are "
    "doing, the setting, mood/lighting, and any clearly readable text. "
    "Do not speculate about identity or invent details that aren't visible."
)

_VIDEO_CAPTION_SYSTEM = (
    "You are an assistant that describes short videos concisely. "
    "You'll receive a sequence of frames sampled at roughly 1 frame per "
    "second from the same clip, in chronological order. Reply with one to "
    "three sentences in plain prose summarising what happens across the "
    "clip — the main subject(s), motion or actions, setting, and mood. "
    "Do not list the frames individually and don't speculate."
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

async def caption_image(client: OrchestratorClient, path: Path) -> str:
    """Caption a single image file using the orchestrator's VLM.

    Returns the caption text (stripped). Raises on transport errors; the
    caller should catch and degrade gracefully.
    """
    data_url = file_to_data_url(path)
    messages = [
        {"role": "system", "content": _IMAGE_CAPTION_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Describe this image."},
        ]},
    ]
    return await client.chat_once(
        messages,
        max_tokens=_CAPTION_MAX_TOKENS,
        temperature=0.2,
    )


async def caption_video(
    client: OrchestratorClient,
    path: Path,
    fps: float = 1.0,
    long_edge: int = _DEFAULT_FRAME_LONG_EDGE,
    max_frames: int = _DEFAULT_MAX_FRAMES,
) -> str:
    """Caption a video by sampling frames at `fps` and asking the
    orchestrator to summarise the sequence.

    Frames are resized so the longest edge is `long_edge` pixels before
    being base64-encoded, which keeps token usage bounded on long
    clips. At most `max_frames` frames are sent (covers ~24 seconds at
    1 FPS; clips longer than that get truncated to the start).
    """
    frames = _extract_video_frames(path, fps=fps, long_edge=long_edge, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"no frames could be sampled from {path}")

    user_blocks: List[dict] = []
    for buf in frames:
        user_blocks.append({
            "type": "image_url",
            "image_url": {"url": bytes_to_data_url(buf, mime="image/jpeg")},
        })
    user_blocks.append({
        "type": "text",
        "text": (
            f"These are {len(frames)} frames sampled at ~{fps:g} FPS from a "
            f"short video. Describe what's happening in the clip."
        ),
    })

    messages = [
        {"role": "system", "content": _VIDEO_CAPTION_SYSTEM},
        {"role": "user", "content": user_blocks},
    ]
    return await client.chat_once(
        messages,
        max_tokens=_CAPTION_MAX_TOKENS,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_video_frames(
    path: Path,
    fps: float = 1.0,
    long_edge: int = 512,
    max_frames: int = 24,
) -> List[bytes]:
    """Pull JPEG-encoded frames out of `path`, sampled at roughly `fps`.

    Tries `imageio.v3` (with the pyav plugin) first because that's
    already in our dep tree (`imageio` + `imageio-ffmpeg` + `av`). Falls
    back to walking the file with the default ffmpeg plugin if pyav
    isn't available for some reason. Returns JPEG bytes per frame.
    """
    if long_edge <= 0:
        long_edge = _DEFAULT_FRAME_LONG_EDGE
    if max_frames <= 0:
        max_frames = _DEFAULT_MAX_FRAMES

    import imageio.v3 as iio  # local import keeps server.py import light

    try:
        meta = iio.immeta(str(path), plugin="pyav")
    except Exception:
        meta = iio.immeta(str(path))
    src_fps = float(meta.get("fps") or meta.get("frame_rate") or 30.0)
    if src_fps <= 0:
        src_fps = 30.0
    stride = max(1, int(round(src_fps / max(fps, 0.01))))

    out: List[bytes] = []
    try:
        iterator = iio.imiter(str(path), plugin="pyav")
    except Exception:
        iterator = iio.imiter(str(path))

    for idx, frame in enumerate(iterator):
        if idx % stride != 0:
            continue
        try:
            jpeg_bytes = _resize_and_encode(frame, long_edge=long_edge)
        except Exception:
            continue
        out.append(jpeg_bytes)
        if len(out) >= max_frames:
            break
    return out


def _resize_and_encode(frame, long_edge: int) -> bytes:
    """Resize a numpy-array frame so its longest side <= long_edge, then
    encode as JPEG bytes.
    """
    img = Image.fromarray(frame)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > long_edge:
        scale = long_edge / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
