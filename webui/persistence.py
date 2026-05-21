"""Conversation persistence layer.

On-disk layout (under ``webui/tmp/state/`` by default)::

    state/
      index.json                          # listing for the conversation picker
      conversations/<conv_id>.json        # one file per conversation

Each conversation file is a JSON dump produced by ``Conversation.to_dict``.

Saves are *debounced*: a call to :meth:`PersistenceStore.mark_dirty`
schedules a flush for the configured ``debounce_ms`` later. Subsequent
mark_dirty calls reset the timer so we coalesce bursts of writes (e.g.
a streaming assistant turn that appends many text deltas before the
message is finalised). Atomic ``rename`` is used so a crash never
leaves a half-written file.

Persistence is disabled when the environment variable
``LANCE_NOPERSIST`` is truthy. In that mode :class:`PersistenceStore`
is a no-op, ``mark_dirty`` does nothing, and :meth:`load_all` returns
an empty dict — both for fresh sessions and so tinfoil-hat users can
verify nothing is touching disk.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _log(msg: str) -> None:
    print(f"[persistence {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


class PersistenceStore:
    """Debounced, atomic, on-disk store for Conversation objects.

    Wiring expectation:
      1. Construct once at startup, pass it the ``state.conversations``
         dict and a serializer callable.
      2. Call :meth:`start` to spin up the background flusher thread.
      3. Hook every mutation site (message append, asset add, job state
         change) to call :meth:`mark_dirty(cid)`.
      4. Call :meth:`stop` on shutdown to flush remaining dirty convs.
    """

    def __init__(
        self,
        base_dir: Path,
        enabled: bool = True,
        debounce_ms: int = 500,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.conv_dir = self.base_dir / "conversations"
        self.index_path = self.base_dir / "index.json"
        self.enabled = enabled
        self.debounce_ms = max(50, int(debounce_ms))

        self._dirty: Dict[str, float] = {}  # cid -> monotonic deadline
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._serializer: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None

        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.conv_dir.mkdir(parents=True, exist_ok=True)

    def set_serializer(
        self,
        fn: Callable[[str], Optional[Dict[str, Any]]],
    ) -> None:
        """Provide a ``cid -> dict | None`` callable.

        Returning ``None`` means the conversation was deleted; the file
        will be removed from disk on the next flush.
        """
        self._serializer = fn

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="lance-persistence", daemon=True,
        )
        self._thread.start()
        _log(f"started; base={self.base_dir} debounce={self.debounce_ms}ms")

    def stop(self, timeout: float = 2.0) -> None:
        if not self.enabled or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._flush_all()

    def mark_dirty(self, cid: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._dirty[cid] = time.monotonic() + (self.debounce_ms / 1000.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.1)
            self._flush_due()

    def _flush_due(self) -> None:
        now = time.monotonic()
        ready: list[str] = []
        with self._lock:
            for cid, deadline in list(self._dirty.items()):
                if deadline <= now:
                    ready.append(cid)
                    del self._dirty[cid]
        for cid in ready:
            self._save(cid)
        if ready:
            self._update_index()

    def _flush_all(self) -> None:
        with self._lock:
            cids = list(self._dirty.keys())
            self._dirty.clear()
        for cid in cids:
            self._save(cid)
        if cids:
            self._update_index()

    # ---- I/O ----------------------------------------------------------

    def _save(self, cid: str) -> None:
        if self._serializer is None:
            return
        try:
            data = self._serializer(cid)
        except Exception as e:  # noqa: BLE001
            _log(f"  serializer failed for {cid}: {e}")
            return
        path = self.conv_dir / f"{cid}.json"
        if data is None:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return
        try:
            tmp = self.conv_dir / f"{cid}.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)  # atomic on POSIX + Windows
        except Exception as e:  # noqa: BLE001
            _log(f"  write failed for {cid}: {e}")

    def _update_index(self) -> None:
        entries = []
        try:
            for p in self.conv_dir.glob("*.json"):
                try:
                    with open(p, encoding="utf-8") as f:
                        d = json.load(f)
                    entries.append({
                        "id": d.get("id"),
                        "title": _infer_title(d),
                        "created_at": d.get("created_at"),
                        "updated_at": d.get("updated_at"),
                        "message_count": len(d.get("messages", [])),
                    })
                except Exception:
                    continue
            entries.sort(key=lambda e: e.get("updated_at") or 0, reverse=True)
            tmp = self.index_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"conversations": entries}, f, ensure_ascii=False)
            os.replace(tmp, self.index_path)
        except Exception as e:  # noqa: BLE001
            _log(f"  index update failed: {e}")

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Read every persisted conversation. Returns a ``cid -> dict``
        mapping that the caller is responsible for reconstituting into
        live ``Conversation`` objects."""
        if not self.enabled or not self.conv_dir.exists():
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for p in sorted(self.conv_dir.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                cid = d.get("id")
                if isinstance(cid, str):
                    out[cid] = d
            except Exception as e:  # noqa: BLE001
                _log(f"  load failed for {p.name}: {e}")
        if out:
            _log(f"loaded {len(out)} conversation(s) from disk")
        return out

    def delete(self, cid: str) -> None:
        if not self.enabled:
            return
        try:
            (self.conv_dir / f"{cid}.json").unlink(missing_ok=True)
        except Exception:
            pass
        self._update_index()


def _infer_title(d: Dict[str, Any]) -> str:
    """Cheap conversation title: first non-empty user text bubble, trimmed."""
    for m in d.get("messages", []):
        if m.get("role") != "user":
            continue
        for blk in m.get("display_blocks", []):
            if blk.get("kind") == "text" and (blk.get("text") or "").strip():
                first_line = blk["text"].strip().splitlines()[0]
                return first_line[:60] + ("…" if len(first_line) > 60 else "")
    return "(empty conversation)"


def is_nopersist_env() -> bool:
    """Returns True if persistence should be disabled this session."""
    return _truthy(os.environ.get("LANCE_NOPERSIST"))
