"""
Persistent Wikipedia cache.

WHY
---
Evaluation runs were flapping: identical code, identical claims, different
results. One run passed the typo case and failed the death hoax; the next did
the reverse. The cause was rate limiting — roughly 140 unauthenticated requests
in four minutes — and whichever claim happened to be throttled failed.

An in-memory cache doesn't help because every run starts cold. This one lives
on disk, so:

  * the first run populates it,
  * later runs make almost no network calls,
  * results become deterministic and fast,
  * demos work even if Wikipedia is unreachable at that moment.

Entries expire after TTL_DAYS so facts don't go stale forever.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

TTL_DAYS = 14
_MAX_SENTENCES_PER_PAGE = 120   # keeps the file to a sensible size

_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "wiki_cache.json"

_store: Optional[Dict[str, Any]] = None
_dirty = False


def _load() -> Dict[str, Any]:
    global _store
    if _store is not None:
        return _store
    try:
        raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}

    cutoff = time.time() - TTL_DAYS * 86400
    _store = {k: v for k, v in raw.items()
              if isinstance(v, dict) and v.get("t", 0) > cutoff}
    return _store


def get(key: str):
    entry = _load().get(key)
    return entry.get("v") if entry else None


def put(key: str, value) -> None:
    global _dirty
    if value is None:
        return
    if isinstance(value, list) and len(value) > _MAX_SENTENCES_PER_PAGE:
        value = value[:_MAX_SENTENCES_PER_PAGE]
    _load()[key] = {"t": time.time(), "v": value}
    _dirty = True


def save() -> None:
    """Atomic write — a crash mid-save must not corrupt the cache."""
    global _dirty
    if not _dirty or _store is None:
        return
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(_CACHE_FILE.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_store, fh, ensure_ascii=False)
        os.replace(tmp, _CACHE_FILE)
        _dirty = False
    except Exception:
        pass


def stats() -> Dict[str, Any]:
    store = _load()
    return {
        "entries": len(store),
        "path": str(_CACHE_FILE),
        "size_kb": round(_CACHE_FILE.stat().st_size / 1024, 1)
        if _CACHE_FILE.exists() else 0,
    }


def clear() -> None:
    global _store, _dirty
    _store = {}
    _dirty = True
    save()


# Flush on interpreter exit so a run's fetches are never lost
try:
    import atexit

    atexit.register(save)
except Exception:
    pass
