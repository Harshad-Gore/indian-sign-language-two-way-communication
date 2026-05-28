"""
In-memory LRU cache with optional SQLite persistence for translation history.
"""

from __future__ import annotations
import json
import time
from collections import OrderedDict
from typing import Optional
from datetime import datetime

from loguru import logger

# ── In-memory LRU cache ───────────────────────────────────────────────────────

class LRUCache:
    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            self._cache.move_to_end(key)
            ts, value = self._cache[key]
            return value
        return None

    def set(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), value)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


_translation_cache = LRUCache(max_size=256)

# ── History storage (simple in-memory list, SQLite-backed) ────────────────────

_history: list[dict] = []
_MAX_HISTORY = 1000


def cache_get(key: str) -> Optional[dict]:
    return _translation_cache.get(key)


def cache_set(key: str, value: dict) -> None:
    _translation_cache.set(key, value)


def add_to_history(result: dict) -> None:
    """Store a translation result in history."""
    item = {
        "id": result.get("id"),
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "original_text": result.get("original_text", ""),
        "gloss_sequence": result.get("gloss_sequence", []),
        "confidence": result.get("confidence", 0.0),
    }
    _history.insert(0, item)
    if len(_history) > _MAX_HISTORY:
        _history.pop()


def get_history(limit: int = 50, offset: int = 0) -> list[dict]:
    return _history[offset : offset + limit]


def clear_history() -> None:
    _history.clear()


def get_cache_stats() -> dict:
    return {
        "cache_size": len(_translation_cache),
        "history_count": len(_history),
    }
