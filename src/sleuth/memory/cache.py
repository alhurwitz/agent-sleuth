"""Cache protocol and MemoryCache implementation.

Phase 4 will replace ``MemoryCache`` with ``SqliteCache`` (persistent, TTL-aware).
The ``Cache`` Protocol is frozen; keep it stable.

Namespaces (spec §8):
    "query"  — ``(query_hash, backend_set, depth) → Result``
    "fetch"  — ``url/file → parsed content``
    "plan"   — ``(query_hash, tree_version) → plan``
    "index"  — per-corpus; Phase 2 adds this namespace
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol


class Cache(Protocol):
    """Async key-value store partitioned by namespace.

    Implementations:
        MemoryCache  (Phase 1) — in-memory, no TTL enforcement.
        SqliteCache  (Phase 4) — persistent, per-namespace TTL.
    """

    async def get(self, namespace: str, key: str) -> Any | None:
        """Return the stored value, or ``None`` if absent / expired."""
        ...

    async def set(self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        """Store ``value`` under ``(namespace, key)``.

        ``ttl_s`` is accepted by all implementations; MemoryCache ignores it.
        """
        ...

    async def delete(self, namespace: str, key: str) -> None:
        """Remove a single entry.  No-op if absent."""
        ...

    async def clear(self, namespace: str | None = None) -> None:
        """Remove all entries in ``namespace``, or everything if ``None``."""
        ...


class MemoryCache:
    """In-memory Cache implementation — no persistence, no TTL enforcement.

    Thread-safe for single-threaded async use (no asyncio locks needed because
    dict operations are atomic in CPython).  Phase 4 replaces this with
    ``SqliteCache``.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = defaultdict(dict)

    async def get(self, namespace: str, key: str) -> Any | None:
        return self._store[namespace].get(key)

    async def set(self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        self._store[namespace][key] = value

    async def delete(self, namespace: str, key: str) -> None:
        self._store[namespace].pop(key, None)

    async def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._store.clear()
        else:
            self._store[namespace].clear()
