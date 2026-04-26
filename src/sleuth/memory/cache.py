"""Cache protocol and MemoryCache + SqliteCache implementations.

The ``Cache`` Protocol is frozen (conventions §5.3); keep it stable.

Namespaces (spec §8):
    "query"  — ``(query_hash, backend_set, depth) → Result``
    "fetch"  — ``url/file → parsed content``
    "plan"   — ``(query_hash, tree_version) → plan``
    "index"  — per-corpus; Phase 2 adds this namespace
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import aiosqlite


@runtime_checkable
class Cache(Protocol):
    """Async key-value store partitioned by namespace.

    Implementations:
        MemoryCache  (Phase 1) — in-memory, TTL-aware.
        SqliteCache  (Phase 4) — persistent, per-namespace TTL.
    """

    async def get(self, namespace: str, key: str) -> Any | None:
        """Return the stored value, or ``None`` if absent / expired."""
        ...

    async def set(self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        """Store ``value`` under ``(namespace, key)``."""
        ...

    async def delete(self, namespace: str, key: str) -> None:
        """Remove a single entry.  No-op if absent."""
        ...

    async def clear(self, namespace: str | None = None) -> None:
        """Remove all entries in ``namespace``, or everything if ``None``."""
        ...


class MemoryCache:
    """In-memory Cache implementation with optional TTL enforcement.

    Thread-safe for single-threaded async use (no asyncio locks needed because
    dict operations are atomic in CPython).
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = defaultdict(dict)
        # Maps (namespace, key) -> expiry Unix timestamp (or None)
        self._expiry: dict[str, dict[str, float]] = defaultdict(dict)

    async def get(self, namespace: str, key: str) -> Any | None:
        exp = self._expiry[namespace].get(key)
        if exp is not None and time.time() > exp:
            self._store[namespace].pop(key, None)
            self._expiry[namespace].pop(key, None)
            return None
        return self._store[namespace].get(key)

    async def set(self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        self._store[namespace][key] = value
        if ttl_s is not None:
            self._expiry[namespace][key] = time.time() + ttl_s
        else:
            self._expiry[namespace].pop(key, None)

    async def delete(self, namespace: str, key: str) -> None:
        self._store[namespace].pop(key, None)
        self._expiry[namespace].pop(key, None)

    async def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._store.clear()
            self._expiry.clear()
        else:
            self._store[namespace].clear()
            self._expiry[namespace].clear()


# ---------------------------------------------------------------------------
# SqliteCache
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache (
    key      TEXT PRIMARY KEY,
    value    TEXT NOT NULL,
    expires  REAL          -- Unix timestamp; NULL means no expiry
);
"""

_CLEANUP_EXPIRED = "DELETE FROM cache WHERE expires IS NOT NULL AND expires <= ?;"


class SqliteCache:
    """Durable, per-namespace SQLite cache implementing the Cache Protocol.

    Each namespace gets its own ``<base_path>_<namespace>.sqlite`` file so
    that ``clear(namespace)`` is a cheap table-level operation and namespaces
    don't contend with each other.

    Args:
        base_path: Base path prefix for SQLite files.  Namespace and
            ``.sqlite`` suffix are appended automatically.  Defaults to
            ``~/.sleuth/cache/sleuth``, producing files like
            ``~/.sleuth/cache/sleuth_query.sqlite``.
    """

    DEFAULT_TTLS: ClassVar[dict[str, int]] = {
        "query": 600,  # 10 minutes
        "fetch": 86400,  # 24 hours
        "plan": 3600,  # 1 hour
    }

    def __init__(self, base_path: str | Path | None = None) -> None:
        if base_path is None:
            base_path = Path.home() / ".sleuth" / "cache" / "sleuth"
        self._base = Path(base_path)
        self._base.parent.mkdir(parents=True, exist_ok=True)

    def _db_path(self, namespace: str) -> Path:
        return Path(f"{self._base}_{namespace}.sqlite")

    async def _init_db(self, db: aiosqlite.Connection) -> None:
        """Ensure the cache table exists in *db*."""
        await db.execute(_CREATE_TABLE)
        await db.commit()

    async def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value, or ``None`` on miss or expiry."""
        async with aiosqlite.connect(self._db_path(namespace)) as db:
            await self._init_db(db)
            await db.execute(_CLEANUP_EXPIRED, (time.time(),))
            async with db.execute("SELECT value FROM cache WHERE key = ?;", (key,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl_s: int | None = None,
    ) -> None:
        """Store *value* under *key*.  Uses per-namespace default TTL when
        ``ttl_s`` is ``None``."""
        effective_ttl = ttl_s if ttl_s is not None else self.DEFAULT_TTLS.get(namespace)
        expires = time.time() + effective_ttl if effective_ttl is not None else None
        serialised = json.dumps(value)
        async with aiosqlite.connect(self._db_path(namespace)) as db:
            await self._init_db(db)
            await db.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires) VALUES (?, ?, ?);",
                (key, serialised, expires),
            )
            await db.commit()

    async def delete(self, namespace: str, key: str) -> None:
        async with aiosqlite.connect(self._db_path(namespace)) as db:
            await self._init_db(db)
            await db.execute("DELETE FROM cache WHERE key = ?;", (key,))
            await db.commit()

    async def clear(self, namespace: str | None = None) -> None:
        """Delete all entries in *namespace*, or every namespace's file if
        ``namespace`` is ``None``."""
        if namespace is not None:
            async with aiosqlite.connect(self._db_path(namespace)) as db:
                await self._init_db(db)
                await db.execute("DELETE FROM cache;")
                await db.commit()
        else:
            # Delete all sqlite files sharing the base prefix
            parent = self._base.parent
            stem = self._base.name
            for f in parent.glob(f"{stem}_*.sqlite"):
                f.unlink(missing_ok=True)
