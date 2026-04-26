"""Tests for MemoryCache and SqliteCache (Phase 1 baseline + Phase 4 additions)."""

import asyncio
from pathlib import Path

import pytest

from sleuth.memory.cache import MemoryCache, SqliteCache

# ---------------------------------------------------------------------------
# MemoryCache (Phase 1 baseline — keep passing)
# ---------------------------------------------------------------------------


async def test_set_and_get():
    c = MemoryCache()
    await c.set("query", "key1", "value1")
    assert await c.get("query", "key1") == "value1"


async def test_get_missing_returns_none():
    c = MemoryCache()
    assert await c.get("query", "missing") is None


async def test_delete():
    c = MemoryCache()
    await c.set("query", "k", "v")
    await c.delete("query", "k")
    assert await c.get("query", "k") is None


async def test_clear_namespace():
    c = MemoryCache()
    await c.set("query", "k1", "v1")
    await c.set("fetch", "k2", "v2")
    await c.clear("query")
    assert await c.get("query", "k1") is None
    assert await c.get("fetch", "k2") == "v2"


async def test_clear_all():
    c = MemoryCache()
    await c.set("query", "k1", "v1")
    await c.set("fetch", "k2", "v2")
    await c.clear()
    assert await c.get("query", "k1") is None
    assert await c.get("fetch", "k2") is None


async def test_namespaces_isolated():
    c = MemoryCache()
    await c.set("query", "same_key", "query_val")
    await c.set("fetch", "same_key", "fetch_val")
    assert await c.get("query", "same_key") == "query_val"
    assert await c.get("fetch", "same_key") == "fetch_val"


async def test_ttl_parameter_accepted():
    """MemoryCache ignores TTL (in-memory has no expiry); it must not error."""
    c = MemoryCache()
    await c.set("query", "k", "v", ttl_s=60)
    assert await c.get("query", "k") == "v"


async def test_memory_cache_ttl_expires():
    """MemoryCache enforces TTL when ttl_s is set."""
    c = MemoryCache()
    await c.set("query", "k1", "value", ttl_s=1)
    await asyncio.sleep(1.1)
    assert await c.get("query", "k1") is None


# ---------------------------------------------------------------------------
# SqliteCache (Phase 4)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_cache"  # SqliteCache appends namespace + .sqlite


async def test_sqlite_set_and_get(db_path: Path) -> None:
    cache = SqliteCache(db_path)
    await cache.set("query", "k1", {"result": "hello"})
    val = await cache.get("query", "k1")
    assert val == {"result": "hello"}


async def test_sqlite_get_missing_returns_none(db_path: Path) -> None:
    cache = SqliteCache(db_path)
    assert await cache.get("query", "missing") is None


async def test_sqlite_ttl_expires(db_path: Path) -> None:
    cache = SqliteCache(db_path)
    await cache.set("query", "k1", "value", ttl_s=1)
    await asyncio.sleep(1.1)
    assert await cache.get("query", "k1") is None


async def test_sqlite_delete(db_path: Path) -> None:
    cache = SqliteCache(db_path)
    await cache.set("query", "k1", "value")
    await cache.delete("query", "k1")
    assert await cache.get("query", "k1") is None


async def test_sqlite_clear_namespace(db_path: Path) -> None:
    cache = SqliteCache(db_path)
    await cache.set("query", "k1", "v1")
    await cache.set("fetch", "k2", "v2")
    await cache.clear("query")
    assert await cache.get("query", "k1") is None
    assert await cache.get("fetch", "k2") == "v2"


async def test_sqlite_clear_all(db_path: Path) -> None:
    cache = SqliteCache(db_path)
    await cache.set("query", "k1", "v1")
    await cache.set("fetch", "k2", "v2")
    await cache.clear()
    assert await cache.get("query", "k1") is None
    assert await cache.get("fetch", "k2") is None


async def test_sqlite_persists_across_instances(db_path: Path) -> None:
    """Data written by one SqliteCache instance is readable by another."""
    cache_a = SqliteCache(db_path)
    await cache_a.set("query", "k1", {"persisted": True})
    # New instance, same path
    cache_b = SqliteCache(db_path)
    val = await cache_b.get("query", "k1")
    assert val == {"persisted": True}


async def test_sqlite_separate_files_per_namespace(db_path: Path) -> None:
    """Each namespace gets its own .sqlite file."""
    import anyio

    cache = SqliteCache(db_path)
    await cache.set("query", "k1", "v1")
    await cache.set("fetch", "k2", "v2")
    query_file = anyio.Path(str(db_path) + "_query.sqlite")
    fetch_file = anyio.Path(str(db_path) + "_fetch.sqlite")
    assert await query_file.exists()
    assert await fetch_file.exists()


async def test_sqlite_json_serialization_roundtrip(db_path: Path) -> None:
    """Complex nested structures survive serialization."""
    payload = {"list": [1, 2, 3], "nested": {"key": "value"}, "num": 3.14}
    cache = SqliteCache(db_path)
    await cache.set("plan", "complex", payload)
    assert await cache.get("plan", "complex") == payload


async def test_sqlite_namespace_ttl_defaults(db_path: Path) -> None:
    """SqliteCache.DEFAULT_TTLS exposes per-namespace TTL constants."""
    assert SqliteCache.DEFAULT_TTLS["query"] == 600  # 10 min
    assert SqliteCache.DEFAULT_TTLS["fetch"] == 86400  # 24 h
    assert SqliteCache.DEFAULT_TTLS["plan"] == 3600  # 1 h


# ---------------------------------------------------------------------------
# CacheHitEvent smoke test
# ---------------------------------------------------------------------------

from sleuth.events import CacheHitEvent  # noqa: E402


async def test_cache_hit_event_on_sqlite_hit(db_path: Path) -> None:
    """Verify SqliteCache satisfies Cache protocol and CacheHitEvent is correctly shaped."""
    from sleuth.memory.cache import Cache  # Protocol

    cache = SqliteCache(db_path)
    # SqliteCache structurally satisfies the Cache protocol
    assert isinstance(cache, Cache)

    await cache.set("query", "test-key", {"text": "cached answer"})
    result = await cache.get("query", "test-key")
    assert result == {"text": "cached answer"}

    event = CacheHitEvent(type="cache_hit", kind="query", key="test-key")
    assert event.type == "cache_hit"
    assert event.kind == "query"
