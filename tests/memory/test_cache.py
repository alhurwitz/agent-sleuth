from sleuth.memory.cache import MemoryCache


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
