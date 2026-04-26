"""Smoke tests for Phase 0 conftest fixtures (removed after Phase 1 adds real tests)."""
import pytest


def test_tmp_corpus_fixture_creates_directory(tmp_corpus):
    """tmp_corpus must be a writable directory path."""
    assert tmp_corpus.is_dir()


async def test_respx_mock_fixture_is_active(respx_mock):
    """respx_mock must patch httpx transport so unknown hosts raise ConnectError."""
    import httpx
    import respx

    # register one route
    respx_mock.get("https://example.com/ok").respond(200, text="hello")

    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com/ok")
        assert response.status_code == 200
        assert response.text == "hello"
