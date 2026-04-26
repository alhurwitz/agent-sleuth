from sleuth.errors import (
    BackendError,
    BackendTimeoutError,
    CacheError,
    ConfigError,
    LLMError,
    SleuthError,
)


def test_hierarchy():
    assert issubclass(BackendError, SleuthError)
    assert issubclass(BackendTimeoutError, BackendError)
    assert issubclass(LLMError, SleuthError)
    assert issubclass(CacheError, SleuthError)
    assert issubclass(ConfigError, SleuthError)


def test_backend_timeout_is_backend_error():
    err = BackendTimeoutError("timed out")
    assert isinstance(err, BackendError)
    assert isinstance(err, SleuthError)


def test_errors_carry_message():
    for cls in (SleuthError, BackendError, BackendTimeoutError, LLMError, CacheError, ConfigError):
        e = cls("msg")
        assert str(e) == "msg"
