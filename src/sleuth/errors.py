"""Sleuth exception hierarchy.

All public exceptions inherit from SleuthError so callers can catch broadly.
"""


class SleuthError(Exception):
    """Base class for all Sleuth errors."""


class BackendError(SleuthError):
    """A backend search call failed."""


class BackendTimeoutError(BackendError):
    """A backend search call exceeded its timeout."""


class LLMError(SleuthError):
    """An LLM stream call failed or returned an unexpected shape."""


class CacheError(SleuthError):
    """A cache read or write failed."""


class ConfigError(SleuthError):
    """Agent or backend was misconfigured."""
