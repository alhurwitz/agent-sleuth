"""Logging helpers for the sleuth package.

All internal modules call ``get_logger(__name__.replace("sleuth.", ""))`` which
produces loggers under the ``sleuth`` namespace (e.g. ``sleuth.engine``,
``sleuth.backends.web``).  No handlers are attached here — callers wire
up their own.  The ``SLEUTH_LOG_LEVEL`` env var is intentionally NOT read
at import time; consumers configure log level through normal ``logging``
machinery.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a ``logging.Logger`` under the ``sleuth.<name>`` namespace.

    Args:
        name: Sub-namespace, e.g. ``"engine"``, ``"backends.web"``.

    Returns:
        A logger whose ``.name`` is ``"sleuth.<name>"``.
    """
    return logging.getLogger(f"sleuth.{name}")
