import logging

from sleuth.logging import get_logger


def test_get_logger_returns_sleuth_namespaced_logger():
    logger = get_logger("engine")
    assert logger.name == "sleuth.engine"


def test_get_logger_different_namespaces():
    a = get_logger("backends.web")
    b = get_logger("memory")
    assert a.name == "sleuth.backends.web"
    assert b.name == "sleuth.memory"


def test_root_logger_has_no_handlers_by_default():
    root = logging.getLogger("sleuth")
    assert root.handlers == []
