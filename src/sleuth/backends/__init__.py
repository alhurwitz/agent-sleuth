from sleuth.backends.web import BraveBackend as BraveBackend
from sleuth.backends.web import ExaBackend as ExaBackend
from sleuth.backends.web import SerpAPIBackend as SerpAPIBackend
from sleuth.backends.web import TavilyBackend as Tavily  # backwards-compat alias
from sleuth.backends.web import TavilyBackend as TavilyBackend
from sleuth.backends.web import WebBackend as WebBackend

__all__ = [
    "BraveBackend",
    "ExaBackend",
    "SerpAPIBackend",
    "Tavily",
    "TavilyBackend",
    "WebBackend",
]
