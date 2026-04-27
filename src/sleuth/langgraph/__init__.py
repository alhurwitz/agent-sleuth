"""LangGraph adapter for Sleuth (extras=[langgraph]).

Install: pip install agent-sleuth[langgraph]
"""

from sleuth.langgraph._node import make_sleuth_node

__all__ = ["make_sleuth_node"]
