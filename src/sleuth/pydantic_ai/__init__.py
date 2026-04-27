"""Pydantic AI adapter for Sleuth (extras=[pydantic-ai]).

Install: pip install agent-sleuth[pydantic-ai]
"""

from sleuth.pydantic_ai._tool import SleuthInput, make_sleuth_tool

__all__ = ["SleuthInput", "make_sleuth_tool"]
