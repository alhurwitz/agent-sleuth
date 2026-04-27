"""AutoGen adapter for Sleuth (extras=[autogen]).

Install: pip install agent-sleuth[autogen]
"""

from sleuth.autogen._tool import make_sleuth_autogen_tool, register_sleuth_tool

__all__ = ["make_sleuth_autogen_tool", "register_sleuth_tool"]
