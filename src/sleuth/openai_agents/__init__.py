"""OpenAI Agents SDK adapter for Sleuth (extras=[openai-agents]).

Install: pip install agent-sleuth[openai-agents]
"""

from sleuth.openai_agents._tool import make_sleuth_function_tool

__all__ = ["make_sleuth_function_tool"]
