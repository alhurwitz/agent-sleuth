"""Claude Agent SDK adapter for Sleuth (extras=[claude-agent]).

Install: pip install agent-sleuth[claude-agent]
"""

from sleuth.claude_agent._tool import SleuthClaudeTool

__all__ = ["SleuthClaudeTool"]
