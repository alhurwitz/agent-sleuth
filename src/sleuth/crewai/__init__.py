"""CrewAI adapter for Sleuth (extras=[crewai]).

Install: pip install agent-sleuth[crewai]

Note: CrewAI has no native async callback surface. Use the ``on_event``
parameter to receive Sleuth events synchronously during tool execution.
"""

from sleuth.crewai._tool import SleuthCrewAITool

__all__ = ["SleuthCrewAITool"]
