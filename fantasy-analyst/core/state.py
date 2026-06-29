"""
Shared State

AgentState is the single object that flows through every node in the graph.
Each sub-agent writes its output into a dedicated field; the synthesizer
reads all of them.
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # The original user question
    question: str

    # Which sub-agents the orchestrator decided to invoke
    # e.g. ["stats", "news"] or ["stats"] or ["news"]
    agents_needed: list[str]

    # Sub-agent outputs -- populated by each node, read by synthesizer
    stats_result: str     # historical stats / usage from nflverse
    news_result: str      # injury status / beat reporter notes
    matchup_result: str   # opponent defensive ratings
    vegas_result: str     # implied totals / game lines / props

    # Final answer assembled by the synthesizer
    final_answer: str

    completed_agents: Annotated[int, operator.add]
