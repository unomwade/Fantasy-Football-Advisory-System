"""
Synthesizer Node

Collects all sub-agent outputs and produces the final fantasy recommendation.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from core.state import AgentState
from core.orchestrator import llm  # reuse the same ChatOpenAI instance

SYNTHESIZER_SYSTEM = """
You are a fantasy football analyst producing a final start/sit recommendation.

You will receive outputs from one or more specialist agents:
- Stats Agent   : historical stats and usage data from the nflverse database
- News Agent    : current injury status and beat reporter updates
- Matchup Agent : opponent defensive ratings and points allowed by position
- Vegas Agent   : implied team totals, game lines, and player prop lines

Your job:
1. Synthesize all available signals into a coherent fantasy analysis.
2. Identify conflicts (e.g. great historical stats but injury concern) and call them out.
3. Produce a clear recommendation with a confidence level: High / Medium / Low.
4. Keep it concise -- a fantasy manager needs to make a decision, not read a dissertation.

Format your response as:
**Recommendation:** [Start / Sit / Hold / Pick up]
**Confidence:** [High / Medium / Low]
**Key factors:**
- <factor 1>
- <factor 2>
- ...
**Analysis:** [4-6 sentence narrative]
**Caveats:** [anything that could flip this recommendation]
"""


def synthesizer_node(state: AgentState) -> Command[Literal["synthesizer", END]]:
    # Wait until all dispatched agents have reported in
    if state.get("completed_agents", 0) < len(state.get("agents_needed", [])):
        return Command(goto="synthesizer")  # not ready yet, re-enter

    sections = []
    if state.get("stats_result"):
        sections.append(f"=== HISTORICAL STATS (Stats Agent) ===\n{state['stats_result']}")
    if state.get("news_result"):
        sections.append(f"=== INJURY / NEWS (News Agent) ===\n{state['news_result']}")
    if state.get("matchup_result"):
        sections.append(f"=== MATCHUP DATA (Matchup Agent) ===\n{state['matchup_result']}")
    if state.get("vegas_result"):
        sections.append(f"=== VEGAS / ODDS (Vegas Agent) ===\n{state['vegas_result']}")

    if not sections:
        return Command(
            update={"final_answer": "No agent data returned."},
            goto=END
        )

    combined_context = "\n\n".join(sections)
    user_message = f"Original question: {state['question']}\n\nAgent outputs:\n{combined_context}"

    response = llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=user_message)
    ])

    return Command(
        update={"final_answer": response.content},
        goto=END
    )
