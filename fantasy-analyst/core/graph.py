"""
Graph Builder

Registers all nodes and compiles the LangGraph StateGraph. Import
fantasy_graph from this module wherever the graph needs to be invoked
(e.g. app.py).
"""

from langgraph.graph import StateGraph, START

from core.state import AgentState
from core.orchestrator import orchestrator_node
from core.agent_nodes import (
    stats_agent_node,
    news_agent_node,
    matchup_agent_node,
    vegas_agent_node,
)
from core.synthesizer import synthesizer_node


def build_graph():
    """Build and compile the fantasy agent graph. Call once at app startup."""
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("stats_agent", stats_agent_node)
    builder.add_node("news_agent", news_agent_node)
    builder.add_node("matchup_agent", matchup_agent_node)
    builder.add_node("vegas_agent", vegas_agent_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "orchestrator")

    # Orchestrator -> sub-agents: handled dynamically by Command(goto=[...])
    # inside orchestrator_node. No explicit edges needed.

    return builder.compile()


fantasy_graph = build_graph()


def analyze(question: str) -> dict:
    """
    Run the full fantasy agent graph for a given question.

    Args:
        question: Natural language fantasy football question.

    Returns:
        The full final state dict, including:
          - final_answer:   the synthesized recommendation (str)
          - agents_needed:  which sub-agents were routed to (list[str])
          - stats_result / news_result / matchup_result / vegas_result:
            individual agent outputs, useful for a UI that wants to show
            per-agent detail rather than just the final synthesis.
    """
    initial_state: AgentState = {
        "question": question,
        "agents_needed": [],
        "stats_result": "",
        "news_result": "",
        "matchup_result": "",
        "vegas_result": "",
        "final_answer": "",
        "completed_agents": 0,
    }

    return fantasy_graph.invoke(initial_state)
