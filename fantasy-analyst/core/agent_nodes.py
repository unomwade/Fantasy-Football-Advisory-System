"""
Sub-Agent Nodes

Each node is a thin wrapper around the agent functions defined in agents/.
They all share the same uniform structure: run the agent, write the result
to state, route to the synthesizer.

All four agents share an identical calling signature -- ask_X_agent(question)
-- so all four nodes are structurally identical. The Vegas agent in
particular resolves team/game/prop context internally via its own tools;
no special kwargs are passed from here.
"""

from typing import Literal

from langgraph.types import Command

from core.state import AgentState
from agents.stats_agent import ask_stats_agent
from agents.news_agent import ask_news_agent
from agents.matchup_agent import ask_matchup_agent
from agents.vegas_agent import ask_vegas_agent


def stats_agent_node(state: AgentState) -> Command[Literal["synthesizer"]]:
    try:
        result = ask_stats_agent(state["question"])
    except Exception as e:
        result = f"Stats agent failed: {str(e)}"
    return Command(
        update={"stats_result": result, "completed_agents": 1},
        goto="synthesizer"
    )


def news_agent_node(state: AgentState) -> Command[Literal["synthesizer"]]:
    try:
        result = ask_news_agent(state["question"])
    except Exception as e:
        result = f"News agent failed: {str(e)}"
    return Command(
        update={"news_result": result, "completed_agents": 1},
        goto="synthesizer"
    )


def matchup_agent_node(state: AgentState) -> Command[Literal["synthesizer"]]:
    try:
        result = ask_matchup_agent(state["question"])
    except Exception as e:
        result = f"Matchup agent failed: {str(e)}"
    return Command(
        update={"matchup_result": result, "completed_agents": 1},
        goto="synthesizer"
    )


def vegas_agent_node(state: AgentState) -> Command[Literal["synthesizer"]]:
    try:
        result = ask_vegas_agent(state["question"])
    except Exception as e:
        result = f"Vegas agent failed: {str(e)}"
    return Command(
        update={"vegas_result": result, "completed_agents": 1},
        goto="synthesizer"
    )
