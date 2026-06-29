"""
Orchestrator Node

The orchestrator's only job is routing -- it reads the question and decides
which sub-agents are needed. It does not answer anything itself.

It uses Command(goto=[...]) to fan out to one or more nodes simultaneously.
"""

import json
import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.environ["OPENAI_API_KEY"])

ORCHESTRATOR_SYSTEM = """
You are the routing layer for a fantasy football analysis system.
Given a user question, decide which sub-agents are needed to answer it.

Available sub-agents:
- stats   : historical stats, snap counts, usage rates, season totals (nflverse DB)
- news    : current injury status, beat reporter updates, practice participation
- matchup : opponent defensive ratings, points allowed by position
- vegas   : game totals, implied team totals, player prop lines

Routing rules:
- Start/sit comparison      -> stats, news, matchup, vegas
- Waiver wire pickup        -> stats, news
- Pure historical question  -> stats only
- Injury / availability     -> news only
- Game environment / props  -> vegas only (add stats/news for full player picture)
- Trade evaluation          -> stats, news

Respond ONLY with a JSON object, no markdown, no explanation:
{"agents": ["stats", "news"]}   <- example
Valid agent names: stats, news, matchup, vegas
"""


def orchestrator_node(
    state: AgentState,
) -> Command[Literal["stats_agent", "news_agent", "matchup_agent", "vegas_agent"]]:
    """
    Reads the question, decides which sub-agents to call,
    and fans out to them in parallel via Command(goto=[...]).
    """
    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=state["question"])
    ])

    try:
        routing = json.loads(response.content)
        agents_needed = routing.get("agents", ["stats", "news", "matchup", "vegas"])
    except json.JSONDecodeError:
        agents_needed = ["stats", "news"]

    node_map = {
        "stats": "stats_agent",
        "news": "news_agent",
        "matchup": "matchup_agent",
        "vegas": "vegas_agent",
    }
    target_nodes = [node_map[a] for a in agents_needed if a in node_map]

    return Command(
        update={"agents_needed": agents_needed},
        goto=target_nodes
    )
