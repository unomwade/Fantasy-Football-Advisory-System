"""
Fantasy Analyst -- Streamlit App

Chat interface for the multi-agent fantasy football analysis system, plus
manual controls for refreshing data sources and inspecting the underlying
ChromaDB news collection.
"""

import os

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing anything that reads os.environ at import time
# (every agents/*.py module does this for OPENAI_API_KEY).
load_dotenv()

from core.graph import analyze
from startup import session_startup
from agents.news_agent import news_collection

st.set_page_config(page_title="Fantasy Analyst", page_icon="🏈", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar -- manual controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Data Controls")

    force_refresh = st.checkbox(
        "Force refresh (bypass cache)",
        value=False,
        help="Bypass the TTL gate for news and odds. Use after injury news or line moves."
    )

    if st.button("Refresh data sources", use_container_width=True):
        with st.spinner("Warming data sources..."):
            result = session_startup(force_refresh=force_refresh)

        nfl_status = result["nfl_data"]
        action = nfl_status.get("action")
        if action == "initial_build":
            st.success(
                f"NFL data: built database for seasons {nfl_status.get('seasons_loaded', '?')}."
            )
        elif action == "weekly_refresh":
            st.success(f"NFL data: refreshed season {nfl_status.get('season', '?')}.")
        elif action == "skipped":
            st.info(f"NFL data: up to date ({nfl_status.get('age_hours', '?')}h since last refresh).")

        news_status = result["news"]
        if news_status.get("refreshed"):
            st.success(
                f"News: fetched {news_status['articles_fetched']} articles, "
                f"upserted {news_status['docs_upserted']} docs "
                f"({news_status['doc_count']} total in collection)."
            )
        else:
            st.info(f"News: cache still fresh ({news_status.get('doc_count', '?')} docs).")

        odds_status = result["odds"]
        if odds_status.get("available"):
            st.success(f"Odds: cached {odds_status['games_cached']} games.")
        else:
            st.warning("Odds: unavailable (offseason or API key missing).")

    st.divider()
    st.subheader("News Collection")

    if st.button("View DB contents", use_container_width=True):
        try:
            count = news_collection.count()
            st.write(f"**{count}** documents in collection.")

            if count > 0:
                peek = news_collection.peek(limit=5)
                for i in range(len(peek["ids"])):
                    meta = peek["metadatas"][i]
                    with st.expander(meta.get("headline", "Untitled")[:80]):
                        st.caption(f"Published: {meta.get('published', 'unknown')}")
                        st.caption(f"Players: {meta.get('players', '-')}")
                        st.caption(f"Teams: {meta.get('teams', '-')}")
                        st.text(peek["documents"][i][:400])
        except Exception as e:
            st.error(f"Could not read collection: {e}")

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

st.title("🏈 Fantasy Analyst")
st.caption("Multi-agent fantasy football research -- stats, news, matchups, and Vegas lines.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a fantasy football question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = analyze(question)

        st.markdown(result["final_answer"])

        agents_used = result.get("agents_needed", [])
        if agents_used:
            with st.expander(f"Agent details ({', '.join(agents_used)})"):
                if result.get("stats_result"):
                    st.markdown("**Stats Agent**")
                    st.markdown(result["stats_result"])
                    st.divider()
                if result.get("news_result"):
                    st.markdown("**News Agent**")
                    st.markdown(result["news_result"])
                    st.divider()
                if result.get("matchup_result"):
                    st.markdown("**Matchup Agent**")
                    st.markdown(result["matchup_result"])
                    st.divider()
                if result.get("vegas_result"):
                    st.markdown("**Vegas Agent**")
                    st.markdown(result["vegas_result"])

    st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})
