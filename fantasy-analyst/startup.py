"""
Session Startup

Warms all data sources once per session/container lifetime. Both news and
odds caches are TTL-gated -- calls within the TTL window are no-ops unless
force_refresh=True.

Returns a structured status dict rather than printing, so the Streamlit UI
can render the result (e.g. as a sidebar status panel) instead of needing
to capture stdout.
"""

from agents.news_agent import refresh_news
from agents.vegas_agent import get_implied_totals_cached
from data_loader import ensure_database_current


def session_startup(
    refresh_news_data: bool = True,
    prefetch_odds: bool = True,
    refresh_nfl_data: bool = True,
    force_refresh: bool = False,
) -> dict:
    """
    Warm all data sources for the session.

    Args:
        refresh_news_data: Attempt a news refresh (gated by NEWS_TTL_SECONDS).
                            Set False to skip entirely (e.g. you just ran it).
        prefetch_odds:      Fetch and cache implied totals (gated by ODDS_TTL_SECONDS).
                            Set False to skip (saves API credits if called recently).
        refresh_nfl_data:   Build the DuckDB database if it doesn't exist yet, or
                            refresh the current season if the weekly TTL has elapsed.
                            Set False to skip (e.g. you just ran it, or are offline).
        force_refresh:      Bypass TTL gates for news, odds, AND nfl_data.
                            Use after mid-week injury report drops, line movements,
                            or to force-sync the current NFL week's stats.

    Returns:
        {
            "nfl_data": <ensure_database_current() status dict, or {"skipped": True}>,
            "news": <refresh_news() status dict, or {"skipped": True}>,
            "odds": {"games_cached": int} or {"available": False},
        }
    """
    result = {
        "nfl_data": {"skipped": True},
        "news": {"skipped": True},
        "odds": {"available": False},
    }

    if refresh_nfl_data:
        result["nfl_data"] = ensure_database_current(force=force_refresh)

    if refresh_news_data:
        result["news"] = refresh_news(espn_limit=100, force=force_refresh)

    if prefetch_odds:
        totals = get_implied_totals_cached(force=force_refresh)
        if totals:
            result["odds"] = {"available": True, "games_cached": len(totals)}
        else:
            result["odds"] = {"available": False}

    return result
