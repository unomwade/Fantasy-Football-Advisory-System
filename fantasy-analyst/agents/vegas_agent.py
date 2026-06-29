"""
Vegas Agent

Answers: "What does the market say about scoring volume and individual
player output for this game?"

Two data layers:
  1. Game lines  -- implied team totals from spreads + game totals
                    (fetched once per TTL window, cached)
  2. Player props -- per-player market lines, fetched on demand per game

The agent is tool-equipped so the LLM resolves team/game context and decides
when props are worth the API credit cost, rather than Python pre-processing
player/team matching ahead of time.
"""

import json
import os
import time

import requests
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
PREFERRED_BOOKMAKER = "draftkings"

# Player prop market keys -- each costs ~4 credits per game on the free tier
PROP_MARKETS = {
    "QB": ["player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_rush_yds"],
    "RB": ["player_rush_yds", "player_rush_attempts", "player_reception_yds", "player_receptions"],
    "WR": ["player_reception_yds", "player_receptions"],
    "TE": ["player_reception_yds", "player_receptions"],
}

MARKET_LABELS = {
    "player_pass_yds": "Passing yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_attempts": "Pass attempts",
    "player_rush_yds": "Rushing yards",
    "player_rush_attempts": "Rush attempts",
    "player_reception_yds": "Receiving yards",
    "player_receptions": "Receptions",
}

# ---------------------------------------------------------------------------
# Staleness config -- mirrors the news agent's TTL pattern
# ---------------------------------------------------------------------------
ODDS_TTL_SECONDS = 6 * 3600  # treat game-lines cache as stale after 6 hours
_cached_implied_totals = None
_odds_last_fetched_at = None


def _odds_cache_is_fresh() -> bool:
    if _odds_last_fetched_at is None:
        return False
    return (time.time() - _odds_last_fetched_at) < ODDS_TTL_SECONDS


# -- Game lines --------------------------------------------------------------

def fetch_nfl_odds() -> list:
    """Fetch current NFL game odds (totals + spreads). Costs 2 API credits."""
    url = f"{ODDS_BASE_URL}/sports/americanfootball_nfl/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals,spreads",
        "oddsFormat": "american",
        "bookmakers": PREFERRED_BOOKMAKER,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def compute_implied_totals(games: list) -> list:
    """
    Derive per-team implied totals from raw game odds.

    Math:
      home_implied = (game_total - home_spread) / 2
      away_implied = game_total - home_implied
    """
    results = []
    for game in games:
        game_total = None
        home_spread = None

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "totals":
                    game_total = market["outcomes"][0]["point"]
                elif market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == game["home_team"]:
                            home_spread = outcome["point"]

        if game_total is None or home_spread is None:
            continue

        home_implied = round((game_total - home_spread) / 2, 1)
        away_implied = round(game_total - home_implied, 1)

        results.append({
            "game_id": game["id"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "game_total": game_total,
            "home_spread": home_spread,
            "home_implied": home_implied,
            "away_implied": away_implied,
            "commence_time": game.get("commence_time", ""),
        })

    return sorted(results, key=lambda x: x["home_implied"] + x["away_implied"], reverse=True)


def get_implied_totals_cached(force: bool = False) -> list:
    """
    Return cached implied totals, refreshing if the cache is stale or missing.

    Odds are TTL-gated (ODDS_TTL_SECONDS). Pass force=True to bypass the gate.
    Returns an empty list if the API key is missing.
    """
    global _cached_implied_totals, _odds_last_fetched_at

    if not ODDS_API_KEY:
        return []

    if not force and _odds_cache_is_fresh() and _cached_implied_totals is not None:
        return _cached_implied_totals

    raw = fetch_nfl_odds()
    _cached_implied_totals = compute_implied_totals(raw)
    _odds_last_fetched_at = time.time()
    return _cached_implied_totals


# -- Player props -------------------------------------------------------------

def fetch_player_props(game_id: str, position: str) -> list:
    """
    Fetch player prop lines for a specific game and position group.

    Credit cost: ~4 credits per market per game. All position markets are
    batched into one call to minimise usage.
    """
    markets = PROP_MARKETS.get(position.upper(), PROP_MARKETS["WR"])
    markets_str = ",".join(markets)

    url = f"{ODDS_BASE_URL}/sports/americanfootball_nfl/events/{game_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": markets_str,
        "oddsFormat": "american",
        "bookmakers": PREFERRED_BOOKMAKER,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return []

    data = r.json()
    props = []

    for bookmaker in data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            market_key = market["key"]
            for outcome in market.get("outcomes", []):
                if outcome.get("description") not in ("Over", "Under"):
                    continue
                existing = next(
                    (p for p in props
                     if p["player"] == outcome["name"] and p["market"] == market_key),
                    None
                )
                if existing is None:
                    existing = {
                        "player": outcome["name"],
                        "market": market_key,
                        "line": outcome.get("point"),
                        "over_odds": None,
                        "under_odds": None,
                    }
                    props.append(existing)
                if outcome["description"] == "Over":
                    existing["over_odds"] = outcome["price"]
                else:
                    existing["under_odds"] = outcome["price"]

    return props


def format_game_lines_table(implied_totals: list) -> str:
    """Render implied totals as a plain-text table for LLM injection."""
    if not implied_totals:
        return "No NFL game lines available."

    header = (
        f"{'Game ID':<36} {'Home Team':<25} {'Away Team':<25} "
        f"{'Total':>6} {'Spread':>7} {'Home Imp':>9} {'Away Imp':>9}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for g in implied_totals:
        rows.append(
            f"{g['game_id']:<36} {g['home_team']:<25} {g['away_team']:<25} "
            f"{g['game_total']:>6.1f} {g['home_spread']:>7.1f} "
            f"{g['home_implied']:>9.1f} {g['away_implied']:>9.1f}"
        )
    return "\n".join(rows)


def format_props_for_llm(props: list, player_name: str) -> str:
    """Filter and format prop lines for a specific player."""
    player_lower = player_name.lower()
    player_props = [
        p for p in props
        if player_lower in p["player"].lower()
        or any(part in p["player"].lower() for part in player_lower.split())
    ]
    if not player_props:
        return f"No props found for {player_name}."

    lines = [f"Player props for {player_props[0]['player']}:"]
    for prop in player_props:
        label = MARKET_LABELS.get(prop["market"], prop["market"])
        lines.append(
            f"  {label}: {prop['line']} | Over {prop['over_odds']} / Under {prop['under_odds']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations callable by the LLM
# ---------------------------------------------------------------------------

def vegas_get_game_lines() -> str:
    """Return this week's implied team totals table (uses cached data)."""
    totals = get_implied_totals_cached()
    if not totals:
        return "No game lines available (offseason or API key missing)."
    return format_game_lines_table(totals)


def vegas_get_player_props(game_id: str, position: str, player_name: str) -> str:
    """
    Fetch and return prop lines for a player in a specific game.
    game_id must be the exact Odds API game_id from the game-lines table.
    position must be QB, RB, WR, or TE.
    """
    if not ODDS_API_KEY:
        return "ODDS_API_KEY not configured -- props unavailable."
    props = fetch_player_props(game_id, position)
    return format_props_for_llm(props, player_name)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

vegas_tools = [
    {
        "type": "function",
        "function": {
            "name": "vegas_get_game_lines",
            "description": (
                "Retrieve this week's NFL game lines as a table of implied team totals, "
                "game totals, and spreads. Always call this first -- it also exposes the "
                "game_id values needed to fetch player props. "
                "Includes: game_id, home_team, away_team, game_total, home_spread, "
                "home_implied, away_implied."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vegas_get_player_props",
            "description": (
                "Fetch player prop lines (yards, TDs, receptions) for a specific player "
                "in a specific game. You MUST call vegas_get_game_lines first to obtain "
                "the correct game_id. Props cost API credits -- only call when a specific "
                "player is being evaluated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "The Odds API game_id from the game-lines table."
                    },
                    "position": {
                        "type": "string",
                        "enum": ["QB", "RB", "WR", "TE"],
                        "description": "Player position -- determines which prop markets are fetched."
                    },
                    "player_name": {
                        "type": "string",
                        "description": "Player's full or partial name used to filter results."
                    }
                },
                "required": ["game_id", "position", "player_name"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

VEGAS_SYSTEM_PROMPT = """
You are a fantasy football Vegas/odds analyst.

You have two tools:
- vegas_get_game_lines  : returns this week's implied team totals table.
                          Always call this first -- it contains the game_id
                          values required to look up player props.
- vegas_get_player_props: fetches player-specific prop lines for a given game.
                          Only call when evaluating a specific player.
                          Props cost API credits -- don't fetch unnecessarily.

## Workflow
1. Always call vegas_get_game_lines first.
2. Identify the relevant team(s) from the game-lines table using full team names.
3. If the question is about a specific player and their position is known or
   inferable, call vegas_get_player_props with the correct game_id.
4. Synthesize both data layers in your response.

## Interpreting implied team totals
- Implied team total = market estimate of how many points that team scores
- Higher implied total -> more offensive volume -> more fantasy opportunity
- QB  : benefits directly from a high implied total
- WR/TE: benefit from high implied total + negative game script (trailing = more passing)
- RB  : benefit from high implied total + POSITIVE game script (leading = more rushing)
  Note: large spread favorites may abandon the run early -- flag this for RBs

## Interpreting player props
- The prop line IS the market's projection for that player's output
- It's more specific than the team total -- weight it heavily when available
- Juice (odds) matters: -140 Over means the market leans toward the Over
- Props and implied totals should corroborate each other -- flag when they diverge

## Grading scale (implied totals)
- >= 28: Elite game environment
- 25-27: Good
- 22-24: Average
- 19-21: Tough
- <  19: Very tough

## Output format
- Always report: implied total, opponent implied, game total, home/away status
- If props available: lead with the prop line and juice, then contextualize with team total
- Include a position-specific interpretation
- Be concise -- 4-6 bullet points per player/team
"""

# ---------------------------------------------------------------------------
# Agent entry point -- uniform signature matching all other agents
# ---------------------------------------------------------------------------

def ask_vegas_agent(user_question: str) -> str:
    """
    Interpret Vegas lines and player props for a fantasy question.

    Signature is uniform with ask_stats_agent / ask_news_agent / ask_matchup_agent:
    accepts only a natural language question; all data fetching is driven by
    the LLM via tools.
    """
    messages = [
        {"role": "system", "content": VEGAS_SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=vegas_tools,
        tool_choice="required",  # always start with vegas_get_game_lines
    )
    message = response.choices[0].message

    max_iterations = 6
    iteration = 0

    while message.tool_calls and iteration < max_iterations:
        iteration += 1
        messages.append(message)

        for tool_call in message.tool_calls:
            fn = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn == "vegas_get_game_lines":
                tool_response = vegas_get_game_lines()

            elif fn == "vegas_get_player_props":
                tool_response = vegas_get_player_props(
                    game_id=args["game_id"],
                    position=args["position"],
                    player_name=args["player_name"],
                )

            else:
                tool_response = f"Unknown tool: {fn}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response,
            })

        next_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=vegas_tools,
            tool_choice="auto",
        )
        message = next_response.choices[0].message

    return message.content
