"""
Matchup Agent

Answers: "How good/bad is this player's opponent at stopping their position?"

Works primarily from the play_by_play table (400+ columns), with a curated
column guide injected into the schema response rather than the full DESCRIBE
output, since the full schema would exhaust the context window.
"""

import json
import os

import duckdb
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DB_PATH = os.environ.get("DUCKDB_PATH", "/app/data/nfl_rag.duckdb")

MATCHUP_ALLOWED_TABLES = {
    "play_by_play",
    "player_stats",
    "players",
    "snap_counts",
}

# Key pbp columns the agent needs to know about, grouped by use case.
# This is injected into the system prompt -- it's the view-layer substitute
# until proper SQL views are built.
PBP_COLUMN_GUIDE = """
## play_by_play -- key columns by use case

### Identifiers / filters
- season          INT     -- e.g. 2024
- week            INT     -- 1-18 regular season, 19-22 playoffs
- game_id         VARCHAR -- unique per game
- play_id         BIGINT  -- unique per play
- posteam         VARCHAR -- team with possession (offense)
- defteam         VARCHAR -- defending team
- play_type       VARCHAR -- 'pass', 'run', 'punt', 'field_goal', 'kickoff', 'no_play', etc.
- season_type     VARCHAR -- 'REG' or 'POST'

### Passing / receiver analysis  (filter: play_type = 'pass')
- passer_player_name   VARCHAR  -- QB who threw
- receiver_player_name VARCHAR  -- target receiver
- pass_attempt         INT      -- 1 if pass was attempted
- complete_pass        INT      -- 1 if caught
- passing_yards        DOUBLE   -- yards gained on the pass play (0 if incomplete)
- touchdown            INT      -- 1 if TD scored
- interception         INT      -- 1 if INT thrown
- air_yards            DOUBLE   -- depth of target behind/past LOS
- yards_after_catch    DOUBLE   -- YAC

### Rushing analysis  (filter: play_type = 'run')
- rusher_player_name   VARCHAR  -- ball carrier
- rush_attempt         INT      -- 1 if rush
- rushing_yards        DOUBLE   -- yards gained
- touchdown            INT      -- 1 if TD

### Opponent defensive ratings -- CORRECT query patterns
Always filter season_type = 'REG' unless postseason is specifically requested.

Pass defense vs WR/TE (yards allowed per game to receivers):
  SELECT defteam, season, week, SUM(passing_yards) as pass_yds
  FROM play_by_play
  WHERE play_type = 'pass' AND season = 2024 AND season_type = 'REG'
  GROUP BY defteam, season, week
  -- then aggregate by defteam for per-game average

Rush defense (yards allowed per game to RBs):
  SELECT defteam, season, week, SUM(rushing_yards) as rush_yds
  FROM play_by_play
  WHERE play_type = 'run' AND season = 2024 AND season_type = 'REG'
  GROUP BY defteam, season, week

Points allowed by team:
  Use player_stats table instead -- pbp TD counting is complex.

### Important gotchas
- DO NOT use yards_gained as a passing or rushing total -- it mixes play types.
- DO NOT count touchdowns without filtering play_type -- special teams TDs will pollute the count.
- Penalty plays have play_type = 'no_play' -- exclude unless specifically studying penalties.
- Two-point conversions have play_type = 'pass' or 'run' but touchdown = 0, pts_earned_offense = 2.
"""


def get_matchup_schemas(table_names: list[str]) -> str:
    """Like get_schemas but allows play_by_play and injects the column guide."""
    invalid = [t for t in table_names if t not in MATCHUP_ALLOWED_TABLES]
    if invalid:
        raise ValueError(f"Tables not allowed: {invalid}")

    con = duckdb.connect(DB_PATH)
    try:
        output = []
        for table in table_names:
            if table == "play_by_play":
                output.append(
                    f"### play_by_play (curated column guide)\n{PBP_COLUMN_GUIDE}"
                )
            else:
                df = con.execute(f"DESCRIBE {table}").df()
                schema_md = df.to_markdown(index=False)
                output.append(f"### {table}\n{schema_md}\n")
    finally:
        con.close()

    return "\n".join(output)


def run_matchup_query(sql: str):
    """Read-only query runner; same guardrail as the stats agent's run_query."""
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
    if any(word in sql.upper() for word in forbidden):
        raise ValueError("Dangerous SQL detected.")
    con = duckdb.connect(DB_PATH)
    try:
        result = con.execute(sql).df()
    finally:
        con.close()
    return result


matchup_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_matchup_schemas",
            "description": (
                "Retrieve schema and column guidance for matchup-relevant tables. "
                "For play_by_play, returns a curated column guide instead of the full 400-column schema. "
                "Allowed tables: play_by_play, player_stats, snap_counts, players."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["play_by_play", "player_stats", "snap_counts", "players"]
                        },
                        "description": "Tables to retrieve schemas for."
                    }
                },
                "required": ["table_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_matchup_query",
            "description": "Run a read-only SQL query against matchup-relevant tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A SELECT query."
                    }
                },
                "required": ["sql"]
            }
        }
    }
]


MATCHUP_SYSTEM_PROMPT = """
You are a fantasy football matchup analyst. Your job is to evaluate how favorable or
unfavorable a player's upcoming opponent is for their position.

You have access to:
- get_matchup_schemas: retrieve column information for any allowed table
- run_matchup_query: run SELECT queries against the nflverse database

## Analysis approach by position

QB / WR / TE -- evaluate the opponent's PASS defense:
  - Avg passing yards allowed per game (last 4 weeks and season-to-date)
  - Passing TDs allowed per game
  - Use play_by_play with play_type = 'pass'

RB -- evaluate the opponent's RUSH defense:
  - Avg rushing yards allowed per game (last 4 weeks and season-to-date)
  - Rushing TDs allowed per game
  - Use play_by_play with play_type = 'run'
  - Also check if the opponent allows significant receiving work to RBs (pass plays where
    receiver position is RB -- join via players table if needed)

## Rules
- Always call get_matchup_schemas for play_by_play before writing queries.
- Always filter season_type = 'REG' unless postseason is explicitly requested.
- Compute BOTH recent (last 4 weeks) and full-season averages -- they often diverge.
- Express results as: yards/game, TDs/game, and a qualitative grade (Elite / Good / Average / Weak / Exploitable).
- If the current season has fewer than 4 completed weeks, use all available weeks.
- Never use yards_gained as a position-specific metric -- always use passing_yards or rushing_yards.
- Always include row counts in your SQL (COUNT(DISTINCT game_id)) so the averages are verifiable.
- Return data in a format the synthesizer can use: team name, grade, key stats, and a 1-sentence summary.
"""


def ask_matchup_agent(user_question: str) -> str:
    """
    Evaluate a matchup from a fantasy perspective.

    Example questions:
      'How good is the Dallas defense against WR1s this season?'
      'Is the Chiefs matchup favorable for a RB this week?'
      'How does Jahmyr Gibbs do against the Packers defense historically?'
    """
    messages = [
        {"role": "system", "content": MATCHUP_SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=matchup_tools,
        tool_choice="auto"
    )
    message = response.choices[0].message

    max_iterations = 8  # matchup queries often need 2-3 queries (season + recent + join)
    iteration = 0

    while message.tool_calls and iteration < max_iterations:
        iteration += 1
        messages.append(message)

        for tool_call in message.tool_calls:
            fn = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn == "get_matchup_schemas":
                tool_response = get_matchup_schemas(args["table_names"])

            elif fn == "run_matchup_query":
                try:
                    result_df = run_matchup_query(args["sql"])
                    tool_response = result_df.head(32).to_markdown(index=False)
                    tool_response += f"\n\n(Returned {len(result_df)} rows)"
                except Exception as e:
                    tool_response = f"Query failed: {str(e)}. Fix the SQL and retry."

            else:
                tool_response = f"Unknown tool: {fn}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response
            })

        next_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=matchup_tools,
            tool_choice="auto"
        )
        message = next_response.choices[0].message

    return message.content
