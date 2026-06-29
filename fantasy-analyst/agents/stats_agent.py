"""
Stats Agent

Answers: "How good/bad is this player historically?"

Works from the player_stats and snap_counts tables in DuckDB. The agent is
given two tools -- get_schemas (to inspect table structure before querying)
and run_query (to execute SELECT statements) -- and runs an agentic
tool-calling loop until it produces a final natural-language answer.
"""

import json
import os

import duckdb
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALLOWED_TABLES = {
    "player_stats",
    "snap_counts",
}

DB_PATH = os.environ.get("DUCKDB_PATH", "/app/data/nfl_rag.duckdb")

CATEGORICAL_COLUMNS = {"season_type", "position", "position_group"}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_schemas(table_names: list[str]) -> str:
    """
    Retrieve column names/types for the given tables, plus distinct values
    for known categorical columns (season_type, position, position_group)
    so the LLM doesn't have to guess valid filter values (e.g. 'REG' vs
    'Regular').
    """
    invalid = [t for t in table_names if t not in ALLOWED_TABLES]
    if invalid:
        raise ValueError(f"Tables not allowed: {invalid}")

    con = duckdb.connect(DB_PATH)
    try:
        output = []
        for table in table_names:
            df = con.execute(f"DESCRIBE {table}").df()
            schema_md = df.to_markdown(index=False)
            output.append(f"### {table}\n{schema_md}\n")

            existing_cats = CATEGORICAL_COLUMNS & set(df["column_name"].tolist())
            for col in existing_cats:
                vals = con.execute(
                    f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT 10"
                ).fetchall()
                vals_str = ", ".join(str(r[0]) for r in vals)
                output.append(f"  >> {col} values: {vals_str}\n")
    finally:
        con.close()

    return "\n".join(output)


def run_query(sql: str):
    """Execute a read-only SELECT query against the DuckDB database."""
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
    if any(word in sql.upper() for word in forbidden):
        raise ValueError("Dangerous SQL detected.")

    con = duckdb.connect(DB_PATH)
    try:
        result = con.execute(sql).df()
    finally:
        con.close()

    return result


stat_tools = [
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a SQL query against an allowed table.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_schemas",
            "description": (
                "Retrieve schemas (column names and types) for one or more tables. "
                "Allowed tables: player_stats, snap_counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["player_stats", "snap_counts"]
                        },
                        "description": "List of table names."
                    }
                },
                "required": ["table_names"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a fantasy football statistical analyst.

You have access to two tools:
- get_schemas: retrieve column names and types for any allowed table in the database.
- run_query: run a SELECT query against the database.

Allowed tables: player_stats, snap_counts.

Rules:
- Always call get_schemas first to understand the table structure before querying.
- Never assume column names or data without querying.
- Use SQL SELECT statements only.
- Base all conclusions strictly on returned data.
- If insufficient data exists, say so.
- Always call a tool to retrieve data. Never describe a query without executing it.

## Column disambiguation
- player_stats has both `player_name` and `player_display_name`.
  - `player_name` is often an abbreviated form (e.g. "S.Barkley").
  - `player_display_name` is the full name (e.g. "Saquon Barkley").
  - ALWAYS filter on `player_display_name` when matching a full name from
    the user's question, unless the schema/data suggests otherwise.

## Multi-table comparisons
When a question asks to compare or combine data from two tables (e.g. snap
counts and performance stats), use a single SQL JOIN rather than separate
queries. Note the column names differ between tables:
- player_stats uses: player_display_name, season, week, season_type
- snap_counts uses:  player, season, week, game_type

Join example:
SELECT ps.week, ps.carries, ps.rushing_yards, ps.receiving_yards,
       ps.fantasy_points_ppr, sc.offense_snaps, sc.offense_pct
FROM player_stats ps
JOIN snap_counts sc
    ON ps.player_display_name = sc.player
    AND ps.season = sc.season
    AND ps.week = sc.week
WHERE ps.player_display_name = '<player>'
    AND ps.season = <year>
    AND ps.season_type = 'REG';

Your goal is to compare players and recommend optimal fantasy decisions.
"""


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def ask_stats_agent(user_question: str) -> str:
    """
    Answer a fantasy football stats question by running an agentic
    tool-calling loop against the DuckDB database.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=stat_tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    max_iterations = 5
    iteration = 0
    while message.tool_calls and iteration < max_iterations:
        iteration += 1
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "run_query":
                try:
                    result_df = run_query(arguments["sql"])
                    tool_response = result_df.to_markdown(index=False)
                except Exception as e:
                    tool_response = (
                        f"Query failed with exception {str(e)}. "
                        "Please fix the SQL and try again."
                    )

            elif function_name == "get_schemas":
                tool_response = get_schemas(arguments["table_names"])

            else:
                tool_response = f"Unknown tool: {function_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response
            })

        next_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=stat_tools,
            tool_choice="auto"
        )
        message = next_response.choices[0].message

    return message.content
