"""
NFL Data Loader

Builds and refreshes the DuckDB database from nflverse-data releases.

Two distinct operations:
  - Initial build: pulls the last 10 seasons for all four tables. Runs once,
    when the DuckDB file doesn't exist yet (e.g. first container start with
    a fresh volume).
  - Weekly refresh: re-pulls only the CURRENT season for all four tables and
    replaces just that season's rows. Cheap, and keeps in-season data
    current (new weeks, post-game stat corrections) without re-downloading
    nine years of parquet files you already have.

Staleness for the weekly refresh is tracked via a small marker file
(data/.last_nflverse_refresh) rather than an in-memory timestamp, since this
needs to survive container restarts -- unlike the news/odds TTLs, which only
need to survive within a single running session.
"""

import os
import time
from datetime import datetime, timezone

import duckdb
import requests

DB_PATH = os.environ.get("DUCKDB_PATH", "/app/data/nfl_rag.duckdb")
REFRESH_MARKER_PATH = os.environ.get(
    "NFLVERSE_REFRESH_MARKER", "/app/data/.last_nflverse_refresh"
)

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"

# How far back the initial build goes.
INITIAL_BUILD_YEARS = 10

# How long a weekly refresh stays "fresh" before another one is attempted.
# 7 days covers the in-season Tuesday-after-MNF cadence; if the app isn't
# opened for a few weeks, the next refresh just catches up in one shot.
WEEKLY_REFRESH_TTL_SECONDS = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def _current_nfl_season() -> int:
    """
    Best-effort current NFL season year.

    NFL seasons are labeled by the year they START in (the 2025 season
    runs Sept 2025 - Feb 2026). nflverse parquet files follow this
    convention. New season data on nflverse generally doesn't appear until
    early September, so before September we still treat the prior calendar
    year as "current" -- the just-finished season is what's most likely to
    still be getting corrections, and the new season has no games yet.
    """
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 9 else now.year - 1


def _download_parquet(url: str, dest_path: str) -> bool:
    """Download a parquet file. Returns False (without raising) on 404 --
    some season/table combinations don't exist (e.g. snap_counts for a
    season nflverse hasn't published yet)."""
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return True


# ---------------------------------------------------------------------------
# Table-specific fetchers
# ---------------------------------------------------------------------------

def _fetch_season_files(table: str, release_tag: str, seasons: range, tmp_dir: str) -> list[str]:
    """Download one parquet per season for a season-keyed table. Returns
    the list of local paths actually downloaded (skipping any 404s)."""
    paths = []
    for season in seasons:
        url = f"{NFLVERSE_RELEASES}/{release_tag}/{table}_{season}.parquet"
        dest = os.path.join(tmp_dir, f"{table}_{season}.parquet")
        if _download_parquet(url, dest):
            paths.append(dest)
    return paths


def _fetch_players_file(tmp_dir: str) -> str | None:
    """players.parquet is a single rolling file, not season-keyed."""
    url = f"{NFLVERSE_RELEASES}/players/players.parquet"
    dest = os.path.join(tmp_dir, "players.parquet")
    return dest if _download_parquet(url, dest) else None


# ---------------------------------------------------------------------------
# Build / refresh operations
# ---------------------------------------------------------------------------

def build_database(tmp_dir: str = "/tmp/nflverse_build") -> dict:
    """
    Initial build: create all four tables from scratch, pulling the last
    INITIAL_BUILD_YEARS seasons. Only call this when DUCKDB_PATH doesn't
    exist yet -- it drops and recreates tables.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    current_season = _current_nfl_season()
    seasons = range(current_season - INITIAL_BUILD_YEARS + 1, current_season + 1)

    con = duckdb.connect(DB_PATH)
    status = {}

    try:
        pbp_files = _fetch_season_files("play_by_play", "pbp", seasons, tmp_dir)
        if pbp_files:
            con.execute(f"""
                CREATE OR REPLACE TABLE play_by_play AS
                SELECT * FROM read_parquet({pbp_files!r})
            """)
        status["play_by_play"] = len(pbp_files)

        stats_files = _fetch_season_files("player_stats", "player_stats", seasons, tmp_dir)
        if stats_files:
            con.execute(f"""
                CREATE OR REPLACE TABLE player_stats AS
                SELECT * FROM read_parquet({stats_files!r})
            """)
        status["player_stats"] = len(stats_files)

        snap_files = _fetch_season_files("snap_counts", "snap_counts", seasons, tmp_dir)
        if snap_files:
            con.execute(f"""
                CREATE OR REPLACE TABLE snap_counts AS
                SELECT * FROM read_parquet({snap_files!r})
            """)
        status["snap_counts"] = len(snap_files)

        players_file = _fetch_players_file(tmp_dir)
        if players_file:
            con.execute(f"""
                CREATE OR REPLACE TABLE players AS
                SELECT * FROM read_parquet('{players_file}')
            """)
        status["players"] = 1 if players_file else 0

    finally:
        con.close()

    _write_refresh_marker()
    status["seasons_loaded"] = f"{seasons.start}-{seasons.stop - 1}"
    return status


def refresh_current_season(tmp_dir: str = "/tmp/nflverse_refresh") -> dict:
    """
    Weekly refresh: re-pull ONLY the current season for the three
    season-keyed tables (play_by_play, player_stats, snap_counts) and
    replace that season's rows. players.parquet is also re-pulled since
    it's small and changes (trades, new players) happen mid-season.

    Uses DELETE + INSERT rather than dropping the whole table, so prior
    seasons aren't touched and a failed/partial refresh can't wipe
    historical data.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    season = _current_nfl_season()

    con = duckdb.connect(DB_PATH)
    status = {"season": season}

    try:
        for table, release_tag in [
            ("play_by_play", "pbp"),
            ("player_stats", "player_stats"),
            ("snap_counts", "snap_counts"),
        ]:
            url = f"{NFLVERSE_RELEASES}/{release_tag}/{table}_{season}.parquet"
            dest = os.path.join(tmp_dir, f"{table}_{season}.parquet")

            if not _download_parquet(url, dest):
                status[table] = "no data available yet"
                continue

            con.execute(f"DELETE FROM {table} WHERE season = {season}")
            con.execute(f"""
                INSERT INTO {table}
                SELECT * FROM read_parquet('{dest}')
            """)
            status[table] = "refreshed"

        players_file = _fetch_players_file(tmp_dir)
        if players_file:
            con.execute(f"""
                CREATE OR REPLACE TABLE players AS
                SELECT * FROM read_parquet('{players_file}')
            """)
            status["players"] = "refreshed"

    finally:
        con.close()

    _write_refresh_marker()
    return status


# ---------------------------------------------------------------------------
# Staleness tracking (file-based -- survives container restarts)
# ---------------------------------------------------------------------------

def _write_refresh_marker() -> None:
    os.makedirs(os.path.dirname(REFRESH_MARKER_PATH), exist_ok=True)
    with open(REFRESH_MARKER_PATH, "w") as f:
        f.write(str(time.time()))


def _last_refresh_age_seconds() -> float | None:
    if not os.path.exists(REFRESH_MARKER_PATH):
        return None
    with open(REFRESH_MARKER_PATH) as f:
        last = float(f.read().strip())
    return time.time() - last


def ensure_database_current(force: bool = False) -> dict:
    """
    The single entry point session_startup() should call.

    - If DUCKDB_PATH doesn't exist: runs the full initial build.
    - If it exists but the weekly refresh TTL has elapsed (or force=True):
      runs refresh_current_season().
    - Otherwise: no-op.
    """
    if not os.path.exists(DB_PATH):
        result = build_database()
        return {"action": "initial_build", **result}

    age = _last_refresh_age_seconds()
    is_stale = force or age is None or age >= WEEKLY_REFRESH_TTL_SECONDS

    if not is_stale:
        return {
            "action": "skipped",
            "reason": "cache_fresh",
            "age_hours": round(age / 3600, 1),
        }

    result = refresh_current_season()
    return {"action": "weekly_refresh", **result}
