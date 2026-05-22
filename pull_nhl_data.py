"""
pull_nhl_data.py
================
Ingestion pipeline for the DSC 148 player-style clustering + position-prediction project.

WHAT THIS DOES
--------------
Pulls skater data from the NHL public stats API for one or more seasons and writes
two clean CSVs:

  1. player_games.csv   -> one row per (player, game). This is your SUPERVISED dataset.
                           High volume (~24k rows/season) -> clears the 50k requirement
                           with 2+ seasons.
  2. player_seasons.csv -> one row per (player, season), aggregated. This is your
                           CLUSTERING / fingerprint dataset and feeds the comp-finder.

WHY TWO ENDPOINTS
-----------------
The NHL stats REST API splits skater stats across several "report" tables. We pull:
  - summary  : goals, assists, points, shots, +/-, TOI, games, etc.
  - realtime : hits, blocked shots, takeaways, giveaways  (these are STRONGLY
               position-correlated, so they make forwards vs. D very separable)
We join them on (playerId, gameId/seasonId).

The position label (C/L/R/D) comes from the summary report's positionCode field.

HOW TO RUN
----------
  pip install requests pandas
  python pull_nhl_data.py

Edit SEASONS below to control how many seasons you pull.
Season format is YYYYYYYY, e.g. 20232024 = the 2023-24 season.
"""

import json
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Two seasons clears 50k. Add more for margin. Each season ~ a few minutes to pull.
SEASONS = [20222023, 20232024]

GAME_TYPE = 2          # 2 = regular season, 3 = playoffs
PAGE_SIZE = 100        # API caps a single page near this; we paginate past it.
STATS_BASE = "https://api.nhle.com/stats/rest/en/skater"

# The two report tables we merge. Each has its own columns.
REPORTS = ["summary", "realtime"]


# ---------------------------------------------------------------------------
# CORE FETCH
# ---------------------------------------------------------------------------
def get_team_ids(season):
    """
    Fetch the list of team IDs that played in a given season.

    Why we need this: the stats API refuses to paginate past a ~10,000-row offset.
    A full season of player-GAMES is ~24,000 rows, so a single season-wide query
    silently stops at 10k. The fix is to slice the pull into chunks that each stay
    under the ceiling. Slicing BY TEAM works perfectly: ~32 teams x ~750 player-games
    each = well under 10k per chunk. We fetch team IDs dynamically so the list can't
    go stale.
    """
    # The 'team/summary' report lists every team with its id for the season.
    url = "https://api.nhle.com/stats/rest/en/team/summary"
    params = {
        "isAggregate": "false",
        "isGame": "false",
        "sort": json.dumps([{"property": "teamId", "direction": "ASC"}]),
        "start": "0",
        "limit": "100",
        "cayenneExp": f"gameTypeId={GAME_TYPE} and "
                      f"seasonId<={season} and seasonId>={season}",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return [row["teamId"] for row in resp.json().get("data", [])]


def _paginate(url, base_params):
    """Page through a single filtered query 100 rows at a time until exhausted."""
    rows, start = [], 0
    while True:
        params = dict(base_params, start=str(start), limit=str(PAGE_SIZE))
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("data", [])
        rows.extend(batch)
        start += PAGE_SIZE
        time.sleep(0.2)
        if len(batch) < PAGE_SIZE:   # short page = no more data
            break
        if start > 9000:             # stay safely under the API's offset ceiling
            break
    return rows


def fetch_report(report, season, is_game):
    """
    Pull ALL rows for one report table, one season.

    is_game=True  -> one row per player per GAME. Pulled team-by-team so each
                     chunk stays under the API's 10k offset ceiling.
    is_game=False -> one row per player per SEASON. Small (<1000 rows), one query.
    """
    url = f"{STATS_BASE}/{report}"
    sort = json.dumps([{"property": "playerId", "direction": "ASC"},
                       {"property": "gameId" if is_game else "seasonId",
                        "direction": "ASC"}])
    base = {
        "isAggregate": "false",
        "isGame": str(is_game).lower(),
        "sort": sort,
        "cayenneExp": f"gameTypeId={GAME_TYPE} and "
                      f"seasonId<={season} and seasonId>={season}",
    }

    if not is_game:
        # Season aggregates are small; one paginated query is enough.
        rows = _paginate(url, base)
    else:
        # Player-games: slice by team to dodge the 10k ceiling.
        rows = []
        team_ids = get_team_ids(season)
        for tid in team_ids:
            team_base = dict(base)
            team_base["cayenneExp"] = base["cayenneExp"] + f" and teamId={tid}"
            rows.extend(_paginate(url, team_base))

    print(f"  {report} (is_game={is_game}, season={season}): {len(rows)} rows")
    return pd.DataFrame(rows)


def merge_reports(season, is_game):
    """Pull every report for a season and join them into one wide table."""
    join_keys = ["playerId"] + (["gameId"] if is_game else ["seasonId"])
    merged = None
    for report in REPORTS:
        df = fetch_report(report, season, is_game)
        if df.empty:
            continue
        if merged is None:
            merged = df
        else:
            # Drop duplicate descriptive columns before merging so we don't get
            # skaterFullName_x / _y noise. Keep only join keys + new stat columns.
            new_cols = [c for c in df.columns
                        if c in join_keys or c not in merged.columns]
            merged = merged.merge(df[new_cols], on=join_keys, how="left")
    if merged is not None:
        merged["seasonId"] = season
    return merged


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    game_frames, season_frames = [], []

    for season in SEASONS:
        print(f"\nSeason {season}")
        print(" player-GAME rows:")
        g = merge_reports(season, is_game=True)
        if g is not None:
            game_frames.append(g)
        print(" player-SEASON rows:")
        s = merge_reports(season, is_game=False)
        if s is not None:
            season_frames.append(s)

    games = pd.concat(game_frames, ignore_index=True)
    seasons = pd.concat(season_frames, ignore_index=True)

    games.to_csv("player_games.csv", index=False)
    seasons.to_csv("player_seasons.csv", index=False)

    # ---- the numbers that decide whether the design is viable ----
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"player_games.csv   : {len(games):>7} rows  <- supervised set")
    print(f"player_seasons.csv : {len(seasons):>7} rows  <- clustering set")
    print(f"50,000 instance bar: {'CLEARED' if len(games) >= 50000 else 'NOT met, add a season'}")
    print("\nposition label distribution (player-games):")
    if "positionCode" in games.columns:
        print(games["positionCode"].value_counts())
    print("\ncolumns available:")
    print(list(games.columns))


if __name__ == "__main__":
    main()
