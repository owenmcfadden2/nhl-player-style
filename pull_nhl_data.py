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
def fetch_report(report, season, is_game):
    """
    Pull ALL rows for one report table, one season.

    is_game=True  -> one row per player per GAME (high volume, for supervised set)
    is_game=False -> one row per player per SEASON (aggregated, for clustering)

    We paginate with start/limit because a season of player-games is ~24k rows
    and we don't want to rely on limit=-1 silently truncating.
    """
    rows = []
    start = 0
    # Sorting by playerId keeps pagination stable across requests.
    sort = json.dumps([{"property": "playerId", "direction": "ASC"},
                        {"property": "gameId" if is_game else "seasonId",
                         "direction": "ASC"}])

    while True:
        params = {
            "isAggregate": "false",
            "isGame": str(is_game).lower(),
            "sort": sort,
            "start": str(start),
            "limit": str(PAGE_SIZE),
            # cayenneExp is the API's filter language. Restrict to this season + game type.
            "cayenneExp": f"gameTypeId={GAME_TYPE} and "
                          f"seasonId<={season} and seasonId>={season}",
        }
        url = f"{STATS_BASE}/{report}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("data", [])
        rows.extend(batch)
        start += PAGE_SIZE
        time.sleep(0.3)  # be polite to the API
        # Proper end-of-data signal: a page that came back smaller than we asked
        # for (or empty) means there's nothing left. This is what fixes the bug
        # where the pull stopped at one page.
        if len(batch) < PAGE_SIZE:
            break
        # Safety stop so a bug can't loop forever.
        if start > 200000:
            print("  WARNING: hit pagination safety cap")
            break

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
