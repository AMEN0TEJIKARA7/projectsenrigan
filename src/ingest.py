"""Ingest Oracle's Elixir yearly CSVs into one game-per-row table.

Raw OE data is 12 rows per game (10 players + 2 team rows). This collapses each
game to a single row holding both sides' identity and draft, with the blue-side
result as the target. Only pre-game columns are read (see config.USE_COLS).
"""

import sys

import pandas as pd

from config import (
    DRAFT_COLS, INCLUDED_LEAGUES, LEAGUE_REMAP, PROCESSED_DIR, RAW_PATTERN,
    REPO_DIR, USE_COLS, raw_files,
)

ROLES = ["top", "jng", "mid", "bot", "sup"]


def load_raw() -> pd.DataFrame:
    paths = raw_files()
    if not paths:
        raise FileNotFoundError(
            f"no {RAW_PATTERN.format(year='YYYY')} files in {REPO_DIR}")
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=USE_COLS, low_memory=False)
        frames.append(df)
        print(f"  {path.name}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["league"].isin(INCLUDED_LEAGUES)].copy()
    df["league"] = df["league"].replace(LEAGUE_REMAP)
    return df


def drop_incomplete_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only games with the full 12 rows, 2 team rows, and a valid result.

    `datacompleteness` is deliberately not filtered on: it flags missing
    post-game detail, which is irrelevant here since no post-game column is
    loaded. Structural completeness of identity/draft is what matters.
    """
    counts = df.groupby("gameid")["gameid"].transform("size")
    df = df[counts == 12]

    team_rows = df["position"] == "team"
    n_teams = df[team_rows].groupby("gameid")["gameid"].transform("size")
    valid = set(df[team_rows][n_teams == 2]["gameid"])
    df = df[df["gameid"].isin(valid)]

    df = df[df["result"].isin([0, 1])]
    resolved = df[df["position"] == "team"].groupby("gameid")["result"].sum()
    return df[df["gameid"].isin(set(resolved[resolved == 1].index))]


def build_games(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot team + player rows into one row per game, blue/red oriented."""
    teams = df[df["position"] == "team"].copy()
    players = df[df["position"] != "team"].copy()

    keys = ["gameid", "league", "year", "split", "playoffs", "date", "game", "patch"]
    blue = teams[teams["side"] == "Blue"]
    red = teams[teams["side"] == "Red"]

    games = blue[keys + ["teamname", "teamid", "result"]].rename(
        columns={"teamname": "blue_team", "teamid": "blue_teamid", "result": "blue_win"}
    )
    games = games.merge(
        red[["gameid", "teamname", "teamid"]].rename(
            columns={"teamname": "red_team", "teamid": "red_teamid"}
        ),
        on="gameid", how="inner",
    )

    # Per-side draft: team-row bans, and champion picked in each role.
    for side, prefix in (("Blue", "blue"), ("Red", "red")):
        side_teams = teams[teams["side"] == side]
        bans = side_teams[["gameid"] + [f"ban{i}" for i in range(1, 6)]].rename(
            columns={f"ban{i}": f"{prefix}_ban{i}" for i in range(1, 6)}
        )
        games = games.merge(bans, on="gameid", how="left")

        side_players = players[players["side"] == side]
        champs = side_players.pivot_table(
            index="gameid", columns="position", values="champion", aggfunc="first"
        )
        champs = champs.reindex(columns=ROLES)
        champs.columns = [f"{prefix}_champ_{r}" for r in ROLES]
        games = games.merge(champs, on="gameid", how="left")

        ids = side_players.pivot_table(
            index="gameid", columns="position", values="playerid", aggfunc="first"
        )
        ids = ids.reindex(columns=ROLES)
        ids.columns = [f"{prefix}_player_{r}" for r in ROLES]
        games = games.merge(ids, on="gameid", how="left")

    games["date"] = pd.to_datetime(games["date"])
    games["blue_win"] = games["blue_win"].astype(int)
    games["playoffs"] = games["playoffs"].fillna(0).astype(int)
    return games.sort_values("date").reset_index(drop=True)


def main() -> int:
    print("Loading raw OE files (pre-game columns only)...")
    df = load_raw()
    print(f"  total raw rows: {len(df):,}")

    df = filter_leagues(df)
    print(f"After league filter: {len(df):,} rows, {df['gameid'].nunique():,} games")

    df = drop_incomplete_games(df)
    print(f"After completeness filter: {df['gameid'].nunique():,} games")

    games = build_games(df)
    before = len(games)
    games = games.drop_duplicates(subset="gameid")
    if before != len(games):
        print(f"  dropped {before - len(games)} duplicate gameids")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "games.parquet"
    games.to_parquet(out, index=False)

    print(f"\nWrote {len(games):,} games -> {out}")
    print(f"  date range: {games['date'].min()} -> {games['date'].max()}")
    print(f"  blue win rate: {games['blue_win'].mean():.4f}")
    print(f"  leagues: {games['league'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
