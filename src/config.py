"""Shared configuration: league scope, column classification, paths."""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_DIR / "data" / "processed"

YEARS = [2022, 2023, 2024, 2025, 2026]

RAW_PATTERN = "{year}_LoL_esports_match_data_from_OraclesElixir.csv"

# --- League scope -----------------------------------------------------------
# Tier 1 regional leagues, kept under their own label.
TIER1 = [
    "LPL", "LCK", "LEC",
    "LCS", "LTA N", "LTA S", "LTA",  # NA/Americas pre- and post-2025 restructure
    "PCS", "CBLOL", "VCS", "LJL", "TCL", "LLA",
]

# Single-region academy / challenger leagues, kept separate: each has a dense,
# internally connected team pool.
TIER2 = ["LCKC", "LDL", "NACL", "LAS", "CBLOLA", "LCSA", "LJLA"]

# European minor-league ecosystem, pooled into one label. These leagues share a
# team pool and cross-play in EU Masters, so a single rating pool is stabler
# than seven thin fragments.
EU_MINOR_SOURCES = ["LFL", "LFL2", "LVP SL", "PRM", "PRMP", "NLC", "EM"]
EU_MINOR_LABEL = "EU_MINOR"

INTERNATIONAL = ["MSI", "WLDs"]

INCLUDED_LEAGUES = set(TIER1 + TIER2 + EU_MINOR_SOURCES + INTERNATIONAL)

LEAGUE_REMAP = {src: EU_MINOR_LABEL for src in EU_MINOR_SOURCES}

# --- Column classification --------------------------------------------------
# Known before the game starts: identity, scheduling, competitive context.
META_COLS = [
    "gameid", "datacompleteness", "league", "year", "split", "playoffs",
    "date", "game", "patch", "participantid", "side", "position",
    "playername", "playerid", "teamname", "teamid",
]

# Known at end of draft. Available pre-first-blood but AFTER pre-match odds
# close, so these are kept in a separate feature set from true pre-game inputs.
DRAFT_COLS = [
    "firstPick", "champion",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5",
]

TARGET_COL = "result"

# Everything read from the raw CSVs. Every other column in the 165-column OE
# schema is post-game (gamelength, objectives, economy, all *at10/15/20/25
# snapshots) and is never loaded, to make outcome leakage structurally
# impossible rather than merely avoided by convention.
USE_COLS = META_COLS + DRAFT_COLS + [TARGET_COL]
