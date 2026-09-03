"""Chronological pre-game feature construction.

Every feature for a game is emitted from state built only from strictly earlier
games, then the game's result updates that state. A single forward pass over
date-sorted games therefore cannot leak outcome information.

Ratings are a global Elo pool rather than per-league pools: international events
(Worlds, MSI) and inter-region play are the only bridges that make cross-league
strength comparable, and separate pools would sever them.

The state itself and the feature-row construction live in `state.py`, so that
`train_final.py` can keep the terminal state and `predict.py` can score an
unplayed match through the exact same code path this pass uses.
"""

import sys

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from state import (  # re-exported for callers that imported them from here
    CHAMP_HALFLIFE_DAYS, CHAMP_PRIOR_WEIGHT, ELO_INIT, ELO_SCALE, FORM_WINDOW,
    K_BASE, K_PROVISIONAL, LEAGUE_ELO_INIT, LEAGUE_K, PLAYER_ELO_INIT,
    PLAYER_K, PROVISIONAL_GAMES, REVERSION_HALFLIFE_DAYS, ROLES, ChampionMeta,
    FeatureState, _expected, _revert,
)


def build_features(games: pd.DataFrame, return_state: bool = False):
    """Forward-pass features for every game, oldest first.

    With `return_state=True`, also returns the `FeatureState` after the last
    game — the ratings and champion meta needed to score a future match.
    """
    games = games.sort_values("date").reset_index(drop=True)
    state = FeatureState()

    rows = []
    for g in games.itertuples(index=False):
        row, ctx = state.score(g)
        rows.append(row)
        state.update(ctx, g.blue_win)

    feats = pd.DataFrame(rows)
    return (feats, state) if return_state else feats


def main() -> int:
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    print(f"Building features for {len(games):,} games...")
    feats = build_features(games)

    out = PROCESSED_DIR / "features.parquet"
    feats.to_parquet(out, index=False)
    print(f"Wrote {len(feats):,} rows, {feats.shape[1]} cols -> {out}")

    # Elo sanity: predictive power of the rating alone, excluding the
    # provisional early period where every team sits at the init value.
    print("\nLeague strength (cross-league Elo, final):")
    print(feats.groupby("league")["blue_league_elo"].last().sort_values(ascending=False).round(1).to_string())

    warm = feats[(feats["blue_games_played"] >= 10) & (feats["red_games_played"] >= 10)]
    acc = ((warm["elo_expected"] > 0.5) == (warm["blue_win"] == 1)).mean()
    ll = -np.mean(
        warm["blue_win"] * np.log(warm["elo_expected"].clip(1e-9, 1 - 1e-9))
        + (1 - warm["blue_win"]) * np.log(1 - warm["elo_expected"].clip(1e-9, 1 - 1e-9))
    )
    print(f"\nElo-only (warm teams, n={len(warm):,}): acc={acc:.4f} logloss={ll:.4f}")
    print(f"Baseline (always blue): acc={warm['blue_win'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
