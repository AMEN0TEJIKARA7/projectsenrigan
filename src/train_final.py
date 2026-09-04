"""Fit the deployment model on all history and serialize it for scoring.

Walk-forward (`walkforward.py`) answers "how good is this recipe"; it fits and
throws away a model per period and never keeps the rating pool. This script
answers "what do I score tomorrow's game with": it runs the feature pass once
over every game, keeps the terminal `FeatureState`, fits the recipe walk-forward
selected, and writes model + state + name lookups into one artifact.

The train/calibrate split mirrors walk-forward exactly — base model on the first
(1 - CALIB_FRACTION) of history, sigmoid calibrator on the most recent tail —
so the probabilities this artifact emits are the ones the reported metrics
describe. `--refit-full` refits the base model on 100% of history afterwards,
which uses the most recent games but leaves the calibrator fitted against a
slightly different model; it is off by default for that reason.

Three variants are fitted:
  pregame     — usable before champ select, i.e. against pre-match odds
  draft       — pregame + champion features, usable once picks are locked
  draft_only  — champion features alone, for reading a draft in isolation

Usage:
    python3 train_final.py [--out PATH] [--refit-full] [--as-of YYYY-MM-DD]
"""

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from config import PROCESSED_DIR, REPO_DIR, raw_files
from features import build_features
from state import ROLES
from walkforward import (
    CALIB_FRACTION, CALIB_METHOD, CATEGORICAL_FEATURES, DECAY_HALFLIFE_DAYS,
    DRAFT_FEATURES, PREGAME_FEATURES, decay_weights, make_logistic, prepare,
)

ARTIFACT_VERSION = 1
DEFAULT_OUT = REPO_DIR / "models" / "lol_logistic_sigmoid_v1.joblib"

FEATURE_SETS = {
    "pregame": PREGAME_FEATURES,
    "draft": PREGAME_FEATURES + DRAFT_FEATURES,
    # Champion meta alone — no Elo, form, roster or head-to-head. Deliberately
    # weak: it answers "what does the draft say by itself", which is a
    # different question from "who wins", and its walk-forward numbers should
    # be read that way.
    "draft_only": DRAFT_FEATURES,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_player_teams(games: pd.DataFrame) -> dict:
    """Per-player team history, used to disambiguate shared display names."""
    parts = []
    for side in ("blue", "red"):
        for r in ROLES:
            parts.append(
                games[[f"{side}_player_{r}", f"{side}_team", "date"]].rename(
                    columns={f"{side}_player_{r}": "playerid", f"{side}_team": "team"})
            )
    long = pd.concat(parts, ignore_index=True).dropna(subset=["playerid", "team"])
    long = long.sort_values("date")
    last = long.drop_duplicates(subset="playerid", keep="last")
    return {
        "player_last_team": dict(zip(last["playerid"], last["team"])),
        "player_last_date": dict(zip(last["playerid"], last["date"])),
        "player_teams": {pid: sorted(set(t)) for pid, t
                         in long.groupby("playerid")["team"]},
    }


def load_player_names(player_last_date: dict) -> dict:
    """Map display names to playerids, keeping ties instead of hiding them.

    `ingest.py` keeps only `playerid` (stable across renames), but nobody types
    those. The translation is genuinely ambiguous — 190-odd display names in this
    data belong to more than one player — so a name maps to a *list* of
    candidates, ordered so that players in the rating pool and recent activity
    come first, and `predict.py` breaks the tie with the team.
    """
    frames = []
    for path in raw_files():
        frames.append(pd.read_csv(path, usecols=["date", "playername", "playerid"],
                                  low_memory=False).dropna(subset=["playerid", "playername"]))
    if not frames:
        return {"player_name_to_ids": {}, "player_id_to_name": {}}

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    df["playerid"] = df["playerid"].astype(str)
    df["playername"] = df["playername"].astype(str)

    last = df.drop_duplicates(subset="playerid", keep="last")
    id_to_name = dict(zip(last["playerid"], last["playername"]))

    # Every alias a player has used resolves to their id, not just their
    # current handle, so older rosters still parse.
    pairs = df[["playername", "playerid"]].drop_duplicates()
    epoch = pd.Timestamp.min
    name_to_ids: dict = {}
    for name, pid in zip(pairs["playername"], pairs["playerid"]):
        name_to_ids.setdefault(name, []).append(pid)
    for name, ids in name_to_ids.items():
        ids.sort(key=lambda p: (p in player_last_date, player_last_date.get(p, epoch)),
                 reverse=True)

    n_ambiguous = sum(1 for ids in name_to_ids.values() if len(ids) > 1)
    print(f"  {n_ambiguous:,} display names map to more than one player "
          f"(disambiguated by team at predict time)")
    return {"player_name_to_ids": name_to_ids, "player_id_to_name": id_to_name}


def fit_variant(feats: pd.DataFrame, numeric: list, ref_date, refit_full: bool):
    """Fit base model + calibrator using the walk-forward recipe."""
    cut = int(len(feats) * (1 - CALIB_FRACTION))
    calib_df = feats.iloc[cut:]
    # Walk-forward recipe: base model never sees the calibration tail. With
    # --refit-full it does, and the calibrator is then fitted in-sample for the
    # base model — a known, documented compromise, not the default.
    fit_df = feats if refit_full else feats.iloc[:cut]

    model = make_logistic(numeric)
    model.fit(
        prepare(fit_df, for_lgbm=False, numeric=numeric),
        fit_df["blue_win"].to_numpy(),
        clf__sample_weight=decay_weights(fit_df["date"], ref_date, DECAY_HALFLIFE_DAYS),
    )

    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=CALIB_METHOD)
    calibrator.fit(
        prepare(calib_df, for_lgbm=False, numeric=numeric),
        calib_df["blue_win"].to_numpy(),
        sample_weight=decay_weights(calib_df["date"], ref_date, DECAY_HALFLIFE_DAYS),
    )

    return {
        "model": model,
        "calibrator": calibrator,
        "numeric_features": list(numeric),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "n_fit": len(fit_df),
        "n_calib": len(calib_df),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"artifact path (default: {DEFAULT_OUT})")
    ap.add_argument("--refit-full", action="store_true",
                    help="refit the base model on 100%% of history after calibrating")
    ap.add_argument("--as-of", default=None,
                    help="recency-weight reference date (default: last game in the data)")
    args = ap.parse_args(argv)

    games_path = PROCESSED_DIR / "games.parquet"
    if not games_path.exists():
        print(f"Missing {games_path}. Run `python3 ingest.py` first.", file=sys.stderr)
        return 1

    games = pd.read_parquet(games_path)
    print(f"Building features + terminal state over {len(games):,} games...")
    feats, state = build_features(games, return_state=True)
    feats = feats.sort_values("date").reset_index(drop=True)

    ref_date = pd.Timestamp(args.as_of) if args.as_of else feats["date"].max()
    print(f"  history: {feats['date'].min().date()} -> {feats['date'].max().date()}")
    print(f"  recency-weight reference: {pd.Timestamp(ref_date).date()} "
          f"(half-life {DECAY_HALFLIFE_DAYS:.0f}d)")
    print(f"  rating pool: {len(state.team_last_date):,} teams, "
          f"{len(state.player_games):,} players, {len(state.league_elo)} leagues")

    variants = {}
    for name, cols in FEATURE_SETS.items():
        print(f"\nFitting '{name}' ({len(cols)} numeric features)...")
        variants[name] = fit_variant(feats, cols, ref_date, args.refit_full)
        print(f"  base fit on {variants[name]['n_fit']:,} games, "
              f"calibrator on {variants[name]['n_calib']:,}")

    print("\nBuilding name lookups...")
    lookups = build_player_teams(games)
    lookups.update(load_player_names(lookups["player_last_date"]))

    champ_cols = [f"{s}_champ_{r}" for s in ("blue", "red") for r in ROLES]
    champions = sorted({c for col in champ_cols for c in games[col].dropna().astype(str)})
    teams = state.known_teams()

    lookups.update({
        "teams": teams,
        "team_last_played": {t: state.team_last_date[t] for t in teams},
        "team_home_league": dict(state.team_home_league),
        "champions": champions,
        "leagues": sorted(feats["league"].dropna().unique().tolist()),
    })
    print(f"  {len(teams):,} teams, {len(champions)} champions, "
          f"{len(lookups.get('player_name_to_ids', {})):,} player names")

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "variants": variants,
        "default_variant": "draft",
        "state": state,
        "lookups": lookups,
        "training": {
            "n_games": len(feats),
            "date_min": feats["date"].min(),
            "date_max": feats["date"].max(),
            "ref_date": pd.Timestamp(ref_date),
            "decay_halflife_days": DECAY_HALFLIFE_DAYS,
            "calib_fraction": CALIB_FRACTION,
            "calib_method": CALIB_METHOD,
            "refit_full": bool(args.refit_full),
            "blue_win_rate": float(feats["blue_win"].mean()),
        },
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nWrote artifact -> {args.out} ({size_mb:.1f} MB)")
    print("Score a match with: python3 predict.py --blue \"T1\" --red \"Gen.G\"")
    print("\nNote: metrics for this recipe come from `walkforward.py`. This fit is "
          "in-sample by construction and its training log loss is not an estimate "
          "of live performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
