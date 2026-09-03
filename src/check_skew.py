"""Assert the prediction path reproduces the training path exactly.

Train/serve skew is the failure mode that does not announce itself: the model
loads, the number looks plausible, and it is quietly wrong. This replays real
games through the *prediction* entry point — display names resolved through the
artifact's lookups, a `GameInput` built as `predict.py` builds one — and demands
the resulting feature row equal the row `features.py` produced for that same
game, to the bit.

It also checks that the artifact's fitted pipeline maps both rows to the same
probability, which catches column-order and dtype drift that exact row equality
would not.

Usage:
    python3 check_skew.py [--n 300] [--model PATH]
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, REPO_DIR
from features import build_features
from engine import resolve_players
from state import ROLES, FeatureState, GameInput
from walkforward import prepare

DEFAULT_ARTIFACT = REPO_DIR / "models" / "lol_logistic_sigmoid_v1.joblib"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="most recent games to replay")
    ap.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    args = ap.parse_args(argv)

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    truth = build_features(games)          # what training saw, per game
    # Replay in the exact order the training pass processed games. The date
    # sort is not stable on tied timestamps (31 games share one), so re-sorting
    # here — or sorting a slice — can reorder rating updates and fail this check
    # for the wrong reason. The truth frame is already in processing order.
    games = games.set_index("gameid").loc[truth["gameid"]].reset_index()
    assert (truth["gameid"].to_numpy() == games["gameid"].to_numpy()).all()
    art = joblib.load(args.model)
    lookups = art["lookups"]
    id_to_name = lookups["player_id_to_name"]

    cut = len(games) - args.n
    print(f"Replaying games {cut:,}..{len(games):,} "
          f"({games['date'].iloc[cut].date()} -> {games['date'].iloc[-1].date()})")

    # State as of just before the replay window, advanced in training's row
    # order, then advanced game by game so each replayed game is scored
    # against exactly the state training used.
    state = FeatureState()
    for g in games.iloc[:cut].itertuples(index=False):
        _, ctx = state.score(g)
        state.update(ctx, g.blue_win)

    feature_cols = sorted({c for v in art["variants"].values() for c in v["numeric_features"]})
    worst = {c: 0.0 for c in feature_cols}
    prob_gaps, checked = [], 0
    misresolved: list = []
    roster_warnings: list = []

    for i in range(cut, len(games)):
        g = games.iloc[i]

        # Round-trip playerids through display names and back through the same
        # team-aware resolver `predict.py` uses, so name collisions are under
        # test rather than bypassed.
        def roster(side):
            team = g[f"{side}_team"]
            names, known = [], []
            for r in ROLES:
                pid = g[f"{side}_player_{r}"]
                known.append(pid)
                names.append(id_to_name.get(pid, "") if isinstance(pid, str) else "")
            got = resolve_players(names, lookups, roster_warnings, side, team)
            for want, have in zip(known, got):
                if isinstance(want, str) and want != have:
                    misresolved.append((g["gameid"], team, id_to_name.get(want), want, have))
            return got

        gi = GameInput.build(
            date=g["date"], league=g["league"],
            blue_team=g["blue_team"], red_team=g["red_team"],
            blue_roster=roster("blue"), red_roster=roster("red"),
            blue_champs=[g[f"blue_champ_{r}"] for r in ROLES],
            red_champs=[g[f"red_champ_{r}"] for r in ROLES],
            playoffs=g["playoffs"], patch=g["patch"], game=g["game"], gameid=g["gameid"],
        )
        pred_row, ctx = state.score(gi)
        train_row = truth.iloc[i]
        assert train_row["gameid"] == g["gameid"]

        for c in feature_cols:
            a, b = train_row[c], pred_row[c]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                raise SystemExit(f"NaN mismatch on {c} at game {g['gameid']}: {a!r} vs {b!r}")
            worst[c] = max(worst[c], abs(float(a) - float(b)))

        # Same fitted pipeline, both rows: identical probability or bust.
        for name, v in art["variants"].items():
            cols = v["numeric_features"]
            Xa = prepare(pd.DataFrame([train_row]), for_lgbm=False, numeric=cols)
            Xb = prepare(pd.DataFrame([pred_row]), for_lgbm=False, numeric=cols)
            pa = v["calibrator"].predict_proba(Xa)[0, 1]
            pb = v["calibrator"].predict_proba(Xb)[0, 1]
            prob_gaps.append(abs(pa - pb))

        checked += 1
        state.update(ctx, int(g["blue_win"]))

    max_feat = max(worst.values())
    max_prob = max(prob_gaps)
    print(f"\nReplayed {checked:,} games x {len(feature_cols)} model features")
    print(f"  max feature difference:     {max_feat:.3e}")
    print(f"  max probability difference: {max_prob:.3e}")

    print(f"  players misresolved by name:  {len(misresolved)}")

    offenders = {c: d for c, d in worst.items() if d > 0}
    if offenders:
        print("\n  features that differed:")
        for c, d in sorted(offenders.items(), key=lambda kv: -kv[1]):
            print(f"    {c:<28} {d:.3e}")
    if misresolved:
        print("\n  name collisions the team hint did not settle:")
        for gid, team, nm, want, have in misresolved[:10]:
            print(f"    {team}: {nm!r} -> {have} (wanted {want})")

    ok = max_feat == 0.0 and max_prob < 1e-12 and not misresolved
    print(f"\n{'PASS: prediction path is bit-identical to training' if ok else 'FAIL: train/serve skew detected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
