"""Score an unplayed match from the artifact written by `train_final.py`.

Thin command-line front end over `engine.Predictor`, which builds the feature
row through `FeatureState.score()` — the same call the training pass makes for
every historical game — against the terminal rating pool stored in the
artifact. Nothing about feature construction is re-implemented here.

Usage:
    python3 predict.py --blue "T1" --red "Gen.G"
    python3 predict.py --blue "T1" --red "Gen.G" --playoffs \
        --blue-champs "K'Sante,Vi,Azir,Jinx,Nautilus" \
        --red-champs  "Rumble,Sejuani,Orianna,Varus,Renata Glasc"
    python3 predict.py --blue "T1" --red "Gen.G" --explain --json

Champion picks are optional: with them the draft model runs, without them the
pre-game model does. Rosters default to each team's last-seen starting five.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPO_DIR
from engine import Predictor, ResolveError

DEFAULT_ARTIFACT = REPO_DIR / "models" / "lol_logistic_sigmoid_v1.joblib"


def split_list(raw, kind):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 5:
        raise SystemExit(f"error: --{kind} needs exactly 5 comma-separated entries "
                         f"in role order (top,jng,mid,bot,sup); got {len(parts)}")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blue", required=True, help="blue-side team name")
    ap.add_argument("--red", required=True, help="red-side team name")
    ap.add_argument("--league", default=None,
                    help="league the game is played in (default: blue team's home league)")
    ap.add_argument("--date", default=None,
                    help="match date/time, e.g. 2026-09-05 or 2026-09-05T17:00 "
                         "(default: now, UTC; drives idle days and rating reversion)")
    ap.add_argument("--playoffs", action="store_true", help="mark as a playoff game")
    ap.add_argument("--blue-champs", default=None,
                    help="5 champions, role order top,jng,mid,bot,sup")
    ap.add_argument("--red-champs", default=None, help="same, red side")
    ap.add_argument("--blue-roster", default=None,
                    help="5 player names, role order (default: last-seen five)")
    ap.add_argument("--red-roster", default=None, help="same, red side")
    ap.add_argument("--variant", choices=["auto", "pregame", "draft", "draft_only"], default="auto",
                    help="which model to use (default: auto, by whether champs are given)")
    ap.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT, help="artifact path")
    ap.add_argument("--explain", action="store_true", help="show per-feature contributions")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"error: no artifact at {args.model}. Run `python3 train_final.py` first.")

    try:
        result = Predictor(args.model).predict_dict(
            args.blue, args.red, league=args.league, date=args.date,
            playoffs=args.playoffs,
            blue_champs=split_list(args.blue_champs, "blue-champs"),
            red_champs=split_list(args.red_champs, "red-champs"),
            blue_roster=split_list(args.blue_roster, "blue-roster"),
            red_roster=split_list(args.red_roster, "red-roster"),
            variant=args.variant, explain=args.explain,
        )
    except (ResolveError, ValueError) as e:
        raise SystemExit(f"error: {e}")

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    blue, red, p = result["blue_team"], result["red_team"], result["p_blue_win"]
    print(f"\n{blue}  (blue)  vs  {red}  (red)")
    print(f"{result['league']} · {result['date']}{' · playoffs' if result['playoffs'] else ''}"
          f" · {result['variant']} model")
    print("-" * 58)
    print(f"  P({blue} wins) = {p:.1%}      fair odds {result['fair_odds_blue']:.2f}")
    print(f"  P({red} wins) = {1-p:.1%}      fair odds {result['fair_odds_red']:.2f}")
    print(f"\n  uncalibrated {result['p_blue_win_uncalibrated']:.1%} · "
          f"Elo alone {result['elo_only_p_blue']:.1%}")
    print(f"  Elo {result['blue_elo']:.0f} vs {result['red_elo']:.0f} "
          f"({result['blue_elo'] - result['red_elo']:+.0f}) · head-to-head "
          f"{result['h2h_blue_wins']}-{result['h2h_red_wins']}")
    if result["blue_form"] is not None and result["red_form"] is not None:
        print(f"  form (last 20) {result['blue_form']:.0%} vs {result['red_form']:.0%}")

    if args.explain:
        print("\n  log-odds contributions (toward blue):")
        for k, x in result["contributions"].items():
            print(f"    {k:<34} {x:+.4f}")
        print(f"    {'(intercept)':<34} {result['intercept']:+.4f}")

    if result["warnings"]:
        print("\n  warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")
    print(f"\n  artifact trained through {result['artifact']['trained_through']} "
          f"(commit {result['artifact']['git_commit']})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
