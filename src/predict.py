"""Score an unplayed match from the artifact written by `train_final.py`.

The feature row is built by `FeatureState.score()` — the same call the training
pass makes for every historical game — against the terminal rating pool stored
in the artifact. Nothing about feature construction is re-implemented here, so a
change to `state.py` moves training and prediction together or not at all.

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
import copy
import json
import re
import sys
import unicodedata
from difflib import get_close_matches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from config import REPO_DIR
from state import ROLES, GameInput
from walkforward import prepare

DEFAULT_ARTIFACT = REPO_DIR / "models" / "lol_logistic_sigmoid_v1.joblib"
STALE_DAYS = 90          # idle longer than this and the rating is a weak prior


# --- name resolution --------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve(value: str, candidates, kind: str, required: bool = True):
    """Match user input to a name the model knows, tolerating case/punctuation.

    Team and champion names in Oracle's Elixir carry apostrophes, periods and
    spacing that nobody types consistently ("Gen.G", "K'Sante"), so an exact
    match is tried first and a normalized one second, before giving up with
    suggestions rather than silently scoring a stranger.
    """
    if value is None:
        return None
    candidates = list(candidates)
    if value in candidates:
        return value

    index = {}
    for c in candidates:
        index.setdefault(_norm(c), c)
    key = _norm(value)
    if key in index:
        return index[key]

    close = get_close_matches(key, list(index), n=5, cutoff=0.6)
    hint = f" Did you mean: {', '.join(index[c] for c in close)}?" if close else ""
    if required:
        raise SystemExit(f"error: unknown {kind} {value!r}.{hint}")
    return None


def resolve_players(names, lookups, warnings, side, team):
    """Turn display names into the playerids the rating pool is keyed by.

    Display names are not unique — a couple of hundred of them belong to more
    than one player — so the team is used to break ties. Getting this wrong
    silently swaps in a stranger's Elo, which is why an unresolvable tie warns
    rather than quietly picking the popular one.
    """
    if names is None:
        return None
    name_to_ids = lookups.get("player_name_to_ids", {})
    last_team = lookups.get("player_last_team", {})
    teams_of = lookups.get("player_teams", {})

    out = []
    for n in names:
        n = (n or "").strip()
        if not n:
            out.append(np.nan)
            continue
        resolved = resolve(n, name_to_ids.keys(), "player", required=False)
        if resolved is None:
            warnings.append(f"{side} player {n!r} not in the data; treated as unknown")
            out.append(np.nan)
            continue

        candidates = name_to_ids[resolved]
        if len(candidates) == 1:
            out.append(candidates[0])
            continue

        on_team = [p for p in candidates if last_team.get(p) == team]
        if not on_team:
            on_team = [p for p in candidates if team in teams_of.get(p, ())]
        if on_team:
            out.append(on_team[0])
            if len(on_team) > 1:
                warnings.append(f"{len(on_team)} players named {resolved!r} have played "
                                f"for {team}; used the most recent")
        else:
            out.append(candidates[0])
            warnings.append(
                f"{len(candidates)} players are named {resolved!r} and none played for "
                f"{team}; used the most recent (last seen with "
                f"{last_team.get(candidates[0], 'unknown team')})")
    return out


def split_list(raw, kind):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 5:
        raise SystemExit(f"error: --{kind} needs exactly 5 comma-separated entries "
                         f"in role order (top,jng,mid,bot,sup); got {len(parts)}")
    return parts


# --- explanation ------------------------------------------------------------

def contributions(model, X_row) -> pd.Series:
    """Per-feature log-odds contribution: coefficient x scaled value.

    Only meaningful for the logistic pipeline, which is the point of having
    kept one: it says which inputs moved this particular number.
    """
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    z = prep.transform(X_row)
    z = z.toarray().ravel() if hasattr(z, "toarray") else np.asarray(z).ravel()
    names = prep.get_feature_names_out()
    return pd.Series(z * clf.coef_.ravel(), index=names)


# --- main -------------------------------------------------------------------

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
    ap.add_argument("--variant", choices=["auto", "pregame", "draft"], default="auto",
                    help="which model to use (default: auto, by whether champs are given)")
    ap.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT, help="artifact path")
    ap.add_argument("--explain", action="store_true", help="show per-feature contributions")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"error: no artifact at {args.model}. Run `python3 train_final.py` first.")

    art = joblib.load(args.model)
    lookups, state = art["lookups"], copy.deepcopy(art["state"])
    warnings: list[str] = []

    blue = resolve(args.blue, lookups["teams"], "team")
    red = resolve(args.red, lookups["teams"], "team")
    if blue == red:
        raise SystemExit("error: --blue and --red are the same team")

    # Game timestamps in the data are UTC-naive; keep the same convention.
    date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz="UTC")
    date = date.tz_localize(None) if date.tzinfo else date
    if state.last_date is not None and date < state.last_date:
        warnings.append(
            f"--date {date} predates the last game in the artifact "
            f"({state.last_date}); ratings include games played after it, and "
            f"idle days go negative for teams that played in between")

    home = lookups.get("team_home_league", {})
    league = args.league or home.get(blue) or home.get(red)
    if league is None:
        raise SystemExit("error: no home league on record for either team; pass --league")
    league = resolve(league, lookups["leagues"], "league", required=False) or league
    if league not in lookups["leagues"]:
        warnings.append(f"league {league!r} unseen in training; its one-hot column is all zeros")

    # Rosters: what the model actually keys player Elo and continuity on.
    blue_roster = resolve_players(split_list(args.blue_roster, "blue-roster"),
                                  lookups, warnings, "blue", blue)
    red_roster = resolve_players(split_list(args.red_roster, "red-roster"),
                                 lookups, warnings, "red", red)
    id_to_name = lookups.get("player_id_to_name", {})
    for side, team, roster in (("blue", blue, blue_roster), ("red", red, red_roster)):
        if roster is None:
            last = state.last_roster(team)
            if last is None:
                warnings.append(f"no roster on record for {team}; player features fall back to defaults")
            else:
                warnings.append(
                    f"{side} roster assumed from {team}'s last game: "
                    + ", ".join(id_to_name.get(p, str(p)) for p in last))
                if side == "blue":
                    blue_roster = list(last)
                else:
                    red_roster = list(last)

    blue_champs = split_list(args.blue_champs, "blue-champs")
    red_champs = split_list(args.red_champs, "red-champs")
    blue_champs = [resolve(c, lookups["champions"], "champion") for c in blue_champs] if blue_champs else None
    red_champs = [resolve(c, lookups["champions"], "champion") for c in red_champs] if red_champs else None

    variant = args.variant
    if variant == "auto":
        variant = "draft" if (blue_champs and red_champs) else "pregame"
    if variant == "draft" and not (blue_champs and red_champs):
        raise SystemExit("error: --variant draft needs both --blue-champs and --red-champs")
    if variant == "pregame" and (blue_champs or red_champs):
        warnings.append("champion picks ignored: the pre-game model does not use them")

    # Staleness: an Elo reverted most of the way to 1500 is barely a prediction.
    for side, team in (("blue", blue), ("red", red)):
        last_played = state.team_last_date.get(team)
        if last_played is None:
            warnings.append(f"{team} has no games in the pool; rated at the 1500 default")
        else:
            idle = (date - last_played).days
            if idle > STALE_DAYS:
                warnings.append(f"{team} last played {idle} days ago ({last_played.date()}); "
                                f"its rating has reverted toward the pool mean")

    gi = GameInput.build(
        date=date, league=league, blue_team=blue, red_team=red,
        blue_roster=blue_roster, red_roster=red_roster,
        blue_champs=blue_champs, red_champs=red_champs,
        playoffs=int(args.playoffs), game=np.nan, gameid="prediction",
    )
    row, _ = state.score(gi)
    feat_row = pd.DataFrame([row])

    v = art["variants"][variant]
    X = prepare(feat_row, for_lgbm=False, numeric=v["numeric_features"])
    p_raw = float(v["model"].predict_proba(X)[0, 1])
    p = float(v["calibrator"].predict_proba(X)[0, 1])

    result = {
        "blue_team": blue, "red_team": red, "league": league,
        "date": str(date.date()), "playoffs": int(args.playoffs),
        "variant": variant,
        "p_blue_win": round(p, 4), "p_red_win": round(1 - p, 4),
        "p_blue_win_uncalibrated": round(p_raw, 4),
        "elo_only_p_blue": round(float(row["elo_expected"]), 4),
        "fair_odds_blue": round(1 / p, 3) if p > 0 else None,
        "fair_odds_red": round(1 / (1 - p), 3) if p < 1 else None,
        "blue_elo": round(float(row["blue_elo"]), 1),
        "red_elo": round(float(row["red_elo"]), 1),
        "blue_form": None if pd.isna(row["blue_form"]) else round(float(row["blue_form"]), 3),
        "red_form": None if pd.isna(row["red_form"]) else round(float(row["red_form"]), 3),
        "h2h": f"{int(row['h2h_blue_wins'])}-{int(row['h2h_red_wins'])}",
        "artifact": {"created_utc": art["created_utc"], "git_commit": art["git_commit"],
                     "trained_through": str(art["training"]["date_max"].date())},
        "warnings": warnings,
    }

    if args.explain:
        contrib = contributions(v["model"], X).sort_values(key=np.abs, ascending=False)
        result["contributions"] = {k: round(float(x), 4) for k, x in contrib.head(12).items()}
        result["intercept"] = round(float(v["model"].named_steps["clf"].intercept_[0]), 4)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"\n{blue}  (blue)  vs  {red}  (red)")
    print(f"{league} · {date.date()}{' · playoffs' if args.playoffs else ''} · {variant} model")
    print("-" * 58)
    print(f"  P({blue} wins) = {p:.1%}      fair odds {1/p:.2f}")
    print(f"  P({red} wins) = {1-p:.1%}      fair odds {1/(1-p):.2f}")
    print(f"\n  uncalibrated {p_raw:.1%} · Elo alone {row['elo_expected']:.1%}")
    print(f"  Elo {row['blue_elo']:.0f} vs {row['red_elo']:.0f} "
          f"({row['elo_diff']:+.0f}) · head-to-head {int(row['h2h_blue_wins'])}-{int(row['h2h_red_wins'])}")
    if not pd.isna(row["blue_form"]) and not pd.isna(row["red_form"]):
        print(f"  form (last 20) {row['blue_form']:.0%} vs {row['red_form']:.0%}")

    if args.explain:
        print("\n  log-odds contributions (toward blue):")
        for k, x in list(result["contributions"].items()):
            print(f"    {k:<34} {x:+.4f}")
        print(f"    {'(intercept)':<34} {result['intercept']:+.4f}")

    if warnings:
        print("\n  warnings:")
        for w in warnings:
            print(f"    - {w}")
    print(f"\n  artifact trained through {result['artifact']['trained_through']} "
          f"(commit {art['git_commit']})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
