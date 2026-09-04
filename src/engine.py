"""Reusable scoring engine over a `train_final.py` artifact.

`Predictor` owns everything `predict.py` used to do inline: load the artifact,
resolve human names (teams, champions, players) to the keys the rating pool
uses, build a `GameInput`, and score it through `FeatureState.score()` — the
same call the training pass makes for every historical game. The CLI, the
desktop app, and the skew check all go through this one class so there is a
single place where "a match" turns into "a feature row".
"""

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from state import ROLES, GameInput
from walkforward import prepare

STALE_DAYS = 90          # idle longer than this and the rating is a weak prior


class ResolveError(ValueError):
    """A name the model does not know, with the closest matches it does."""

    def __init__(self, kind: str, value: str, suggestions: list):
        self.kind, self.value, self.suggestions = kind, value, suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"unknown {kind} {value!r}.{hint}")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve(value: Optional[str], candidates, kind: str, required: bool = True):
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
    if required:
        raise ResolveError(kind, value, [index[c] for c in close])
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


def _search(query: str, candidates, limit: int) -> list:
    """Prefix matches first, then substring, preserving candidate order."""
    q = _norm(query)
    if not q:
        return list(candidates)[:limit]
    prefix, inner = [], []
    for c in candidates:
        k = _norm(c)
        if k.startswith(q):
            prefix.append(c)
        elif q in k:
            inner.append(c)
        if len(prefix) >= limit:
            break
    return (prefix + inner)[:limit]


@dataclass
class Prediction:
    p_blue: float
    p_blue_raw: float
    variant: str
    row: dict
    warnings: list = field(default_factory=list)
    contributions: Optional[dict] = None
    intercept: Optional[float] = None


class Predictor:
    """Scores unplayed matches against the artifact's terminal rating pool."""

    def __init__(self, artifact_path):
        self.path = Path(artifact_path)
        self.art = joblib.load(self.path)
        self.lookups = self.art["lookups"]
        self._state = self.art["state"]          # pristine; copied per call
        self.id_to_name = self.lookups.get("player_id_to_name", {})

    # --- metadata -----------------------------------------------------------

    def meta(self) -> dict:
        t = self.art["training"]
        return {
            "trained_through": str(pd.Timestamp(t["date_max"]).date()),
            "created_utc": self.art["created_utc"],
            "git_commit": self.art["git_commit"],
            "n_games": int(t["n_games"]),
            "n_teams": len(self.lookups["teams"]),
            "n_champions": len(self.lookups["champions"]),
            "leagues": list(self.lookups["leagues"]),
            "variants": list(self.art["variants"].keys()),
        }

    # --- search / lookups (for typeahead) -----------------------------------

    def search_teams(self, query: str, limit: int = 8) -> list:
        return _search(query, self.lookups["teams"], limit)

    def search_champions(self, query: str, limit: int = 8) -> list:
        return _search(query, self.lookups["champions"], limit)

    def search_players(self, query: str, team: Optional[str] = None, limit: int = 8) -> list:
        names = list(self.lookups.get("player_name_to_ids", {}).keys())
        if team:
            # Players who last played for this team float to the top.
            last_team = self.lookups.get("player_last_team", {})
            own = {self.id_to_name.get(p) for p, t in last_team.items() if t == team}
            names.sort(key=lambda n: n not in own)
        return _search(query, names, limit)

    def search_leagues(self, query: str, limit: int = 8) -> list:
        return _search(query, self.lookups["leagues"], limit)

    def team_info(self, team: str) -> dict:
        team = resolve(team, self.lookups["teams"], "team")
        st = self._state
        last = st.last_roster(team)
        last_played = st.team_last_date.get(team)
        return {
            "team": team,
            "home_league": self.lookups.get("team_home_league", {}).get(team),
            "last_played": str(last_played.date()) if last_played is not None else None,
            "games": int(st.team_games.get(team, 0)),
            "elo": round(float(st.team_elo.get(team, 1500.0)), 1),
            "roster": [self.id_to_name.get(p, str(p)) if isinstance(p, str) else ""
                       for p in (last or ())],
        }

    # --- scoring ------------------------------------------------------------

    def predict(self, blue: str, red: str, *, league: Optional[str] = None,
                date=None, playoffs: bool = False,
                blue_champs: Optional[list] = None, red_champs: Optional[list] = None,
                blue_roster: Optional[list] = None, red_roster: Optional[list] = None,
                variant: str = "auto", explain: bool = False) -> Prediction:
        lookups = self.lookups
        state = self._state.fork_for_scoring()
        warnings: list = []

        blue = resolve(blue, lookups["teams"], "team")
        red = resolve(red, lookups["teams"], "team")
        if blue == red:
            raise ValueError("blue and red are the same team")

        # Game timestamps in the data are UTC-naive; keep the same convention.
        date = pd.Timestamp(date) if date is not None else pd.Timestamp.now(tz="UTC")
        date = date.tz_localize(None) if date.tzinfo else date
        if state.last_date is not None and date < state.last_date:
            warnings.append(
                f"date {date} predates the last game in the artifact "
                f"({state.last_date}); ratings include games played after it, and "
                f"idle days go negative for teams that played in between")

        home = lookups.get("team_home_league", {})
        league = league or home.get(blue) or home.get(red)
        if league is None:
            raise ValueError("no home league on record for either team; pass a league")
        league = resolve(league, lookups["leagues"], "league", required=False) or league
        if league not in lookups["leagues"]:
            warnings.append(f"league {league!r} unseen in training; its one-hot column is all zeros")

        # Rosters: what the model actually keys player Elo and continuity on.
        blue_roster = resolve_players(_five(blue_roster, "blue roster"), lookups, warnings, "blue", blue)
        red_roster = resolve_players(_five(red_roster, "red roster"), lookups, warnings, "red", red)
        for side, team in (("blue", blue), ("red", red)):
            roster = blue_roster if side == "blue" else red_roster
            if roster is None:
                last = state.last_roster(team)
                if last is None:
                    warnings.append(f"no roster on record for {team}; player features fall back to defaults")
                else:
                    warnings.append(
                        f"{side} roster assumed from {team}'s last game: "
                        + ", ".join(self.id_to_name.get(p, str(p)) for p in last))
                    if side == "blue":
                        blue_roster = list(last)
                    else:
                        red_roster = list(last)

        blue_champs = _five(blue_champs, "blue champs")
        red_champs = _five(red_champs, "red champs")
        blue_champs = [resolve(c, lookups["champions"], "champion") for c in blue_champs] if blue_champs else None
        red_champs = [resolve(c, lookups["champions"], "champion") for c in red_champs] if red_champs else None

        if variant == "auto":
            variant = "draft" if (blue_champs and red_champs) else "pregame"
        if variant not in self.art["variants"]:
            raise ValueError(f"this artifact has no {variant!r} model; retrain with train_final.py")
        if variant in ("draft", "draft_only") and not (blue_champs and red_champs):
            raise ValueError(f"{variant} variant needs both sides' champions")
        if variant == "pregame" and (blue_champs or red_champs):
            warnings.append("champion picks ignored: the pre-game model does not use them")

        # Staleness: an Elo reverted most of the way to 1500 is barely a prediction.
        for team in (blue, red):
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
            playoffs=int(playoffs), game=np.nan, gameid="prediction",
        )
        row, _ = state.score(gi)

        v = self.art["variants"][variant]
        X = prepare(pd.DataFrame([row]), for_lgbm=False, numeric=v["numeric_features"])
        p_raw = float(v["model"].predict_proba(X)[0, 1])
        p = float(v["calibrator"].predict_proba(X)[0, 1])

        pred = Prediction(p_blue=p, p_blue_raw=p_raw, variant=variant, row=row, warnings=warnings)
        if explain:
            contrib = contributions(v["model"], X).sort_values(key=np.abs, ascending=False)
            pred.contributions = {k: float(x) for k, x in contrib.head(12).items()}
            pred.intercept = float(v["model"].named_steps["clf"].intercept_[0])
        return pred

    def predict_dict(self, blue: str, red: str, **kw) -> dict:
        """`predict()` flattened to plain JSON-safe values for CLIs and UIs."""
        pr = self.predict(blue, red, **kw)
        row, p = pr.row, pr.p_blue
        out = {
            "blue_team": row["blue_team"], "red_team": row["red_team"],
            "league": row["league"], "date": str(pd.Timestamp(row["date"]).date()),
            "playoffs": int(row["playoffs"]), "variant": pr.variant,
            "p_blue_win": round(p, 4), "p_red_win": round(1 - p, 4),
            "p_blue_win_uncalibrated": round(pr.p_blue_raw, 4),
            "elo_only_p_blue": round(float(row["elo_expected"]), 4),
            "fair_odds_blue": round(1 / p, 3) if p > 0 else None,
            "fair_odds_red": round(1 / (1 - p), 3) if p < 1 else None,
            "blue_elo": round(float(row["blue_elo"]), 1),
            "red_elo": round(float(row["red_elo"]), 1),
            "blue_form": _opt(row["blue_form"]), "red_form": _opt(row["red_form"]),
            "h2h_blue_wins": int(row["h2h_blue_wins"]), "h2h_red_wins": int(row["h2h_red_wins"]),
            "artifact": {"created_utc": self.art["created_utc"], "git_commit": self.art["git_commit"],
                         "trained_through": self.meta()["trained_through"]},
            "warnings": pr.warnings,
        }
        if pr.contributions is not None:
            out["contributions"] = {k: round(x, 4) for k, x in pr.contributions.items()}
            out["intercept"] = round(pr.intercept, 4)
        return out


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


def _five(values, what: str):
    if values is None:
        return None
    values = list(values)
    if len(values) != 5:
        raise ValueError(f"{what} needs exactly 5 entries in role order {ROLES}; got {len(values)}")
    return values


def _opt(v):
    return None if pd.isna(v) else round(float(v), 3)
