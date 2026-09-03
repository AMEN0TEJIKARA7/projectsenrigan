"""Rating/meta state and the single feature-emission path.

This module holds everything that `features.py` used to keep in local variables
inside its forward-pass loop. It exists so that training and prediction emit a
feature row through *identical* code: `FeatureState.score()` is the only place a
feature row is ever constructed, and `FeatureState.update()` is the only place
state advances. Any change to one automatically applies to the other, which is
what makes train/serve skew structurally impossible rather than a convention.

Invariant preserved from the original implementation: `score()` reads only state
built from strictly earlier games, and `update()` is called only afterwards with
that game's result. A single date-ordered pass therefore cannot leak outcomes.

Pickling note: every default factory here is a module-level function rather than
a lambda, so a populated `FeatureState` can be serialized into a model artifact.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

ROLES = ["top", "jng", "mid", "bot", "sup"]

# Elo settings.
ELO_INIT = 1500.0
ELO_SCALE = 400.0
K_BASE = 24.0
K_PROVISIONAL = 60.0       # higher K until a team has this many games
PROVISIONAL_GAMES = 10
# Inactivity reversion: ratings decay toward the pool mean with a 180-day
# half-life of the deviation, so a team returning after a long layoff carries a
# softened prior rather than a stale one.
REVERSION_HALFLIFE_DAYS = 180.0

PLAYER_ELO_INIT = 1500.0
PLAYER_K = 16.0

FORM_WINDOW = 20           # games retained for short-term form

# Champion meta state. A short half-life is deliberate: champion strength turns
# over with patches far faster than team strength does.
CHAMP_HALFLIFE_DAYS = 90.0
CHAMP_PRIOR_WEIGHT = 30.0  # pseudo-games pulling a champion's rate toward 0.5

# League strength. Updated only on cross-league games (internationals and the
# teams that move between leagues), since intra-league games carry no
# information about how two leagues compare.
LEAGUE_ELO_INIT = 1500.0
LEAGUE_K = 12.0

INTERNATIONAL_LEAGUES = ("MSI", "WLDs")


# --- picklable default factories -------------------------------------------

def _elo_init() -> float:
    return ELO_INIT


def _player_elo_init() -> float:
    return PLAYER_ELO_INIT


def _league_elo_init() -> float:
    return LEAGUE_ELO_INIT


def _zero_int() -> int:
    return 0


def _form_deque() -> deque:
    return deque(maxlen=FORM_WINDOW)


def _pair_counter() -> list:
    return [0, 0]


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def _revert(rating: float, last_date, now, halflife: float = REVERSION_HALFLIFE_DAYS) -> float:
    """Shrink a rating's deviation from the mean based on days idle."""
    if last_date is None:
        return rating
    days = (now - last_date).total_seconds() / 86400.0
    if days <= 0:
        return rating
    return ELO_INIT + (rating - ELO_INIT) * (0.5 ** (days / halflife))


class ChampionMeta:
    """Time-decayed champion win/pick rates with lazy decay on access.

    Counts are decayed only when a champion is touched, so the cost stays O(1)
    per lookup instead of re-decaying every champion on every game.
    """

    def __init__(self, halflife: float = CHAMP_HALFLIFE_DAYS):
        self.halflife = halflife
        self.wins: dict = defaultdict(float)
        self.games: dict = defaultdict(float)
        self.last: dict = {}
        self.total_games = 0.0
        self.total_last = None

    def _decay(self, champ, now) -> None:
        prev = self.last.get(champ)
        if prev is not None:
            f = 0.5 ** ((now - prev).total_seconds() / 86400.0 / self.halflife)
            self.wins[champ] *= f
            self.games[champ] *= f
        self.last[champ] = now

    def _decay_total(self, now) -> None:
        if self.total_last is not None:
            f = 0.5 ** ((now - self.total_last).total_seconds() / 86400.0 / self.halflife)
            self.total_games *= f
        self.total_last = now

    def win_rate(self, champ, now) -> float:
        if not isinstance(champ, str):
            return np.nan
        self._decay(champ, now)
        n = self.games[champ]
        return (self.wins[champ] + 0.5 * CHAMP_PRIOR_WEIGHT) / (n + CHAMP_PRIOR_WEIGHT)

    def volume(self, champ, now) -> float:
        """Decayed games played, i.e. how much evidence the rate rests on."""
        if not isinstance(champ, str):
            return 0.0
        self._decay(champ, now)
        return self.games[champ]

    def presence(self, champ, now) -> float:
        """Share of recent games featuring this champion: a meta-priority proxy."""
        if not isinstance(champ, str):
            return np.nan
        self._decay(champ, now)
        self._decay_total(now)
        return self.games[champ] / self.total_games if self.total_games > 0 else np.nan

    def update(self, champ, won: int, now) -> None:
        if not isinstance(champ, str):
            return
        self._decay(champ, now)
        self.games[champ] += 1.0
        self.wins[champ] += won

    def update_total(self, n_games: float, now) -> None:
        self._decay_total(now)
        self.total_games += n_games


@dataclass
class GameInput:
    """One game's pre-game facts, in the shape `score()` consumes.

    Mirrors the attribute names of a `games.parquet` itertuples row so the
    training pass can hand its rows straight through, while prediction can
    construct one for a match that has not been played.
    """

    date: Any
    league: str
    blue_team: str
    red_team: str
    playoffs: int = 0
    patch: Any = np.nan
    gameid: Any = None
    game: Any = np.nan                    # game number within the series
    blue_win: Any = None                  # unknown at predict time
    blue_player_top: Any = np.nan
    blue_player_jng: Any = np.nan
    blue_player_mid: Any = np.nan
    blue_player_bot: Any = np.nan
    blue_player_sup: Any = np.nan
    red_player_top: Any = np.nan
    red_player_jng: Any = np.nan
    red_player_mid: Any = np.nan
    red_player_bot: Any = np.nan
    red_player_sup: Any = np.nan
    blue_champ_top: Any = np.nan
    blue_champ_jng: Any = np.nan
    blue_champ_mid: Any = np.nan
    blue_champ_bot: Any = np.nan
    blue_champ_sup: Any = np.nan
    red_champ_top: Any = np.nan
    red_champ_jng: Any = np.nan
    red_champ_mid: Any = np.nan
    red_champ_bot: Any = np.nan
    red_champ_sup: Any = np.nan

    @classmethod
    def build(cls, *, date, league, blue_team, red_team, blue_roster=None,
              red_roster=None, blue_champs=None, red_champs=None,
              playoffs=0, patch=np.nan, game=np.nan, gameid=None):
        """Construct from per-side role-ordered lists (top, jng, mid, bot, sup)."""
        def spread(prefix, kind, values):
            values = list(values) if values is not None else [np.nan] * 5
            if len(values) != 5:
                raise ValueError(f"{prefix}_{kind} needs 5 entries in role order {ROLES}")
            return {f"{prefix}_{kind}_{r}": v for r, v in zip(ROLES, values)}

        kwargs = dict(date=date, league=league, blue_team=blue_team,
                      red_team=red_team, playoffs=playoffs, patch=patch,
                      game=game, gameid=gameid)
        kwargs.update(spread("blue", "player", blue_roster))
        kwargs.update(spread("red", "player", red_roster))
        kwargs.update(spread("blue", "champ", blue_champs))
        kwargs.update(spread("red", "champ", red_champs))
        return cls(**kwargs)


@dataclass
class GameContext:
    """Values computed during `score()` that `update()` needs to advance state.

    Passing these forward explicitly, rather than recomputing them, guarantees
    the rating update uses exactly the numbers the feature row was emitted from.
    """

    date: Any
    blue: str
    red: str
    blue_roster: tuple
    red_roster: tuple
    blue_champs: list
    red_champs: list
    b_elo: float
    r_elo: float
    b_played: int
    r_played: int
    b_pelo: float
    r_pelo: float
    b_league: str
    r_league: str
    b_lelo: float
    r_lelo: float
    cross_league: bool
    league: str
    h2h_key: tuple
    series_key: tuple


class FeatureState:
    """Rolling pre-game state: team/player/league Elo, form, h2h, champion meta."""

    def __init__(self):
        self.team_elo: dict = defaultdict(_elo_init)
        self.team_games: dict = defaultdict(_zero_int)
        self.team_last_date: dict = {}
        self.team_form: dict = defaultdict(_form_deque)
        self.team_last_roster: dict = {}
        self.player_elo: dict = defaultdict(_player_elo_init)
        self.player_games: dict = defaultdict(_zero_int)
        self.champ_meta = ChampionMeta()
        # Champion strength conditioned on role, and on the opposing role pick.
        # A champion's value is role-specific, and lane matchups are the
        # mechanism people expect draft to matter through.
        self.role_meta = {r: ChampionMeta() for r in ROLES}
        self.matchup_meta = ChampionMeta()
        self.league_elo: dict = defaultdict(_league_elo_init)
        self.team_home_league: dict = {}
        # Series state keyed by (team pair, date, league): game number within a
        # best-of and the running score, both known before the next game starts.
        self.series: dict = defaultdict(_pair_counter)
        self.h2h: dict = defaultdict(_pair_counter)
        self.last_date = None            # newest game folded into this state

    # --- helpers ------------------------------------------------------------

    def _continuity(self, team, roster):
        """Share of the starting five unchanged from the team's previous game.

        Roster churn degrades the meaning of team Elo, so this is what tells the
        model when to trust it less.
        """
        prev = self.team_last_roster.get(team)
        if prev is None:
            return np.nan
        known = [p for p in roster if isinstance(p, str)]
        if not known:
            return np.nan
        return len(set(roster) & set(prev)) / 5.0

    def _roster_elo(self, roster):
        vals = [self.player_elo[p] for p in roster if isinstance(p, str)]
        return float(np.mean(vals)) if vals else np.nan

    def _roster_exp(self, roster):
        vals = [self.player_games[p] for p in roster if isinstance(p, str)]
        return float(np.mean(vals)) if vals else np.nan

    # --- the single feature-emission path -----------------------------------

    def score(self, g) -> tuple[dict, GameContext]:
        """Emit the pre-game feature row for `g` from current state.

        Reads only; state is advanced by `update()`. `g` is any object carrying
        the `GameInput` attribute names — a `games.parquet` itertuples row, or a
        `GameInput` for an unplayed match.
        """
        date = g.date
        blue, red = g.blue_team, g.red_team
        blue_roster = tuple(getattr(g, f"blue_player_{r}") for r in ROLES)
        red_roster = tuple(getattr(g, f"red_player_{r}") for r in ROLES)

        b_elo = _revert(self.team_elo[blue], self.team_last_date.get(blue), date)
        r_elo = _revert(self.team_elo[red], self.team_last_date.get(red), date)

        b_played, r_played = self.team_games[blue], self.team_games[red]
        b_idle = ((date - self.team_last_date[blue]).total_seconds() / 86400.0
                  if blue in self.team_last_date else np.nan)
        r_idle = ((date - self.team_last_date[red]).total_seconds() / 86400.0
                  if red in self.team_last_date else np.nan)

        b_form = np.mean(self.team_form[blue]) if self.team_form[blue] else np.nan
        r_form = np.mean(self.team_form[red]) if self.team_form[red] else np.nan

        b_cont = self._continuity(blue, blue_roster)
        r_cont = self._continuity(red, red_roster)

        b_pelo, r_pelo = self._roster_elo(blue_roster), self._roster_elo(red_roster)

        # Draft state. Available only after champ select, so these are kept
        # separate from features known before the match starts.
        blue_champs = [getattr(g, f"blue_champ_{r}") for r in ROLES]
        red_champs = [getattr(g, f"red_champ_{r}") for r in ROLES]

        def champ_agg(champs, fn):
            vals = [v for v in (fn(c, date) for c in champs) if not pd.isna(v)]
            return float(np.mean(vals)) if vals else np.nan

        b_cwr = champ_agg(blue_champs, self.champ_meta.win_rate)
        r_cwr = champ_agg(red_champs, self.champ_meta.win_rate)
        b_pres = champ_agg(blue_champs, self.champ_meta.presence)
        r_pres = champ_agg(red_champs, self.champ_meta.presence)

        # Role-conditioned champion win rates.
        b_role_wr, r_role_wr = [], []
        for role, bc, rc in zip(ROLES, blue_champs, red_champs):
            b_role_wr.append(self.role_meta[role].win_rate(bc, date))
            r_role_wr.append(self.role_meta[role].win_rate(rc, date))
        b_rwr = float(np.nanmean(b_role_wr)) if not all(pd.isna(v) for v in b_role_wr) else np.nan
        r_rwr = float(np.nanmean(r_role_wr)) if not all(pd.isna(v) for v in r_role_wr) else np.nan

        # Lane matchup: blue champ vs red champ in the same role, keyed
        # directionally so the rate is "how often does A beat B here".
        mu_rates, mu_vol = [], []
        for role, bc, rc in zip(ROLES, blue_champs, red_champs):
            if isinstance(bc, str) and isinstance(rc, str):
                mkey = f"{role}|{bc}|{rc}"
                mu_rates.append(self.matchup_meta.win_rate(mkey, date))
                mu_vol.append(self.matchup_meta.volume(mkey, date))
        b_mu = float(np.mean(mu_rates)) if mu_rates else np.nan
        b_mu_vol = float(np.mean(mu_vol)) if mu_vol else np.nan

        key = (blue, red) if blue <= red else (red, blue)
        rec = self.h2h[key]
        b_h2h, r_h2h = (rec[0], rec[1]) if blue <= red else (rec[1], rec[0])

        # League strength gap, using each team's home league rather than the
        # league this game is played in, so internationals compare regions.
        b_league = self.team_home_league.get(blue, g.league)
        r_league = self.team_home_league.get(red, g.league)
        b_lelo, r_lelo = self.league_elo[b_league], self.league_elo[r_league]
        cross_league = b_league != r_league

        skey = (key, date.date(), g.league)
        s_rec = self.series[skey]
        s_blue, s_red = (s_rec[0], s_rec[1]) if blue <= red else (s_rec[1], s_rec[0])

        row = {
            "gameid": g.gameid,
            "date": date,
            "league": g.league,
            "playoffs": g.playoffs,
            "patch": g.patch,
            "blue_team": blue,
            "red_team": red,
            "blue_win": g.blue_win,
            "blue_elo": b_elo,
            "red_elo": r_elo,
            "elo_diff": b_elo - r_elo,
            "elo_expected": _expected(b_elo, r_elo),
            "blue_games_played": b_played,
            "red_games_played": r_played,
            "games_played_diff": b_played - r_played,
            "blue_days_idle": b_idle,
            "red_days_idle": r_idle,
            "blue_form": b_form,
            "red_form": r_form,
            "form_diff": (b_form - r_form) if not (np.isnan(b_form) or np.isnan(r_form)) else np.nan,
            "blue_roster_continuity": b_cont,
            "red_roster_continuity": r_cont,
            "blue_player_elo": b_pelo,
            "red_player_elo": r_pelo,
            "player_elo_diff": (b_pelo - r_pelo) if not (np.isnan(b_pelo) or np.isnan(r_pelo)) else np.nan,
            "blue_roster_exp": self._roster_exp(blue_roster),
            "red_roster_exp": self._roster_exp(red_roster),
            "h2h_blue_wins": b_h2h,
            "h2h_red_wins": r_h2h,
            "h2h_games": b_h2h + r_h2h,
            "blue_champ_wr": b_cwr,
            "red_champ_wr": r_cwr,
            "champ_wr_diff": (b_cwr - r_cwr) if not (pd.isna(b_cwr) or pd.isna(r_cwr)) else np.nan,
            "blue_champ_presence": b_pres,
            "red_champ_presence": r_pres,
            "champ_presence_diff": (b_pres - r_pres) if not (pd.isna(b_pres) or pd.isna(r_pres)) else np.nan,
            "blue_league_elo": b_lelo,
            "red_league_elo": r_lelo,
            "league_elo_diff": b_lelo - r_lelo,
            "cross_league": int(cross_league),
            "series_game": g.game,
            "series_blue_wins": s_blue,
            "series_red_wins": s_red,
            "series_score_diff": s_blue - s_red,
            "blue_role_champ_wr": b_rwr,
            "red_role_champ_wr": r_rwr,
            "role_champ_wr_diff": (b_rwr - r_rwr) if not (pd.isna(b_rwr) or pd.isna(r_rwr)) else np.nan,
            "matchup_wr": b_mu,
            "matchup_volume": b_mu_vol,
        }

        ctx = GameContext(
            date=date, blue=blue, red=red,
            blue_roster=blue_roster, red_roster=red_roster,
            blue_champs=blue_champs, red_champs=red_champs,
            b_elo=b_elo, r_elo=r_elo, b_played=b_played, r_played=r_played,
            b_pelo=b_pelo, r_pelo=r_pelo,
            b_league=b_league, r_league=r_league,
            b_lelo=b_lelo, r_lelo=r_lelo, cross_league=cross_league,
            league=g.league, h2h_key=key, series_key=skey,
        )
        return row, ctx

    # --- state advance ------------------------------------------------------

    def update(self, ctx: GameContext, result: int) -> None:
        """Fold one finished game's outcome into state. Call only after `score()`."""
        date, blue, red = ctx.date, ctx.blue, ctx.red
        result = int(result)

        exp_b = _expected(ctx.b_elo, ctx.r_elo)
        k_b = K_PROVISIONAL if ctx.b_played < PROVISIONAL_GAMES else K_BASE
        k_r = K_PROVISIONAL if ctx.r_played < PROVISIONAL_GAMES else K_BASE
        self.team_elo[blue] = ctx.b_elo + k_b * (result - exp_b)
        self.team_elo[red] = ctx.r_elo + k_r * ((1 - result) - (1 - exp_b))

        self.team_games[blue] += 1
        self.team_games[red] += 1
        self.team_last_date[blue] = date
        self.team_last_date[red] = date
        self.team_form[blue].append(result)
        self.team_form[red].append(1 - result)
        if any(isinstance(p, str) for p in ctx.blue_roster):
            self.team_last_roster[blue] = ctx.blue_roster
        if any(isinstance(p, str) for p in ctx.red_roster):
            self.team_last_roster[red] = ctx.red_roster

        if not (np.isnan(ctx.b_pelo) or np.isnan(ctx.r_pelo)):
            exp_pb = _expected(ctx.b_pelo, ctx.r_pelo)
            for p in ctx.blue_roster:
                if isinstance(p, str):
                    self.player_elo[p] += PLAYER_K * (result - exp_pb)
                    self.player_games[p] += 1
            for p in ctx.red_roster:
                if isinstance(p, str):
                    self.player_elo[p] += PLAYER_K * ((1 - result) - (1 - exp_pb))
                    self.player_games[p] += 1

        key = ctx.h2h_key
        if blue <= red:
            self.h2h[key][0 if result else 1] += 1
        else:
            self.h2h[key][1 if result else 0] += 1

        for c in ctx.blue_champs:
            self.champ_meta.update(c, result, date)
        for c in ctx.red_champs:
            self.champ_meta.update(c, 1 - result, date)
        self.champ_meta.update_total(10.0, date)

        for role, bc, rc in zip(ROLES, ctx.blue_champs, ctx.red_champs):
            self.role_meta[role].update(bc, result, date)
            self.role_meta[role].update(rc, 1 - result, date)
            if isinstance(bc, str) and isinstance(rc, str):
                self.matchup_meta.update(f"{role}|{bc}|{rc}", result, date)
                self.matchup_meta.update(f"{role}|{rc}|{bc}", 1 - result, date)

        if ctx.cross_league:
            exp_lb = _expected(ctx.b_lelo, ctx.r_lelo)
            self.league_elo[ctx.b_league] = ctx.b_lelo + LEAGUE_K * (result - exp_lb)
            self.league_elo[ctx.r_league] = ctx.r_lelo + LEAGUE_K * ((1 - result) - (1 - exp_lb))

        # A team's home league is where it plays domestically; international
        # events never overwrite it.
        if ctx.league not in INTERNATIONAL_LEAGUES:
            self.team_home_league[blue] = ctx.league
            self.team_home_league[red] = ctx.league

        if blue <= red:
            self.series[ctx.series_key][0 if result else 1] += 1
        else:
            self.series[ctx.series_key][1 if result else 0] += 1

        self.last_date = date if self.last_date is None else max(self.last_date, date)

    # --- lookups used at predict time ---------------------------------------

    def known_teams(self) -> list:
        """Teams the rating pool has actually seen, most recently played first."""
        return sorted(self.team_last_date, key=lambda t: self.team_last_date[t], reverse=True)

    def last_roster(self, team) -> Optional[tuple]:
        return self.team_last_roster.get(team)
