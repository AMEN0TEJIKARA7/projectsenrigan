"""Chronological pre-game feature construction.

Every feature for a game is emitted from state built only from strictly earlier
games, then the game's result updates that state. A single forward pass over
date-sorted games therefore cannot leak outcome information.

Ratings are a global Elo pool rather than per-league pools: international events
(Worlds, MSI) and inter-region play are the only bridges that make cross-league
strength comparable, and separate pools would sever them.
"""

import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

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
# 89 teams that move between leagues), since intra-league games carry no
# information about how two leagues compare.
LEAGUE_ELO_INIT = 1500.0
LEAGUE_K = 12.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


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


def _revert(rating: float, last_date, now, halflife: float = REVERSION_HALFLIFE_DAYS) -> float:
    """Shrink a rating's deviation from the mean based on days idle."""
    if last_date is None:
        return rating
    days = (now - last_date).total_seconds() / 86400.0
    if days <= 0:
        return rating
    return ELO_INIT + (rating - ELO_INIT) * (0.5 ** (days / halflife))


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("date").reset_index(drop=True)

    team_elo: dict = defaultdict(lambda: ELO_INIT)
    team_games: dict = defaultdict(int)
    team_last_date: dict = {}
    team_form: dict = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    team_last_roster: dict = {}
    player_elo: dict = defaultdict(lambda: PLAYER_ELO_INIT)
    player_games: dict = defaultdict(int)
    champ_meta = ChampionMeta()
    # Champion strength conditioned on role, and on the opposing role pick.
    # A champion's value is role-specific, and lane matchups are the mechanism
    # people expect draft to matter through.
    role_meta = {r: ChampionMeta() for r in ROLES}
    matchup_meta = ChampionMeta()
    league_elo: dict = defaultdict(lambda: LEAGUE_ELO_INIT)
    team_home_league: dict = {}
    # Series state keyed by (team pair, date, league): game number within a
    # best-of and the running score, both known before the next game starts.
    series: dict = defaultdict(lambda: [0, 0])
    h2h: dict = defaultdict(lambda: [0, 0])  # (a,b) sorted key -> [a_wins, b_wins]

    rows = []
    for g in games.itertuples(index=False):
        date = g.date
        blue, red = g.blue_team, g.red_team
        blue_roster = tuple(getattr(g, f"blue_player_{r}") for r in ROLES)
        red_roster = tuple(getattr(g, f"red_player_{r}") for r in ROLES)

        # --- pre-game state (features) --------------------------------------
        b_elo = _revert(team_elo[blue], team_last_date.get(blue), date)
        r_elo = _revert(team_elo[red], team_last_date.get(red), date)

        b_played, r_played = team_games[blue], team_games[red]
        b_idle = ((date - team_last_date[blue]).total_seconds() / 86400.0
                  if blue in team_last_date else np.nan)
        r_idle = ((date - team_last_date[red]).total_seconds() / 86400.0
                  if red in team_last_date else np.nan)

        b_form = np.mean(team_form[blue]) if team_form[blue] else np.nan
        r_form = np.mean(team_form[red]) if team_form[red] else np.nan

        # Roster continuity: share of the starting five unchanged from the
        # team's previous game. Roster churn degrades the meaning of team Elo.
        def continuity(team, roster):
            prev = team_last_roster.get(team)
            if prev is None:
                return np.nan
            known = [p for p in roster if isinstance(p, str)]
            if not known:
                return np.nan
            return len(set(roster) & set(prev)) / 5.0

        b_cont = continuity(blue, blue_roster)
        r_cont = continuity(red, red_roster)

        def roster_elo(roster):
            vals = [player_elo[p] for p in roster if isinstance(p, str)]
            return float(np.mean(vals)) if vals else np.nan

        b_pelo, r_pelo = roster_elo(blue_roster), roster_elo(red_roster)

        def roster_exp(roster):
            vals = [player_games[p] for p in roster if isinstance(p, str)]
            return float(np.mean(vals)) if vals else np.nan

        # Draft state. Available only after champ select, so these are kept
        # separate from features known before the match starts.
        blue_champs = [getattr(g, f"blue_champ_{r}") for r in ROLES]
        red_champs = [getattr(g, f"red_champ_{r}") for r in ROLES]

        def champ_agg(champs, fn):
            vals = [v for v in (fn(c, date) for c in champs) if not pd.isna(v)]
            return float(np.mean(vals)) if vals else np.nan

        b_cwr = champ_agg(blue_champs, champ_meta.win_rate)
        r_cwr = champ_agg(red_champs, champ_meta.win_rate)
        b_pres = champ_agg(blue_champs, champ_meta.presence)
        r_pres = champ_agg(red_champs, champ_meta.presence)

        # Role-conditioned champion win rates.
        b_role_wr, r_role_wr = [], []
        for role, bc, rc in zip(ROLES, blue_champs, red_champs):
            b_role_wr.append(role_meta[role].win_rate(bc, date))
            r_role_wr.append(role_meta[role].win_rate(rc, date))
        b_rwr = float(np.nanmean(b_role_wr)) if not all(pd.isna(v) for v in b_role_wr) else np.nan
        r_rwr = float(np.nanmean(r_role_wr)) if not all(pd.isna(v) for v in r_role_wr) else np.nan

        # Lane matchup: blue champ vs red champ in the same role, keyed
        # directionally so the rate is "how often does A beat B here".
        mu_rates, mu_vol = [], []
        for role, bc, rc in zip(ROLES, blue_champs, red_champs):
            if isinstance(bc, str) and isinstance(rc, str):
                mkey = f"{role}|{bc}|{rc}"
                mu_rates.append(matchup_meta.win_rate(mkey, date))
                mu_vol.append(matchup_meta.volume(mkey, date))
        b_mu = float(np.mean(mu_rates)) if mu_rates else np.nan
        b_mu_vol = float(np.mean(mu_vol)) if mu_vol else np.nan

        key = (blue, red) if blue <= red else (red, blue)
        rec = h2h[key]
        b_h2h, r_h2h = (rec[0], rec[1]) if blue <= red else (rec[1], rec[0])

        # League strength gap, using each team's home league rather than the
        # league this game is played in, so internationals compare regions.
        b_league = team_home_league.get(blue, g.league)
        r_league = team_home_league.get(red, g.league)
        b_lelo, r_lelo = league_elo[b_league], league_elo[r_league]
        cross_league = b_league != r_league

        skey = (key, date.date(), g.league)
        s_rec = series[skey]
        s_blue, s_red = (s_rec[0], s_rec[1]) if blue <= red else (s_rec[1], s_rec[0])

        rows.append({
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
            "blue_roster_exp": roster_exp(blue_roster),
            "red_roster_exp": roster_exp(red_roster),
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
        })

        # --- post-game state update -----------------------------------------
        result = g.blue_win
        exp_b = _expected(b_elo, r_elo)
        k_b = K_PROVISIONAL if b_played < PROVISIONAL_GAMES else K_BASE
        k_r = K_PROVISIONAL if r_played < PROVISIONAL_GAMES else K_BASE
        team_elo[blue] = b_elo + k_b * (result - exp_b)
        team_elo[red] = r_elo + k_r * ((1 - result) - (1 - exp_b))

        team_games[blue] += 1
        team_games[red] += 1
        team_last_date[blue] = date
        team_last_date[red] = date
        team_form[blue].append(result)
        team_form[red].append(1 - result)
        if any(isinstance(p, str) for p in blue_roster):
            team_last_roster[blue] = blue_roster
        if any(isinstance(p, str) for p in red_roster):
            team_last_roster[red] = red_roster

        if not (np.isnan(b_pelo) or np.isnan(r_pelo)):
            exp_pb = _expected(b_pelo, r_pelo)
            for p in blue_roster:
                if isinstance(p, str):
                    player_elo[p] += PLAYER_K * (result - exp_pb)
                    player_games[p] += 1
            for p in red_roster:
                if isinstance(p, str):
                    player_elo[p] += PLAYER_K * ((1 - result) - (1 - exp_pb))
                    player_games[p] += 1

        if blue <= red:
            h2h[key][0 if result else 1] += 1
        else:
            h2h[key][1 if result else 0] += 1

        for c in blue_champs:
            champ_meta.update(c, result, date)
        for c in red_champs:
            champ_meta.update(c, 1 - result, date)
        champ_meta.update_total(10.0, date)

        for role, bc, rc in zip(ROLES, blue_champs, red_champs):
            role_meta[role].update(bc, result, date)
            role_meta[role].update(rc, 1 - result, date)
            if isinstance(bc, str) and isinstance(rc, str):
                matchup_meta.update(f"{role}|{bc}|{rc}", result, date)
                matchup_meta.update(f"{role}|{rc}|{bc}", 1 - result, date)

        if cross_league:
            exp_lb = _expected(b_lelo, r_lelo)
            league_elo[b_league] = b_lelo + LEAGUE_K * (result - exp_lb)
            league_elo[r_league] = r_lelo + LEAGUE_K * ((1 - result) - (1 - exp_lb))

        # A team's home league is where it plays domestically; international
        # events never overwrite it.
        if g.league not in ("MSI", "WLDs"):
            team_home_league[blue] = g.league
            team_home_league[red] = g.league

        if blue <= red:
            series[skey][0 if result else 1] += 1
        else:
            series[skey][1 if result else 0] += 1

    return pd.DataFrame(rows)


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
