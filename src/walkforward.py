"""Chronological walk-forward evaluation with time-decay weighting.

Expanding-window design: for each test period, the model trains only on games
that finished strictly before that period opens. Calibration is fitted on the
tail of the training window (never the test period), so reported probabilities
are honest out-of-sample.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

from config import PROCESSED_DIR

warnings.filterwarnings("ignore")

# Known before the match starts, so usable against pre-match odds.
PREGAME_FEATURES = [
    "elo_diff", "elo_expected", "player_elo_diff",
    "blue_games_played", "red_games_played", "games_played_diff",
    "blue_days_idle", "red_days_idle",
    "blue_form", "red_form", "form_diff",
    "blue_roster_continuity", "red_roster_continuity",
    "blue_roster_exp", "red_roster_exp",
    "h2h_blue_wins", "h2h_red_wins", "h2h_games",
    "playoffs",
]

# Known only after champ select. Usable for live/in-play markets, not for
# pre-match odds, so they are evaluated as a separate feature set.
#
# Pooled champion rates are used rather than the role-conditioned and
# lane-matchup variants also computed in features.py: on walk-forward those
# were flat to slightly worse, since per-matchup samples are too thin to
# out-predict the pooled rate. League-strength and series-context features were
# likewise dropped as flat, team Elo being a global pool that already carries
# cross-league strength.
DRAFT_FEATURES = [
    "blue_champ_wr", "red_champ_wr", "champ_wr_diff",
    "blue_champ_presence", "red_champ_presence", "champ_presence_diff",
]

NUMERIC_FEATURES = PREGAME_FEATURES
CATEGORICAL_FEATURES = ["league"]

BURN_IN_MONTHS = 12        # warm the rating pool before scoring anything
DECAY_HALFLIFE_DAYS = 365.0
CALIB_FRACTION = 0.15      # tail of train window reserved for calibration
# Sigmoid over isotonic: the calibration slice is small enough that isotonic
# overfits it and degrades log loss, despite improving ECE.
CALIB_METHOD = "sigmoid"


def decay_weights(dates: pd.Series, ref_date, halflife: float) -> np.ndarray:
    """Exponential recency weights; an infinite half-life means no decay."""
    age_days = (ref_date - dates).dt.total_seconds().to_numpy() / 86400.0
    if not np.isfinite(halflife):
        return np.ones(len(age_days))
    return 0.5 ** (np.maximum(age_days, 0.0) / halflife)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def make_logistic(numeric=None) -> Pipeline:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50),
             CATEGORICAL_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_lightgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
    )


def prepare(df: pd.DataFrame, for_lgbm: bool, numeric=None) -> pd.DataFrame:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    X = df[numeric + CATEGORICAL_FEATURES].copy()
    if for_lgbm:
        X["league"] = X["league"].astype("category")
    return X


def run_walkforward(feats: pd.DataFrame, freq: str = "QE",
                    halflife: float = DECAY_HALFLIFE_DAYS,
                    calib_method: str = CALIB_METHOD,
                    numeric=None,
                    verbose: bool = True) -> pd.DataFrame:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    feats = feats.sort_values("date").reset_index(drop=True)
    start = feats["date"].min() + pd.DateOffset(months=BURN_IN_MONTHS)
    periods = pd.date_range(start=start, end=feats["date"].max(), freq=freq)

    preds = []
    for i, period_end in enumerate(periods):
        period_start = periods[i - 1] if i > 0 else start
        train = feats[feats["date"] < period_start]
        test = feats[(feats["date"] >= period_start) & (feats["date"] < period_end)]
        if len(test) == 0 or len(train) < 2000:
            continue

        # Temporal calibration split: the most recent slice of train is held
        # out to fit the calibrator, mirroring the real deployment gap.
        cut = int(len(train) * (1 - CALIB_FRACTION))
        fit_df, calib_df = train.iloc[:cut], train.iloc[cut:]

        y_fit = fit_df["blue_win"].to_numpy()
        w_fit = decay_weights(fit_df["date"], period_start, halflife)

        out = test[["gameid", "date", "league", "blue_win", "elo_expected"]].copy()
        out["period_start"] = period_start

        for name, is_lgbm in (("logistic", False), ("lightgbm", True)):
            model = make_lightgbm() if is_lgbm else make_logistic(numeric)
            X_fit = prepare(fit_df, is_lgbm, numeric)
            model.fit(X_fit, y_fit, **({"sample_weight": w_fit} if is_lgbm
                                       else {"clf__sample_weight": w_fit}))

            calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=calib_method)
            calibrator.fit(prepare(calib_df, is_lgbm, numeric), calib_df["blue_win"],
                           sample_weight=decay_weights(calib_df["date"], period_start, halflife))

            X_test = prepare(test, is_lgbm, numeric)
            out[f"p_{name}"] = model.predict_proba(X_test)[:, 1]
            out[f"p_{name}_cal"] = calibrator.predict_proba(X_test)[:, 1]

        preds.append(out)
        if verbose:
            print(f"  {period_start.date()} -> {period_end.date()}  "
                  f"train={len(train):>6,}  test={len(test):>5,}")

    return pd.concat(preds, ignore_index=True)


def _metrics_row(label: str, y, p) -> None:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    print(f"{label:<26} {log_loss(y, p):>9.4f} {brier_score_loss(y, p):>8.4f} "
          f"{((p > 0.5) == (y == 1)).mean():>7.4f} {roc_auc_score(y, p):>7.4f} "
          f"{expected_calibration_error(y, p):>7.4f}")


def reliability_table(y, p, n_bins: int = 10) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=True)
    print(f"\n{'bin':<14} {'n':>7} {'predicted':>10} {'actual':>8} {'gap':>8}")
    print("-" * 50)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        pred, act = p[mask].mean(), y[mask].mean()
        print(f"{bins[b]:.1f}-{bins[b+1]:.1f}{'':<6} {mask.sum():>7,} "
              f"{pred:>10.4f} {act:>8.4f} {act - pred:>+8.4f}")


def report(preds: pd.DataFrame, label: str) -> None:
    y = preds["blue_win"].to_numpy()
    print(f"\n=== {label} ===")
    print(f"{'model':<26} {'logloss':>9} {'brier':>8} {'acc':>7} {'auc':>7} {'ece':>7}")
    print("-" * 68)
    for name, col in [("elo only", "elo_expected"),
                      ("logistic", "p_logistic"), ("logistic + calib", "p_logistic_cal"),
                      ("lightgbm", "p_lightgbm"), ("lightgbm + calib", "p_lightgbm_cal")]:
        _metrics_row(name, y, preds[col].to_numpy())
    base = np.full_like(y, y.mean(), dtype=float)
    print(f"{'base rate':<26} {log_loss(y, base):>9.4f} {brier_score_loss(y, base):>8.4f} "
          f"{max(y.mean(), 1 - y.mean()):>7.4f} {'—':>7} {expected_calibration_error(y, base):>7.4f}")


def main() -> int:
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    print(f"Walk-forward over {len(feats):,} games "
          f"({feats['date'].min().date()} -> {feats['date'].max().date()})")

    sets = {
        "PRE-GAME ONLY (usable vs pre-match odds)": PREGAME_FEATURES,
        "PRE-GAME + DRAFT (post champ select)": PREGAME_FEATURES + DRAFT_FEATURES,
    }
    best = None
    for label, cols in sets.items():
        print(f"\nFitting: {label}")
        preds = run_walkforward(feats, numeric=cols)
        report(preds, label)
        if "PRE-GAME ONLY" in label:
            preds.to_parquet(PROCESSED_DIR / "walkforward_preds_pregame.parquet", index=False)
        else:
            preds.to_parquet(PROCESSED_DIR / "walkforward_preds_draft.parquet", index=False)
            best = preds

    print("\nReliability, best model (logistic + calib, pre-game + draft):")
    reliability_table(best["blue_win"].to_numpy(), best["p_logistic_cal"].to_numpy())
    return 0


if __name__ == "__main__":
    sys.exit(main())
