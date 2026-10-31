"""Rolling-origin backtesting, Optuna tuning, and conformal prediction intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from .models import Forecaster, LightGBMForecaster


# --------------------------------------------------------------------------- metrics
def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "WAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(y))),  # weighted absolute % error
        "Bias": float(np.sum(yhat - y) / np.sum(np.abs(y))),  # + means over-forecasting
    }


# --------------------------------------------------------------------------- folds
def origins(last_observed: pd.Timestamp, horizon: int, n_folds: int, skip: int = 0) -> list[pd.Timestamp]:
    """Forecast origins, oldest first. Fold i forecasts (origin, origin + horizon].

    The `n_folds` test windows tile the most recent `n_folds * horizon` days; `skip` shifts the whole set
    further back (used to place Optuna's validation folds *before* the test folds).
    """
    return [last_observed - pd.Timedelta(days=horizon * (k + skip)) for k in range(n_folds, 0, -1)]


def split(features: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = features[features["y"].notna()]
    train = observed[observed.index <= origin]
    test = observed[(observed.index > origin) & (observed.index <= origin + pd.Timedelta(days=horizon))]
    return train, test


def backtest(
    model_factory: Callable[[], Forecaster], features: pd.DataFrame, fold_origins: list, horizon: int
) -> pd.DataFrame:
    """Refit at every origin and forecast the next `horizon` days. Returns one row per forecast day."""
    rows = []
    for origin in fold_origins:
        train, test = split(features, origin, horizon)
        model = model_factory().fit(train)
        pred = model.predict(test)
        rows.append(
            pd.DataFrame(
                {
                    "model": model.name,
                    "origin": origin,
                    "date": test.index,
                    "h": (test.index - origin).days,
                    "y": test["y"].to_numpy(dtype=float),
                    "yhat": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in preds.groupby("model", sort=False):
        per_fold = g.groupby("origin").apply(
            lambda f: metrics(f["y"], f["yhat"])["WAPE"], include_groups=False
        )
        out.append({"model": name, **metrics(g["y"], g["yhat"]), "WAPE_fold_std": float(per_fold.std())})
    return pd.DataFrame(out).sort_values("WAPE").reset_index(drop=True)


# --------------------------------------------------------------------------- tuning
def tune_lightgbm(features: pd.DataFrame, tuning_origins: list, horizon: int, n_trials: int, seed: int):
    """Optuna search on validation folds that end before the test folds start (no test-set peeking)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        preds = backtest(lambda: LightGBMForecaster(params, seed=seed), features, tuning_origins, horizon)
        return metrics(preds["y"], preds["yhat"])["WAPE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(  # start from the hand-picked defaults so tuning can only help
        {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# --------------------------------------------------------------------------- conformal intervals
def relative_residuals(preds: pd.DataFrame) -> np.ndarray:
    """(actual - forecast) / forecast. Relative errors keep intervals proportional across seasons."""
    p = preds[preds["yhat"] > 0]
    return ((p["y"] - p["yhat"]) / p["yhat"]).to_numpy()


def conformal_bounds(residuals: np.ndarray, level: float) -> tuple[float, float]:
    """Split-conformal quantiles of relative residuals (with the finite-sample correction)."""
    n = len(residuals)
    alpha = 1 - level
    lo_q = max(0.0, np.floor((n + 1) * (alpha / 2)) / n)
    hi_q = min(1.0, np.ceil((n + 1) * (1 - alpha / 2)) / n)
    return float(np.quantile(residuals, lo_q)), float(np.quantile(residuals, hi_q))


def apply_interval(yhat: np.ndarray, bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = bounds
    return np.clip(yhat * (1 + lo), 0, None), yhat * (1 + hi)


def interval_coverage(preds: pd.DataFrame, levels: tuple[float, ...]) -> pd.DataFrame:
    """Calibrate on every fold except the latest, then measure coverage on the latest (held-out) fold."""
    folds = sorted(preds["origin"].unique())
    calib, holdout = preds[preds["origin"].isin(folds[:-1])], preds[preds["origin"] == folds[-1]]
    rows = []
    for level in levels:
        lo, hi = apply_interval(
            holdout["yhat"].to_numpy(), conformal_bounds(relative_residuals(calib), level)
        )
        y = holdout["y"].to_numpy()
        rows.append({"level": level, "coverage": float(np.mean((y >= lo) & (y <= hi))), "n_days": len(y)})
    return pd.DataFrame(rows)
