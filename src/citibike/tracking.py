"""Experiment tracking (MLflow), the forecast log, and the auto-updated README results table."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

for _var in ("MLFLOW_DISABLE_AGENT_HINT", "MLFLOW_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ.setdefault(_var, "true")

import mlflow  # noqa: E402  (env vars above must be set before import)
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from .evaluation import metrics  # noqa: E402

log = logging.getLogger(__name__)
EXPERIMENT = "citibike-daily-demand"


# --------------------------------------------------------------------------- MLflow
def log_runs(
    uri: str,
    preds: pd.DataFrame,
    summary: pd.DataFrame,
    best_params: dict,
    study: optuna.Study,
    run_info: dict,
) -> None:
    """One parent run per pipeline execution, one nested run per model."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"pipeline {run_info['data_end']}"):
        mlflow.log_params({k: v for k, v in run_info.items() if k != "best_params"})
        mlflow.log_dict(best_params, "optuna_best_params.json")
        mlflow.log_metric("optuna_best_validation_wape", study.best_value)
        for row in summary.itertuples(index=False):
            with mlflow.start_run(run_name=row.model, nested=True):
                mlflow.log_param("model", row.model)
                if "tuned" in row.model:
                    mlflow.log_params(best_params)
                mlflow.log_metrics(
                    {k: getattr(row, k) for k in ("MAE", "RMSE", "WAPE", "Bias", "WAPE_fold_std")}
                )
        mlflow.log_table(summary, "backtest_summary.json")
    log.info("Logged %d model runs to MLflow (%s)", len(summary), uri)


# --------------------------------------------------------------------------- forecast log
def save_forecast(forecast: pd.DataFrame, origin: pd.Timestamp, log_dir: Path) -> Path:
    """Keep every published forecast so it can be scored once the real numbers come out."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"forecast_from_{origin.date()}.csv"
    forecast.assign(origin=origin.date()).to_csv(path, index=False, float_format="%.1f")
    return path


def past_forecast_accuracy(log_dir: Path, totals: pl.DataFrame, last_obs: pd.Timestamp) -> pd.DataFrame:
    """Join earlier forecasts to the actual rides published since. This is true out-of-sample accuracy."""
    actual = totals.to_pandas().set_index("date")["y"]
    actual.index = pd.to_datetime(actual.index)
    rows = []
    for path in sorted(log_dir.glob("forecast_from_*.csv")):
        fc = pd.read_csv(path, parse_dates=["date"])
        if pd.Timestamp(fc["origin"].iloc[0]) >= last_obs:
            continue  # this run's own forecast, nothing to score yet
        fc["y"] = fc["date"].map(actual)
        rows.append(fc.dropna(subset=["y"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- README
def update_readme(readme: Path, summary: pd.DataFrame, coverage: pd.DataFrame, run_info: dict) -> None:
    """Rewrite the block between the RESULTS markers with the latest numbers."""
    lines = [
        f"_Last updated {run_info['run_utc']} · data {run_info['data_start']} to {run_info['data_end']} · "
        f"{run_info['total_rides']:,} rides · {run_info['n_folds']} backtest folds of {run_info['horizon']} days_",
        "",
        "| Model | WAPE | MAE (rides/day) | RMSE | Bias |",
        "|---|---|---|---|---|",
    ]
    for r in summary.itertuples(index=False):
        star = " ⭐" if r.model == run_info["champion"] else ""
        lines.append(f"| {r.model}{star} | {r.WAPE:.1%} | {r.MAE:,.0f} | {r.RMSE:,.0f} | {r.Bias:+.1%} |")
    cov = ", ".join(f"{int(c.level * 100)}% interval covered {c.coverage:.0%}" for c in coverage.itertuples())
    lines += ["", f"Conformal intervals on the held-out fold: {cov}."]
    block = "\n".join(lines)
    text = readme.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)"
    new = re.sub(pattern, lambda m: m.group(1) + block + m.group(2), text, flags=re.S)
    readme.write_text(new)


def score(fc: pd.DataFrame) -> dict:
    return metrics(fc["y"], fc["yhat"])
