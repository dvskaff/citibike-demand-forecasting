"""End-to-end pipeline: data -> features -> tuning -> backtests -> intervals -> forecast -> dashboard.

python -m citibike.pipeline            # real Citi Bike + Open-Meteo data (what GitHub Actions runs)
python -m citibike.pipeline --sample   # offline demo on synthetic data, about a minute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from . import downloads as dl
from . import evaluation as ev
from .config import ROOT, Config
from .explain import shap_summary
from .features import build_features, daily_totals, split
from .models import LightGBMForecaster, XGBoostForecaster, baseline_models
from .report import build_site
from .tracking import log_runs, past_forecast_accuracy, save_forecast, update_readme

log = logging.getLogger("citibike")


# --------------------------------------------------------------------------- data
def load_real(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    from .ingest import ingest_trips
    from .weather import build_weather

    daily = ingest_trips(cfg.tripdata_bucket_url, cfg.history_start, cfg.raw_dir, cfg.interim_dir)
    last = daily["date"].max()
    weather = build_weather(
        cfg.latitude,
        cfg.longitude,
        cfg.timezone,
        date.fromisoformat(cfg.history_start),
        last + timedelta(days=cfg.horizon),
    )
    return daily, weather


def load_sample(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    from .sample import sample_inputs

    return sample_inputs(horizon=cfg.horizon, seed=cfg.seed)


# --------------------------------------------------------------------------- main
def run(cfg: Config, sample: bool = False) -> dict:
    t0 = time.time()
    daily, weather = (load_sample if sample else load_real)(cfg)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(cfg.processed_dir / "daily_trips.parquet")
    weather.write_parquet(cfg.processed_dir / "weather.parquet")

    totals = daily_totals(daily)
    feats = build_features(totals, weather, cfg.horizon)
    observed, future = split(feats)
    last_obs = observed.index.max()
    log.info(
        "Data: %s to %s, %d days, %s rides",
        observed.index.min().date(),
        last_obs.date(),
        len(observed),
        f"{int(observed['y'].sum()):,}",
    )

    test_origins = ev.origins(last_obs, cfg.horizon, cfg.n_backtest_folds)
    tune_origins = ev.origins(last_obs, cfg.horizon, cfg.n_tuning_folds, skip=cfg.n_backtest_folds)

    # 1. Tune LightGBM on validation folds that precede the test folds.
    log.info("Tuning LightGBM with Optuna (%d trials)...", cfg.optuna_trials)
    study = ev.tune_lightgbm(feats, tune_origins, cfg.horizon, cfg.optuna_trials, cfg.seed)
    best_params = study.best_params
    log.info("Best validation WAPE %.4f with %s", study.best_value, best_params)

    # 2. Backtest every model on the same test folds.
    candidates = [
        *[(lambda m=m: m) for m in baseline_models()],
        lambda: LightGBMForecaster({}, seed=cfg.seed, name="LightGBM (default)"),
        lambda: LightGBMForecaster(best_params, seed=cfg.seed, name="LightGBM (Optuna-tuned)"),
        lambda: XGBoostForecaster(seed=cfg.seed),
    ]
    preds = pd.concat(
        [ev.backtest(factory, feats, test_origins, cfg.horizon) for factory in candidates], ignore_index=True
    )
    summary = ev.summarize(preds)
    champion_name = summary.iloc[0]["model"]
    log.info("Backtest results:\n%s", summary.to_string(index=False))

    # 3. Conformal intervals from the champion's backtest errors.
    champ_preds = preds[preds["model"] == champion_name]
    coverage = ev.interval_coverage(champ_preds, cfg.interval_levels)
    bounds = {
        lvl: ev.conformal_bounds(ev.relative_residuals(champ_preds), lvl) for lvl in cfg.interval_levels
    }

    # 4. Refit the champion on all history and forecast the next `horizon` days.
    factory = next(f for f in candidates if f().name == champion_name)
    champion = factory().fit(observed)
    yhat = champion.predict(future)
    forecast = pd.DataFrame(
        {"date": future.index, "yhat": yhat, "weather_source": future["weather_source"].to_numpy()}
    )
    for lvl, b in bounds.items():
        lo, hi = ev.apply_interval(yhat, b)
        forecast[f"lo_{int(lvl * 100)}"], forecast[f"hi_{int(lvl * 100)}"] = lo, hi

    # 5. Explain the tuned tree model (refit on all history) with SHAP.
    lgbm = LightGBMForecaster(best_params, seed=cfg.seed).fit(observed)
    shap_importance, shap_values = shap_summary(lgbm.model, observed.tail(365))

    # 6. Track: MLflow runs, forecast log, accuracy of earlier forecasts.
    run_info = {
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "data_start": str(observed.index.min().date()),
        "data_end": str(last_obs.date()),
        "n_days": int(len(observed)),
        "total_rides": int(observed["y"].sum()),
        "champion": champion_name,
        "horizon": cfg.horizon,
        "n_folds": cfg.n_backtest_folds,
        "optuna_trials": cfg.optuna_trials,
        "best_params": best_params,
        "sample_mode": sample,
    }
    log_runs(cfg.mlflow_uri, preds, summary, best_params, study, run_info)
    if not sample:
        save_forecast(forecast, last_obs, cfg.forecast_log_dir)
    tracking = (
        past_forecast_accuracy(cfg.forecast_log_dir, totals, last_obs) if not sample else pd.DataFrame()
    )

    # 7. Cleaned data for download, dashboard, README.
    downloads = dl.write(cfg.site_dir, daily, weather, forecast)
    build_site(
        cfg.site_dir,
        daily=daily,
        weather=weather,
        totals=totals,
        preds=preds,
        summary=summary,
        coverage=coverage,
        forecast=forecast,
        shap_importance=shap_importance,
        shap_values=shap_values,
        features=observed.tail(365),
        tracking=tracking,
        run_info=run_info,
        downloads=downloads,
    )
    metrics_out = {
        **run_info,
        "summary": summary.round(4).to_dict(orient="records"),
        "coverage": coverage.round(4).to_dict(orient="records"),
        "interval_bounds": {str(k): v for k, v in bounds.items()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    (cfg.site_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))
    if not sample:
        update_readme(ROOT / "README.md", summary, coverage, run_info)
    log.info("Done in %.0fs. Dashboard: %s", time.time() - t0, cfg.site_dir / "index.html")
    return metrics_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample", action="store_true", help="run offline on synthetic data")
    parser.add_argument("--start", help="first day of history (YYYY-MM-DD)")
    parser.add_argument("--trials", type=int, help="Optuna trials")
    parser.add_argument("--horizon", type=int, help="days to forecast (max 30)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for var in ("MLFLOW_DISABLE_AGENT_HINT", "MLFLOW_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ.setdefault(var, "true")
    cfg = Config()
    overrides = {"history_start": args.start, "optuna_trials": args.trials, "horizon": args.horizon}
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    np.random.seed(cfg.seed)
    run(cfg, sample=args.sample)


if __name__ == "__main__":
    main()
