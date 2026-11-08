import json
from dataclasses import replace

from citibike.config import Config
from citibike.pipeline import run


def test_sample_pipeline_end_to_end(tmp_path):
    cfg = replace(
        Config(),
        n_backtest_folds=2,
        n_tuning_folds=1,
        optuna_trials=2,
        data_dir=tmp_path / "data",
        site_dir=tmp_path / "site",
        forecast_log_dir=tmp_path / "forecasts",
        mlflow_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    out = run(cfg, sample=True)
    page = (tmp_path / "site" / "index.html").read_text()
    assert "NYC Citi Bike Demand Forecast" in page and "Demo build" in page
    metrics = json.loads((tmp_path / "site" / "metrics.json").read_text())
    assert metrics["champion"] == out["champion"]
    assert {r["model"] for r in metrics["summary"]} >= {"AutoETS", "LightGBM (Optuna-tuned)", "XGBoost"}

    # cleaned data downloads: totals add up, weather joined, forecast complete
    import pandas as pd

    data = tmp_path / "site" / "data"
    wide = pd.read_csv(data / "citibike_daily.csv")
    long = pd.read_csv(data / "citibike_daily_by_type.csv")
    assert wide["rides"].sum() == long["trips"].sum() == metrics["total_rides"]
    assert (wide["rides"] == wide["rides_member"] + wide["rides_casual"]).all()
    bikes = [c for c in wide.columns if c.endswith("_bike")]
    assert bikes and (wide[bikes].sum(axis=1) == wide["rides"]).all()
    assert wide["temp_mean"].notna().mean() > 0.95 and wide["date"].is_unique
    fc = pd.read_csv(data / "citibike_forecast.csv")
    assert len(fc) == cfg.horizon and (fc["lo_80"] <= fc["yhat"]).all() and (fc["yhat"] <= fc["hi_80"]).all()
    assert 'href="data/citibike_daily.csv"' in page.replace("'", '"') and "Download the data" in page
