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
