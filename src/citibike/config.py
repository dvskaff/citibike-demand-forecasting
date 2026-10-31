"""Project-wide settings. Everything can be overridden from the command line (see pipeline.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- data ---
    history_start: str = "2023-01-01"  # first day of trip history to use
    tripdata_bucket_url: str = "https://s3.amazonaws.com/tripdata"
    # Manhattan, used for city-wide daily weather
    latitude: float = 40.7831
    longitude: float = -73.9712
    timezone: str = "America/New_York"

    # --- forecasting ---
    horizon: int = 30  # days ahead to forecast from the last day of published data
    n_backtest_folds: int = 6  # rolling-origin test windows (each `horizon` days long)
    n_tuning_folds: int = 3  # validation windows *before* the test windows, used only by Optuna
    optuna_trials: int = 40
    interval_levels: tuple[float, ...] = (0.8, 0.95)
    seed: int = 42

    # --- paths ---
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    site_dir: Path = field(default_factory=lambda: ROOT / "site")
    forecast_log_dir: Path = field(default_factory=lambda: ROOT / "forecasts")
    mlflow_uri: str = field(default_factory=lambda: f"sqlite:///{ROOT / 'mlflow.db'}")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """One small Parquet file of daily counts per downloaded source file (cached between runs)."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
