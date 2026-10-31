"""Feature engineering with Polars.

Every lag is at least `MIN_LAG` days old, so a model forecasting up to `MIN_LAG` days ahead never sees
a value from inside its own forecast window (direct multi-horizon forecasting with no leakage).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd
import polars as pl

from .weather import WEATHER_COLS

MIN_LAG = 30
LAGS = [30, 35, 364]
FOURIER_K = 3

CALENDAR_COLS = ["dow", "is_weekend", "month", "week_of_year", "trend"]
HOLIDAY_COLS = ["is_holiday", "day_before_holiday", "day_after_holiday"]
LAG_COLS = [f"lag_{k}" for k in LAGS] + ["roll_mean_7_lag30", "roll_mean_28_lag30"]
FOURIER_COLS = [f"{f}_{k}" for k in range(1, FOURIER_K + 1) for f in ("sin", "cos")]
FEATURE_COLS = CALENDAR_COLS + FOURIER_COLS + HOLIDAY_COLS + WEATHER_COLS + LAG_COLS


def daily_totals(daily: pl.DataFrame) -> pl.DataFrame:
    """Collapse rider/bike breakdown to one total per day, filling any missing day with null."""
    totals = daily.group_by("date").agg(pl.col("trips").sum().alias("y")).sort("date")
    full = pl.DataFrame({"date": pl.date_range(totals["date"].min(), totals["date"].max(), "1d", eager=True)})
    return full.join(totals, on="date", how="left")


def _holiday_dates(years: range) -> set:
    us_ny = holidays.country_holidays("US", subdiv="NY", years=list(years))
    return set(us_ny.keys())


def build_features(totals: pl.DataFrame, weather: pl.DataFrame, horizon: int) -> pd.DataFrame:
    """Feature table for every observed day plus `horizon` future days (whose `y` is null)."""
    if horizon > MIN_LAG:
        raise ValueError(f"horizon ({horizon}) must be <= {MIN_LAG} to keep lag features leak-free")

    last = totals["date"].max()
    future = pl.DataFrame(
        {"date": pl.date_range(last, last + pl.duration(days=horizon), "1d", eager=True)[1:]}
    )
    frame = pl.concat([totals, future.with_columns(pl.lit(None, dtype=totals["y"].dtype).alias("y"))])
    start = frame["date"].min()

    hol = _holiday_dates(range(start.year, frame["date"].max().year + 2))
    hol_series = pl.Series("hol", sorted(hol), dtype=pl.Date)
    two_pi_doy = 2 * np.pi * pl.col("date").dt.ordinal_day() / 365.25

    frame = frame.with_columns(
        (pl.col("date").dt.weekday() - 1).alias("dow"),  # Monday=0
        (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.week().alias("week_of_year"),
        (pl.col("date") - pl.lit(start)).dt.total_days().alias("trend"),
        pl.col("date").is_in(hol_series.implode()).cast(pl.Int8).alias("is_holiday"),
        (pl.col("date") + pl.duration(days=1))
        .is_in(hol_series.implode())
        .cast(pl.Int8)
        .alias("day_before_holiday"),
        (pl.col("date") - pl.duration(days=1))
        .is_in(hol_series.implode())
        .cast(pl.Int8)
        .alias("day_after_holiday"),
        *[(k * two_pi_doy).sin().alias(f"sin_{k}") for k in range(1, FOURIER_K + 1)],
        *[(k * two_pi_doy).cos().alias(f"cos_{k}") for k in range(1, FOURIER_K + 1)],
        *[pl.col("y").shift(k).alias(f"lag_{k}") for k in LAGS],
        pl.col("y").shift(MIN_LAG).rolling_mean(7, min_samples=5).alias("roll_mean_7_lag30"),
        pl.col("y").shift(MIN_LAG).rolling_mean(28, min_samples=20).alias("roll_mean_28_lag30"),
    ).join(weather, on="date", how="left")
    return frame.select(["date", "y", *FEATURE_COLS, "weather_source"]).to_pandas().set_index("date")
