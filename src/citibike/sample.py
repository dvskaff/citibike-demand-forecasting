"""Synthetic data with the same shape as the real sources.

Used by the test suite and by `python -m citibike.pipeline --sample`, so the whole pipeline can be run
and demonstrated offline in about a minute. The real dashboard on GitHub Pages is always built from the
real Citi Bike and Open-Meteo data.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from .weather import WEATHER_COLS


def make_weather(start: date, end: date, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = pl.date_range(start, end, "1d", eager=True)
    doy = np.array([d.timetuple().tm_yday for d in days])
    n = len(days)
    seasonal = 12.5 - 11.5 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    temp_mean = seasonal + rng.normal(0, 3.5, n)
    wet = rng.random(n) < 0.32
    precip = np.where(wet, rng.gamma(0.9, 7.0, n), 0.0)
    snow = np.where((temp_mean < 1) & wet, precip * 0.7, 0.0)
    return (
        pl.DataFrame(
            {
                "date": days,
                "temp_max": temp_mean + rng.uniform(3, 7, n),
                "temp_min": temp_mean - rng.uniform(3, 7, n),
                "temp_mean": temp_mean,
                "precip_mm": precip,
                "snow_cm": snow,
                "precip_hours": np.where(wet, np.clip(precip * 0.9, 1, 24), 0.0),
                "wind_max": rng.gamma(4, 5, n),
            }
        )
        .with_columns(pl.lit("observed").alias("weather_source"))
        .select(["date", *WEATHER_COLS, "weather_source"])
    )


def make_daily_trips(weather: pl.DataFrame, seed: int = 1) -> pl.DataFrame:
    """Rides per day x rider type x bike type, driven by season, weekday, weather and growth."""
    rng = np.random.default_rng(seed)
    w = weather.to_pandas()
    n = len(w)
    t = np.arange(n)
    dow = w["date"].map(lambda d: d.weekday()).to_numpy()
    temp = w["temp_mean"].to_numpy()
    rain = w["precip_mm"].to_numpy()
    snow = w["snow_cm"].to_numpy()

    level = 95_000 * (1 + 0.10 * t / 365)  # ~10% yearly growth
    temp_eff = np.exp(-((temp - 23) ** 2) / (2 * 11**2))
    weekday_eff = np.where(dow >= 5, 0.85, 1.0)
    weather_eff = np.exp(-0.035 * rain) * np.exp(-0.15 * snow)
    total = level * (0.25 + 0.75 * temp_eff) * weekday_eff * weather_eff * rng.lognormal(0, 0.05, n)

    casual_share = np.clip(0.12 + 0.10 * temp_eff + np.where(dow >= 5, 0.07, 0.0), 0, 0.6)
    ebike_share = np.clip(0.55 + 0.08 * t / 365, 0, 0.9)
    rows = []
    for i, d in enumerate(w["date"]):
        for member, ms in (("member", 1 - casual_share[i]), ("casual", casual_share[i])):
            for bike, bs in (("electric_bike", ebike_share[i]), ("classic_bike", 1 - ebike_share[i])):
                rows.append((d, member, bike, int(round(total[i] * ms * bs))))
    return pl.DataFrame(
        rows, schema=["date", "member_casual", "rideable_type", "trips"], orient="row"
    ).with_columns(pl.col("date").cast(pl.Date), pl.col("trips").cast(pl.Int64))


def sample_inputs(end: date | None = None, years: float = 3.5, horizon: int = 30, seed: int = 7):
    """(daily_trips, weather) where weather extends `horizon` days past the last day of trips."""
    end = end or date(2026, 9, 30)
    start = end - timedelta(days=int(365 * years))
    weather = make_weather(start, end + timedelta(days=horizon), seed=seed)
    trips = make_daily_trips(weather.filter(pl.col("date") <= pl.lit(end)), seed=seed + 1)
    future = weather.filter(pl.col("date") > pl.lit(end)).with_columns(
        pl.lit("forecast").alias("weather_source")
    )
    weather = pl.concat([weather.filter(pl.col("date") <= pl.lit(end)), future])
    return trips, weather
