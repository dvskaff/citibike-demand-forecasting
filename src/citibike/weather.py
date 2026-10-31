"""Daily NYC weather from Open-Meteo (free, no API key).

* History comes from the ERA5-based archive API.
* The forecast window uses the forecast API, which also returns the last ~3 months of observed weather.
* Days beyond Open-Meteo's 16-day forecast range fall back to the historical average for that day of
  the year, and the dashboard flags them so the forecast is honest about what it knew.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
]
RENAME = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precip_mm",
    "snowfall_sum": "snow_cm",
    "precipitation_hours": "precip_hours",
    "wind_speed_10m_max": "wind_max",
}
WEATHER_COLS = list(RENAME.values())


def _to_frame(payload: dict, source: str) -> pl.DataFrame:
    daily = payload["daily"]
    df = pl.DataFrame({"date": daily["time"], **{RENAME[v]: daily[v] for v in DAILY_VARS}})
    return df.with_columns(
        pl.col("date").str.to_date(),
        *[pl.col(c).cast(pl.Float64) for c in WEATHER_COLS],
        pl.lit(source).alias("weather_source"),
    )


def fetch_history(lat: float, lon: float, tz: str, start: date, end: date) -> pl.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": tz,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
    return _to_frame(resp.json(), "observed")


def fetch_recent_and_forecast(lat: float, lon: float, tz: str) -> pl.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": tz,
        "past_days": 92,
        "forecast_days": 16,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=120)
    resp.raise_for_status()
    today = date.today()
    df = _to_frame(resp.json(), "forecast")
    return df.with_columns(
        pl.when(pl.col("date") < pl.lit(today))
        .then(pl.lit("observed"))
        .otherwise(pl.lit("forecast"))
        .alias("weather_source")
    )


def climatology_fill(weather: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Average weather for each day-of-year, used for days past the 16-day forecast range."""
    clim = (
        weather.filter(pl.col("weather_source") == "observed")
        .with_columns(pl.col("date").dt.ordinal_day().alias("doy"))
        .group_by("doy")
        .agg([pl.col(c).mean() for c in WEATHER_COLS])
    )
    days = pl.DataFrame({"date": pl.date_range(start, end, "1d", eager=True)}).with_columns(
        pl.col("date").dt.ordinal_day().alias("doy")
    )
    return (
        days.join(clim, on="doy", how="left")
        .drop("doy")
        .with_columns(pl.lit("climatology").alias("weather_source"))
        .select(weather.columns)
    )


def build_weather(lat: float, lon: float, tz: str, start: date, end: date) -> pl.DataFrame:
    """Observed weather for [start, archive limit] + recent/forecast weather + climatology up to `end`."""
    archive_end = min(end, date.today() - timedelta(days=6))
    parts = [fetch_history(lat, lon, tz, start, archive_end)]
    try:
        parts.append(fetch_recent_and_forecast(lat, lon, tz))
    except requests.RequestException as exc:  # forecast API is a nice-to-have
        log.warning("Forecast weather unavailable (%s); using climatology instead", exc)
    weather = (
        pl.concat(parts)
        .drop_nulls(subset=["temp_mean"])
        .unique(subset="date", keep="first", maintain_order=True)  # archive (observed) wins where both exist
        .filter(pl.col("date") <= pl.lit(end))
        .sort("date")
    )
    last = weather["date"].max()
    if last < end:
        weather = pl.concat([weather, climatology_fill(weather, last + timedelta(days=1), end)])
    return weather.sort("date")
