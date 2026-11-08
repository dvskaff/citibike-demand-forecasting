"""Cleaned data for download: the daily ride counts the models are trained on, with the weather, and the latest
forecast. Written to site/data/ on every run, so the dashboard's "Download the data" section always matches it.

* citibike_daily.csv: one row per day, total rides, rides by rider type (member / casual) and bike type, and the
  observed weather that day (Open-Meteo, Central Park).
* citibike_daily_by_type.csv: the same counts in long form (date x rider type x bike type), as aggregated from the
  trip files.
* citibike_forecast.csv: the published forecast with its 80% and 95% intervals.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from .weather import WEATHER_COLS

log = logging.getLogger(__name__)
UNITS = {"temp_max": "°C", "temp_min": "°C", "temp_mean": "°C", "precip_mm": "mm", "snow_cm": "cm",
         "precip_hours": "hours", "wind_max": "km/h"}  # fmt: skip


def wide_daily(daily: pl.DataFrame, weather: pl.DataFrame) -> pd.DataFrame:
    d = daily.to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    out = d.groupby("date")["trips"].sum().rename("rides").to_frame()
    for col in ("member_casual", "rideable_type"):
        p = d.pivot_table(index="date", columns=col, values="trips", aggfunc="sum", fill_value=0)
        out = out.join(p.add_prefix("rides_").astype(int))
    w = weather.filter(pl.col("weather_source") == "observed").to_pandas()
    w["date"] = pd.to_datetime(w["date"])
    out = out.join(w.set_index("date")[WEATHER_COLS].round(2))
    out.index = out.index.date
    return out.rename_axis("date").reset_index()


def write(site_dir: Path, daily: pl.DataFrame, weather: pl.DataFrame, forecast: pd.DataFrame) -> list[dict]:
    """Write the CSVs and return a description of each (for the dashboard)."""
    out_dir = site_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide = wide_daily(daily, weather)
    long = daily.sort("date", "member_casual", "rideable_type").to_pandas()
    fc = forecast.copy()
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    fc = fc.round({c: 0 for c in fc.columns if c.startswith(("yhat", "lo_", "hi_"))})
    files = [
        ("citibike_daily.csv", wide, "One row per day: total rides, rides by rider type and bike type, and the "
         "observed weather (" + ", ".join(f"{c} in {u}" for c, u in UNITS.items()) + ")."),
        ("citibike_daily_by_type.csv", long, "Daily ride counts by rider type and bike type (long form), as "
         "aggregated from the public trip files."),
        ("citibike_forecast.csv", fc, "The published forecast: predicted rides per day (yhat) with 80% and 95% "
         "intervals, and whether that day's weather is a forecast or seasonal average."),
    ]  # fmt: skip
    meta = []
    for name, df, desc in files:
        df.to_csv(out_dir / name, index=False)
        meta.append({"file": f"data/{name}", "name": name, "rows": len(df), "columns": list(df.columns),
                     "kb": round((out_dir / name).stat().st_size / 1024), "description": desc})  # fmt: skip
    log.info("Data downloads: %s", ", ".join(f"{m['name']} ({m['rows']:,} rows)" for m in meta))
    return meta
