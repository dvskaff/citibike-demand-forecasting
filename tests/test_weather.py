from datetime import date, timedelta

import polars as pl

from citibike import weather as w


def _payload(start, n, temp=10.0):
    days = [(start + timedelta(days=i)).isoformat() for i in range(n)]
    return {"daily": {"time": days, **{v: [temp] * n for v in w.DAILY_VARS}}}


def test_to_frame_renames_and_types():
    df = w._to_frame(_payload(date(2025, 1, 1), 3), "observed")
    assert df.columns == ["date", *w.WEATHER_COLS, "weather_source"]
    assert df["date"].dtype == pl.Date


def test_build_weather_fills_far_future_with_climatology(monkeypatch):
    start, end = date(2024, 1, 1), date.today() + timedelta(days=40)
    monkeypatch.setattr(
        w,
        "fetch_history",
        lambda *a: w._to_frame(_payload(start, (date.today() - start).days - 6), "observed"),
    )
    monkeypatch.setattr(
        w,
        "fetch_recent_and_forecast",
        lambda *a: w._to_frame(_payload(date.today() - timedelta(days=5), 21, temp=20.0), "forecast"),
    )
    out = w.build_weather(0, 0, "UTC", start, end)
    assert out["date"].max() == end
    assert out["date"].is_unique().all()
    assert set(out["weather_source"].unique()) == {"observed", "forecast", "climatology"}
