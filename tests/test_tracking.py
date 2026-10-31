from datetime import date

import pandas as pd
import polars as pl

from citibike.report import fig_tracking
from citibike.tracking import past_forecast_accuracy, save_forecast


def test_saved_forecasts_are_scored_once_actuals_arrive(tmp_path):
    days = pl.date_range(date(2025, 1, 1), date(2025, 3, 31), "1d", eager=True)
    totals = pl.DataFrame({"date": days, "y": [1000.0] * len(days)})
    old = pd.DataFrame(
        {"date": pd.date_range("2025-02-01", periods=30), "yhat": 900.0, "weather_source": "forecast"}
    )
    save_forecast(old, pd.Timestamp("2025-01-31"), tmp_path)
    new = pd.DataFrame(
        {"date": pd.date_range("2025-04-01", periods=30), "yhat": 1.0, "weather_source": "forecast"}
    )
    save_forecast(new, pd.Timestamp("2025-03-31"), tmp_path)

    scored = past_forecast_accuracy(tmp_path, totals, pd.Timestamp("2025-03-31"))
    assert len(scored) == 30  # only the older forecast has actuals; this run's own forecast is skipped
    assert (scored["y"] == 1000).all()
    tot = totals.to_pandas().set_index("date")["y"]
    tot.index = pd.to_datetime(tot.index)
    assert len(fig_tracking(scored, tot).data) == 2
