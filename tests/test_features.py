from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from citibike.features import FEATURE_COLS, LAG_COLS, build_features, daily_totals, split
from citibike.sample import sample_inputs


@pytest.fixture(scope="module")
def inputs():
    return sample_inputs(end=date(2025, 6, 30), years=2.2, horizon=30)


def test_feature_table_shape(inputs):
    daily, weather = inputs
    feats = build_features(daily_totals(daily), weather, horizon=30)
    assert feats["y"].isna().sum() == 30  # exactly the forecast window is unlabeled
    assert set(FEATURE_COLS) <= set(feats.columns)
    future = feats[feats["y"].isna()]
    assert future[[c for c in FEATURE_COLS if c not in LAG_COLS]].notna().all().all()


def test_lag_features_never_see_inside_the_forecast_window(inputs):
    """Hiding every value after an origin must not change features for the next 30 days."""
    daily, weather = inputs
    totals = daily_totals(daily)
    origin = date(2025, 3, 31)
    full = build_features(totals, weather, horizon=30)
    hidden = totals.with_columns(
        pl.when(pl.col("date") > origin).then(None).otherwise(pl.col("y")).alias("y")
    )
    masked = build_features(hidden, weather, horizon=30)
    window = (full.index > pd.Timestamp(origin)) & (
        full.index <= pd.Timestamp(origin) + pd.Timedelta(days=30)
    )
    np.testing.assert_allclose(full.loc[window, LAG_COLS].to_numpy(), masked.loc[window, LAG_COLS].to_numpy())


def test_horizon_longer_than_shortest_lag_is_rejected(inputs):
    daily, weather = inputs
    with pytest.raises(ValueError):
        build_features(daily_totals(daily), weather, horizon=31)


def test_holidays_flagged(inputs):
    daily, weather = inputs
    feats = build_features(daily_totals(daily), weather, horizon=30)
    assert feats.loc["2024-07-04", "is_holiday"] == 1
    assert feats.loc["2024-07-03", "day_before_holiday"] == 1


def test_a_day_missing_from_history_is_not_forecast(inputs):
    """A day with no published rides (e.g. a storm shutdown) is history with no label, not a forecast day."""
    daily, weather = inputs
    gap = date(2025, 2, 23)
    feats = build_features(daily_totals(daily.filter(pl.col("date") != gap)), weather, horizon=30)
    observed, future = split(feats)
    assert len(future) == 30 and future.index.min() > observed.index.max()
    assert pd.Timestamp(gap) not in future.index and pd.Timestamp(gap) not in observed.index
