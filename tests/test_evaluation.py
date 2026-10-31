import numpy as np
import pandas as pd

from citibike.evaluation import (
    apply_interval,
    conformal_bounds,
    interval_coverage,
    metrics,
    origins,
    split,
)


def test_metrics_known_values():
    m = metrics(np.array([100, 200]), np.array([110, 180]))
    assert m["MAE"] == 15
    assert abs(m["WAPE"] - 30 / 300) < 1e-12
    assert abs(m["Bias"] - (-10 / 300)) < 1e-12


def test_origins_tile_the_last_days_without_overlap():
    last = pd.Timestamp("2025-06-30")
    test = origins(last, 30, 6)
    tune = origins(last, 30, 3, skip=6)
    assert test[-1] == last - pd.Timedelta(days=30)
    assert np.all(np.diff(test) == pd.Timedelta(days=30))
    assert tune[-1] + pd.Timedelta(days=30) == test[0]  # validation ends where testing begins


def test_split_has_no_overlap():
    idx = pd.date_range("2025-01-01", periods=100, freq="D")
    feats = pd.DataFrame({"y": np.arange(100.0)}, index=idx)
    train, test = split(feats, idx[59], 30)
    assert train.index.max() < test.index.min()
    assert len(test) == 30


def test_conformal_intervals_reach_nominal_coverage():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(6):
        yhat = np.full(30, 1000.0)
        y = yhat * (1 + rng.normal(0, 0.1, 30))
        rows.append(pd.DataFrame({"origin": k, "y": y, "yhat": yhat}))
    preds = pd.concat(rows)
    cov = interval_coverage(preds, (0.8,)).iloc[0]["coverage"]
    assert 0.6 <= cov <= 1.0
    lo, hi = apply_interval(np.array([1000.0]), conformal_bounds(rng.normal(0, 0.1, 500), 0.9))
    assert lo[0] < 1000 < hi[0]
