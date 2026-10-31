"""Forecasting models behind one interface: fit on history up to an origin, predict the next `h` days.

* Statistical baselines (statsforecast): seasonal naive, AutoETS, AutoARIMA. They only see the demand
  series itself (weekly seasonality).
* Machine-learning models (LightGBM, XGBoost): one direct multi-horizon regressor on calendar, holiday,
  weather and lag features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

from .features import FEATURE_COLS


class Forecaster:
    name: str = "base"
    kind: str = "baseline"

    def fit(self, train: pd.DataFrame) -> Forecaster:  # train: feature table rows with y
        raise NotImplementedError

    def predict(self, future: pd.DataFrame) -> np.ndarray:  # future: feature table rows to forecast
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class StatsForecaster(Forecaster):
    kind = "statistical baseline"

    def __init__(self, name: str, model_factory):
        self.name = name
        self._factory = model_factory

    def fit(self, train: pd.DataFrame) -> StatsForecaster:
        y = train["y"].astype(float)
        y = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq="D")).interpolate(
            limit_direction="both"
        )
        self._df = pd.DataFrame({"unique_id": "citibike", "ds": y.index, "y": y.to_numpy()})
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        h = len(future)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(models=[self._factory()], freq="D", n_jobs=1)
            fc = sf.forecast(df=self._df, h=h)
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        return np.clip(fc[col].to_numpy(dtype=float), 0, None)


@dataclass
class LightGBMForecaster(Forecaster):
    hyperparams: dict = field(default_factory=dict)
    seed: int = 42
    name: str = "LightGBM"
    kind: str = "machine learning"

    def _model(self):
        defaults = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )
        return lgb.LGBMRegressor(**{**defaults, **self.hyperparams}, random_state=self.seed, verbose=-1)

    def fit(self, train: pd.DataFrame) -> LightGBMForecaster:
        self.model = self._model().fit(train[FEATURE_COLS], train["y"])
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(future[FEATURE_COLS]), 0, None)

    def params(self) -> dict:
        return self._model().get_params()


@dataclass
class XGBoostForecaster(Forecaster):
    seed: int = 42
    name: str = "XGBoost"
    kind: str = "machine learning"

    def _model(self):
        return xgb.XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            random_state=self.seed,
            n_jobs=2,
        )

    def fit(self, train: pd.DataFrame) -> XGBoostForecaster:
        self.model = self._model().fit(train[FEATURE_COLS], train["y"])
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(future[FEATURE_COLS]), 0, None)

    def params(self) -> dict:
        p = self._model().get_params()
        return {k: v for k, v in p.items() if v is not None and not callable(v)}


def baseline_models() -> list[Forecaster]:
    return [
        StatsForecaster("Seasonal naive", lambda: SeasonalNaive(season_length=7)),
        StatsForecaster("AutoETS", lambda: AutoETS(season_length=7)),
        StatsForecaster("AutoARIMA", lambda: AutoARIMA(season_length=7)),
    ]
