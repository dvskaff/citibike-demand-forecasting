"""SHAP explanations for the tree model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import FEATURE_COLS

PRETTY = {
    "temp_mean": "Mean temperature",
    "temp_max": "Max temperature",
    "temp_min": "Min temperature",
    "precip_mm": "Rain (mm)",
    "precip_hours": "Hours of rain",
    "snow_cm": "Snow (cm)",
    "wind_max": "Max wind",
    "dow": "Day of week",
    "is_weekend": "Weekend",
    "month": "Month",
    "week_of_year": "Week of year",
    "trend": "Long-term trend",
    "is_holiday": "Holiday",
    "day_before_holiday": "Day before holiday",
    "day_after_holiday": "Day after holiday",
    "lag_30": "Rides 30 days earlier",
    "lag_35": "Rides 35 days earlier",
    "lag_364": "Rides same weekday last year",
    "roll_mean_7_lag30": "7-day avg, a month earlier",
    "roll_mean_28_lag30": "28-day avg, a month earlier",
}


def pretty(col: str) -> str:
    if col.startswith(("sin_", "cos_")):
        return f"Seasonal cycle ({col})"
    return PRETTY.get(col, col)


def shap_summary(model, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean |SHAP| per feature, plus per-day SHAP values for the dependence plots."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X[FEATURE_COLS])
    sv = pd.DataFrame(values, columns=FEATURE_COLS, index=X.index)
    importance = (
        pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .assign(label=lambda d: d["feature"].map(pretty))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance, sv
