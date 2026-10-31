"""Builds the static dashboard (site/index.html) that GitHub Pages publishes."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs_version

from .evaluation import metrics

# Reference palette (validated categorical order) + chart chrome
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
REPO_URL = "https://github.com/davidskaff/citibike-demand-forecasting"


def _style(fig: go.Figure, title: str = "", height: int = 380, y_title: str = "", **layout) -> go.Figure:
    base = dict(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height,
        margin=dict(l=60, r=20, t=80 if title else 30, b=40),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**{**base, **layout})
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=AXIS,
        tickfont=dict(color=MUTED),
        zeroline=False,
        title=dict(text=y_title, font=dict(color=MUTED)),
        separatethousands=True,
    )
    return fig


def _div(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True}
    )


# --------------------------------------------------------------------------- figures
def fig_forecast(totals: pd.Series, forecast: pd.DataFrame) -> go.Figure:
    hist = totals.tail(180)
    fig = go.Figure()
    fc = forecast.set_index("date")
    x_band = list(fc.index) + list(fc.index[::-1])
    for lvl, alpha in (("95", 0.12), ("80", 0.22)):
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=list(fc[f"hi_{lvl}"]) + list(fc[f"lo_{lvl}"][::-1]),
                fill="toself",
                fillcolor=f"rgba(235,104,52,{alpha})",
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{lvl}% interval",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            name="Actual rides",
            line=dict(color=BLUE, width=2),
            hovertemplate="%{y:,.0f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=fc["yhat"],
            name="Forecast",
            line=dict(color=ORANGE, width=2),
            hovertemplate="%{y:,.0f}",
        )
    )
    clim = fc[fc["weather_source"] == "climatology"]
    if len(clim):
        fig.add_vrect(
            x0=clim.index.min(),
            x1=clim.index.max() + pd.Timedelta(days=1),
            fillcolor=GRID,
            opacity=0.35,
            line_width=0,
            annotation_text="weather = seasonal average",
            annotation_position="top left",
            annotation_font=dict(color=MUTED, size=11),
        )
    return _style(fig, height=420, y_title="Rides per day")


def fig_model_comparison(summary: pd.DataFrame, champion: str) -> go.Figure:
    s = summary.sort_values("WAPE", ascending=False)
    colors = [
        ORANGE if m == champion else (BLUE if ("LightGBM" in m or "XGBoost" in m) else AQUA)
        for m in s["model"]
    ]
    fig = go.Figure(
        go.Bar(
            x=s["WAPE"] * 100,
            y=s["model"],
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:.1%}" for v in s["WAPE"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: WAPE %{x:.2f}%<extra></extra>",
        )
    )
    _style(fig, height=320, hovermode="closest", showlegend=False)
    fig.update_xaxes(
        title=dict(text="Backtest WAPE (lower is better), %", font=dict(color=MUTED)),
        showgrid=True,
        gridcolor=GRID,
    )
    fig.update_yaxes(showgrid=False)
    return fig


def fig_backtest(preds: pd.DataFrame, champion: str, baseline: str) -> go.Figure:
    fig = go.Figure()
    actual = preds[preds["model"] == champion].set_index("date")["y"]
    fig.add_trace(
        go.Scatter(
            x=actual.index, y=actual, name="Actual", line=dict(color=BLUE, width=2), hovertemplate="%{y:,.0f}"
        )
    )
    for name, color, dash in ((champion, ORANGE, "solid"), (baseline, AQUA, "dot")):
        p = preds[preds["model"] == name]
        for i, (_, g) in enumerate(p.groupby("origin")):
            fig.add_trace(
                go.Scatter(
                    x=g["date"],
                    y=g["yhat"],
                    name=name,
                    legendgroup=name,
                    showlegend=i == 0,
                    line=dict(color=color, width=2, dash=dash),
                    hovertemplate="%{y:,.0f}",
                )
            )
    for origin in sorted(preds["origin"].unique()):
        fig.add_vline(x=origin, line=dict(color=AXIS, width=1, dash="dash"))
    return _style(fig, height=400, y_title="Rides per day")


def fig_error_by_horizon(preds: pd.DataFrame, champion: str, baseline: str) -> go.Figure:
    fig = go.Figure()
    for name, color in ((champion, ORANGE), (baseline, AQUA)):
        p = preds[preds["model"] == name]
        w = p.groupby("h").apply(lambda g: metrics(g["y"], g["yhat"])["WAPE"] * 100, include_groups=False)
        fig.add_trace(
            go.Scatter(
                x=w.index,
                y=w.values,
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate="%{y:.1f}%",
            )
        )
    _style(fig, "Error by days ahead", height=340, y_title="WAPE, %")
    fig.update_xaxes(title=dict(text="Days ahead", font=dict(color=MUTED)))
    return fig


def fig_shap_importance(importance: pd.DataFrame, top: int = 12) -> go.Figure:
    imp = importance.head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=imp["mean_abs_shap"],
            y=imp["label"],
            orientation="h",
            marker=dict(color=BLUE, cornerradius=4),
            hovertemplate="%{y}: ±%{x:,.0f} rides/day on average<extra></extra>",
        )
    )
    _style(fig, "Most important features", height=440, hovermode="closest", showlegend=False)
    fig.update_xaxes(
        title=dict(text="Avg impact on forecast, rides/day (mean |SHAP|)", font=dict(color=MUTED)),
        showgrid=True,
        gridcolor=GRID,
    )
    fig.update_yaxes(showgrid=False)
    return fig


def fig_shap_temperature(shap_values: pd.DataFrame, features: pd.DataFrame) -> go.Figure:
    rainy = features["precip_mm"] >= 2
    fig = go.Figure()
    for mask, name, color in ((~rainy, "Dry day", BLUE), (rainy, "Rainy day (2mm+)", ORANGE)):
        fig.add_trace(
            go.Scatter(
                x=features.loc[mask, "temp_mean"],
                y=shap_values.loc[mask, "temp_mean"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=8, opacity=0.75, line=dict(color=SURFACE, width=1)),
                hovertemplate="%{x:.1f}°C → %{y:+,.0f} rides<extra></extra>",
            )
        )
    _style(
        fig,
        "Temperature effect (last 365 days)",
        height=440,
        hovermode="closest",
        y_title="Effect on forecast (rides/day)",
    )
    fig.update_xaxes(title=dict(text="Mean daily temperature (°C)", font=dict(color=MUTED)))
    return fig


def fig_history(totals: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=totals.index,
            y=totals.values,
            name="Daily rides",
            line=dict(color="rgba(42,120,214,0.35)", width=1),
            hovertemplate="%{y:,.0f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=totals.index,
            y=totals.rolling(28, min_periods=7).mean(),
            name="28-day average",
            line=dict(color=BLUE, width=2),
            hovertemplate="%{y:,.0f}",
        )
    )
    return _style(fig, "Daily rides", height=380, y_title="Rides per day")


def fig_rider_mix(daily: pl.DataFrame) -> go.Figure:
    monthly = (
        daily.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month", "member_casual")
        .agg(pl.col("trips").sum())
        .sort("month")
        .to_pandas()
    )
    fig = go.Figure()
    for kind, color, label in (("member", BLUE, "Members"), ("casual", ORANGE, "Casual riders")):
        m = monthly[monthly["member_casual"] == kind]
        fig.add_trace(
            go.Bar(
                x=m["month"], y=m["trips"], name=label, marker=dict(color=color), hovertemplate="%{y:,.0f}"
            )
        )
    _style(
        fig, "Members vs casual riders", height=380, y_title="Rides per month", barmode="stack", bargap=0.15
    )
    fig.update_traces(marker_line=dict(color=SURFACE, width=1))
    return fig


def fig_ebike_share(daily: pl.DataFrame) -> go.Figure:
    m = (
        daily.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            (
                pl.col("trips").filter(pl.col("rideable_type").str.contains("electric")).sum()
                / pl.col("trips").sum()
                * 100
            ).alias("share")
        )
        .sort("month")
        .to_pandas()
    )
    fig = go.Figure(
        go.Scatter(
            x=m["month"],
            y=m["share"],
            mode="lines+markers",
            line=dict(color=VIOLET, width=2),
            marker=dict(size=8),
            name="E-bike share",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    _style(fig, "Share of rides on e-bikes", height=380, y_title="% of rides", showlegend=False)
    return fig


def fig_weather_scatter(totals: pd.Series, weather: pl.DataFrame) -> go.Figure:
    w = weather.to_pandas().set_index("date")
    w.index = pd.to_datetime(w.index)
    df = w.join(totals.rename("y"), how="inner").dropna(subset=["y"])
    df = df[df.index.dayofweek < 5]
    rainy = df["precip_mm"] >= 2
    fig = go.Figure()
    for mask, name, color in ((~rainy, "Dry weekday", BLUE), (rainy, "Rainy weekday (2mm+)", ORANGE)):
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "temp_mean"],
                y=df.loc[mask, "y"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=8, opacity=0.7, line=dict(color=SURFACE, width=1)),
                hovertemplate="%{x:.1f}°C: %{y:,.0f} rides<extra></extra>",
            )
        )
    _style(fig, "Temperature vs rides (weekdays)", height=380, hovermode="closest", y_title="Rides per day")
    fig.update_xaxes(title=dict(text="Mean daily temperature (°C)", font=dict(color=MUTED)))
    return fig


def fig_tracking(tracking: pd.DataFrame, totals: pd.Series) -> go.Figure:
    fig = go.Figure()
    start = tracking["date"].min() - pd.Timedelta(days=14)
    act = totals[totals.index >= start]
    fig.add_trace(
        go.Scatter(
            x=act.index,
            y=act.values,
            name="Actual rides",
            line=dict(color=BLUE, width=2),
            hovertemplate="%{y:,.0f}",
        )
    )
    for i, (_origin, g) in enumerate(tracking.groupby("origin")):
        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["yhat"],
                name="Published forecast",
                legendgroup="fc",
                showlegend=i == 0,
                line=dict(color=ORANGE, width=2),
                hovertemplate="%{y:,.0f}",
            )
        )
    return _style(fig, height=360, y_title="Rides per day")


# --------------------------------------------------------------------------- page
def _table(df: pd.DataFrame, champion: str) -> str:
    rows = "".join(
        f"<tr class='{'champ' if r.model == champion else ''}'><td>{html.escape(r.model)}"
        f"{' <span class=badge>best</span>' if r.model == champion else ''}</td>"
        f"<td>{r.WAPE:.1%}</td><td>{r.MAE:,.0f}</td><td>{r.RMSE:,.0f}</td><td>{r.Bias:+.1%}</td>"
        f"<td>{r.WAPE_fold_std:.1%}</td></tr>"
        for r in df.itertuples(index=False)
    )
    return (
        "<table><thead><tr><th>Model</th><th>WAPE</th><th>MAE</th><th>RMSE</th><th>Bias</th>"
        f"<th>WAPE spread across folds</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _card(label: str, value: str, note: str = "") -> str:
    return f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>"


def build_site(
    site_dir: Path,
    *,
    daily,
    weather,
    totals,
    preds,
    summary,
    coverage,
    forecast,
    shap_importance,
    shap_values,
    features,
    tracking,
    run_info,
) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    tot = totals.to_pandas().set_index("date")["y"].astype(float)
    tot.index = pd.to_datetime(tot.index)
    forecast = forecast.assign(date=pd.to_datetime(forecast["date"]))
    champion = run_info["champion"]
    baselines = summary[~summary["model"].str.contains("LightGBM|XGBoost")]
    best_baseline = baselines.iloc[0]["model"]
    champ_row = summary.iloc[0]
    base_row = baselines.iloc[0]
    improvement = 1 - champ_row["WAPE"] / base_row["WAPE"]

    last30 = tot.tail(30).sum()
    prior_year = tot[
        (tot.index > tot.index.max() - pd.Timedelta(days=395))
        & (tot.index <= tot.index.max() - pd.Timedelta(days=365))
    ].sum()
    yoy = f"{last30 / prior_year - 1:+.1%} vs same 30 days last year" if prior_year else ""
    cov80 = coverage.loc[np.isclose(coverage["level"], 0.8), "coverage"]
    cov_txt = f"{cov80.iloc[0]:.0%}" if len(cov80) else "n/a"
    n_clim = int((forecast["weather_source"] == "climatology").sum())

    cards = "".join(
        [
            _card(
                "Data through",
                pd.Timestamp(run_info["data_end"]).strftime("%b %d, %Y"),
                f"{run_info['total_rides']:,} rides since {pd.Timestamp(run_info['data_start']).strftime('%b %Y')}",
            ),
            _card("Rides, last 30 days", f"{last30:,.0f}", yoy),
            _card(
                "Next 30 days (forecast)",
                f"{forecast['yhat'].sum():,.0f}",
                f"80% range {forecast['lo_80'].sum():,.0f}–{forecast['hi_80'].sum():,.0f}",
            ),
            _card(
                "Best model",
                html.escape(champion),
                f"WAPE {champ_row['WAPE']:.1%}, {improvement:.0%} better than {html.escape(best_baseline)}",
            ),
            _card("80% interval coverage", cov_txt, "on the held-out backtest month"),
        ]
    )

    if len(tracking):
        tr = metrics(tracking["y"], tracking["yhat"])
        tracking_html = (
            f"<p>Each monthly run saves its forecast in <code>forecasts/</code>. Once Citi Bike publishes the real "
            f"numbers, they are scored here. This is true out-of-sample accuracy, using the weather that was "
            f"forecast at the time. So far: <b>WAPE {tr['WAPE']:.1%}</b> over {len(tracking)} days.</p>"
            + _div(fig_tracking(tracking, tot))
        )
    else:
        tracking_html = (
            "<p class='muted'>Starts after the next monthly run: this run's forecast is saved in "
            "<code>forecasts/</code> and will be scored against the real rides once Citi Bike "
            "publishes them.</p>"
        )

    sample_banner = (
        "<div class='banner'>Demo build on synthetic sample data. The published site uses real Citi Bike data.</div>"
        if run_info.get("sample_mode")
        else ""
    )
    clim_note = (
        f" The last {n_clim} days fall beyond Open-Meteo's 16-day weather forecast, so they use the "
        f"seasonal-average weather for those dates (shaded)."
        if n_clim
        else ""
    )

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Citi Bike Demand Forecast</title>
<meta name="description" content="Daily NYC Citi Bike ridership forecast with weather, backtested models and conformal intervals.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚲</text></svg>">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>
  :root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --grid:{GRID}; --surface:{SURFACE}; --page:#f9f9f7;
          --accent:{ORANGE}; --blue:{BLUE}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--page); color:var(--ink); font-family:{FONT}; line-height:1.55; }}
  header {{ background:#0d366b; color:#fff; padding:40px 16px 32px; }}
  header .wrap, main {{ max-width:1120px; margin:0 auto; }}
  header h1 {{ margin:0 0 6px; font-size:clamp(24px,4vw,34px); letter-spacing:-0.01em; }}
  header p {{ margin:0; color:#cde2fb; max-width:760px; }}
  header a {{ color:#fff; }}
  main {{ padding:24px 16px 64px; }}
  .banner {{ background:#fff4d6; border:1px solid #eda100; padding:10px 14px; border-radius:8px; margin-bottom:16px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:28px; }}
  .card {{ background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:14px 16px; }}
  .card .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .card .value {{ font-size:24px; font-weight:650; margin:4px 0 2px; }}
  .card .note {{ font-size:12.5px; color:var(--ink2); }}
  section {{ background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:20px 20px 8px;
             margin-bottom:20px; }}
  section h2 {{ margin:0 0 4px; font-size:19px; }}
  section > p {{ margin:4px 0 12px; color:var(--ink2); max-width:860px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); gap:8px 20px; }}
  .grid2 > * {{ min-width:0; }}
  h3 {{ font-size:15px; margin:12px 0 2px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; margin:8px 0 16px; }}
  th, td {{ text-align:right; padding:8px 10px; border-bottom:1px solid var(--grid); }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  tr.champ td {{ font-weight:650; }}
  .badge {{ background:var(--accent); color:#fff; border-radius:999px; padding:1px 8px; font-size:11px; margin-left:6px; }}
  .muted {{ color:var(--muted); }}
  code {{ background:#f0efec; padding:1px 5px; border-radius:4px; font-size:.92em; }}
  ul.method li {{ margin-bottom:6px; color:var(--ink2); }}
  .tablewrap {{ overflow-x:auto; }}
  footer {{ text-align:center; color:var(--muted); font-size:13px; padding:0 16px 40px; }}
</style></head>
<body>
<header><div class="wrap">
  <h1>NYC Citi Bike Demand Forecast</h1>
  <p>Daily ridership for New York's bike-share system, forecast {run_info["horizon"]} days ahead from
  public trip data and weather. Retrained automatically every month with GitHub Actions.
  <a href="{REPO_URL}">Code on GitHub</a></p>
</div></header>
<main>
{sample_banner}
<div class="cards">{cards}</div>

<section><h2>Forecast for the next {run_info["horizon"]} days</h2>
<p>Forecast from the best backtested model, with conformal prediction intervals calibrated on its past errors.
Weather for the forecast window comes from Open-Meteo's forecast.{clim_note}</p>
{_div(fig_forecast(tot, forecast))}</section>

<section><h2>Which model wins?</h2>
<p>Every model was refit and scored on the same {run_info["n_folds"]} rolling {
        run_info["horizon"]
    }-day windows
(the most recent {run_info["n_folds"] * run_info["horizon"]} days). WAPE = total absolute error ÷ total rides.
LightGBM's settings were tuned with Optuna on earlier validation windows, so the test windows stay unseen.</p>
<div class="grid2">{_div(fig_model_comparison(summary, champion))}
<div class="tablewrap">{_table(summary, champion)}</div></div></section>

<section><h2>Backtest: forecasts vs what actually happened</h2>
<p>Each dashed line is a forecast origin: the model only knew rides up to that day, then forecast the next
{run_info["horizon"]} days. The best statistical baseline is shown for comparison.</p>
{_div(fig_backtest(preds, champion, best_baseline))}
<div class="grid2">{_div(fig_error_by_horizon(preds, champion, best_baseline))}
<div><h3>Conformal interval coverage</h3><p class="muted">Intervals calibrated on the earlier folds, then checked
on the latest held-out month. Coverage close to the target means the uncertainty is honest.</p>
<table><thead><tr><th>Interval</th><th>Target</th><th>Actual coverage</th><th>Days</th></tr></thead><tbody>
{
        "".join(
            f"<tr><td>{int(c.level * 100)}%</td><td>{c.level:.0%}</td><td>{c.coverage:.0%}</td><td>{c.n_days}</td></tr>"
            for c in coverage.itertuples()
        )
    }
</tbody></table></div></div></section>

<section><h2>What drives demand? (SHAP)</h2>
<p>SHAP values break each forecast into contributions from every feature, measured in rides per day.
The right chart shows how temperature alone moves the forecast, split by dry and rainy days.</p>
<div class="grid2">{_div(fig_shap_importance(shap_importance))}{
        _div(fig_shap_temperature(shap_values, features))
    }</div>
</section>

<section><h2>Ridership history</h2>
<div class="grid2">{_div(fig_history(tot))}{_div(fig_weather_scatter(tot, weather))}</div>
<div class="grid2">{_div(fig_rider_mix(daily))}{_div(fig_ebike_share(daily))}</div></section>

<section><h2>Live accuracy tracking</h2>{tracking_html}</section>

<section><h2>How it works</h2><ul class="method">
<li><b>Data:</b> every public Citi Bike trip file since {
        pd.Timestamp(run_info["data_start"]).strftime("%B %Y")
    },
condensed to daily counts with DuckDB (tens of millions of rows), plus daily weather from the Open-Meteo archive.</li>
<li><b>Features (Polars):</b> day of week, holidays, seasonal cycles, long-term trend, temperature, rain, snow and
wind, plus lagged demand at least {run_info["horizon"]} days old so nothing leaks from the future.</li>
<li><b>Models:</b> seasonal naive, AutoETS and AutoARIMA (statsforecast) vs LightGBM and XGBoost.</li>
<li><b>Evaluation:</b> rolling-origin backtesting; Optuna tuning on separate, earlier validation windows; runs logged
to MLflow.</li>
<li><b>Uncertainty:</b> split-conformal prediction intervals from relative backtest errors.</li>
<li><b>Caveat:</b> backtests use the weather that actually happened, which flatters every model slightly. The
live-tracking section measures accuracy with the weather that was forecast at the time.</li>
<li><b>Automation:</b> GitHub Actions reruns everything monthly after Citi Bike publishes new data, tests the code
with pytest, and redeploys this page.</li></ul></section>
</main>
<script>
  // Plotly sizes charts before the grid settles; resize once layout is final.
  window.addEventListener("load", () => document.querySelectorAll(".plotly-graph-div").forEach(d => Plotly.Plots.resize(d)));
</script>
<footer>Last updated {run_info["run_utc"]} · Data: Citi Bike System Data (Lyft), Open-Meteo.com (CC BY 4.0) ·
<a href="{REPO_URL}">Source code</a></footer>
</body></html>"""
    out = site_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (site_dir / ".nojekyll").touch()
    return out
