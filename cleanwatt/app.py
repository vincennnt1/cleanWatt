"""
CleanWatt — Ontario grid carbon intensity dashboard.
Phase 3: Streamlit UI. All logic lives in advisor.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cleanwatt.advisor import APPLIANCE_PRESETS, best_window, get_forecast, load_models

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
FORECAST_LOG_PATH = DATA_DIR / "forecast_log.parquet"
METRICS_PATH = MODELS_DIR / "eval_metrics.json"

st.set_page_config(page_title="CleanWatt", page_icon="⚡", layout="wide")


@st.cache_resource
def _models():
    return load_models()


@st.cache_data(ttl=300)
def _forecast():
    return get_forecast(_models())


STATUS_COLOR = {"green": "#27ae60", "yellow": "#f39c12", "red": "#c0392b"}
STATUS_LABEL = {
    "green": "Clean grid — great time to run appliances",
    "yellow": "Moderate — average mix right now",
    "red": "High carbon — consider waiting",
}


def _hour_label(h_offset: int, now_ontario: pd.Timestamp) -> str:
    """Convert a +h offset from now into a readable Ontario local time string."""
    t = now_ontario + pd.Timedelta(hours=h_offset)
    h = t.hour
    if h == 0:
        return "12am"
    elif h < 12:
        return f"{h}am"
    elif h == 12:
        return "12pm"
    else:
        return f"{h - 12}pm"


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ CleanWatt")
st.caption("Ontario grid carbon intensity · Is now a good time to run it?")

tab_forecast, tab_accuracy = st.tabs(["Forecast", "Model Accuracy"])


# ── Tab 1: Forecast ───────────────────────────────────────────────────────────
with tab_forecast:
    with st.spinner("Fetching live Ontario grid data..."):
        try:
            data = _forecast()
            load_error = None
        except Exception as exc:
            load_error = str(exc)
            data = None

    if load_error or data is None:
        st.error(f"Could not load live data: {load_error}")
        st.stop()

    current: float = data["current_intensity"]
    status: str = data["status"]
    predictions: list[float] = data["predictions"]
    now_ontario: pd.Timestamp = data["now_ontario"]
    fuel_ts = data["fuel_timestamp"]

    # ── Current intensity row ─────────────────────────────────────────────────
    col_metric, col_badge, col_range = st.columns([1, 2, 1])

    with col_metric:
        st.metric(
            label="Right now",
            value=f"{current:.0f} gCO₂/kWh",
            help="Carbon intensity of Ontario's grid this hour, from IESO fuel mix data.",
        )

    with col_badge:
        color = STATUS_COLOR[status]
        label = STATUS_LABEL[status]
        st.markdown(
            f'<div style="background:{color};padding:18px 12px;border-radius:8px;'
            f'text-align:center;color:white;font-weight:600;font-size:1.05rem;">'
            f"{label}</div>",
            unsafe_allow_html=True,
        )

    with col_range:
        lo, hi = min(predictions), max(predictions)
        st.metric(
            "24h forecast range",
            f"{lo:.0f}–{hi:.0f} gCO₂/kWh",
            help="Min and max predicted carbon intensity over the next 24 hours.",
        )

    st.divider()

    # ── 24h forecast chart ────────────────────────────────────────────────────
    all_vals = [current] + predictions
    lo_all, hi_all = min(all_vals), max(all_vals)
    span = hi_all - lo_all
    green_threshold = lo_all + span / 3 if span >= 1 else hi_all
    yellow_threshold = lo_all + 2 * span / 3 if span >= 1 else hi_all

    chart_df = pd.DataFrame({
        "Hour": list(range(1, 25)),
        "gCO₂/kWh": predictions,
        "Time (Ontario)": [_hour_label(h, now_ontario) for h in range(1, 25)],
        "status": [
            "green" if v <= green_threshold else ("yellow" if v <= yellow_threshold else "red")
            for v in predictions
        ],
    })

    line = (
        alt.Chart(chart_df)
        .mark_line(color="#555555", strokeWidth=2)
        .encode(
            x=alt.X("Hour:Q", title="Hours from now", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y(
                "gCO₂/kWh:Q",
                title="Carbon intensity (gCO₂/kWh)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=["Time (Ontario):N", "gCO₂/kWh:Q"],
        )
    )
    points = (
        alt.Chart(chart_df)
        .mark_circle(size=90)
        .encode(
            x="Hour:Q",
            y="gCO₂/kWh:Q",
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["green", "yellow", "red"],
                    range=["#27ae60", "#f39c12", "#c0392b"],
                ),
                legend=alt.Legend(title="Grid status"),
            ),
            tooltip=["Time (Ontario):N", "gCO₂/kWh:Q", "status:N"],
        )
    )

    st.altair_chart(
        (line + points)
        .properties(title="24-hour carbon intensity forecast", height=320)
        .interactive(),
        use_container_width=True,
    )
    st.caption(
        "⚠️ Calculated from Ontario generation only; cross-border imports are excluded. "
        "Marginal vs. average carbon not distinguished (v1)."
    )

    st.divider()

    # ── Best windows ──────────────────────────────────────────────────────────
    st.subheader("Best windows to run appliances")

    preset_cols = st.columns(len(APPLIANCE_PRESETS) + 1)

    for i, preset in enumerate(APPLIANCE_PRESETS):
        w = data["windows"][preset["name"]]
        start_label = _hour_label(w["start_h"], now_ontario)
        end_label = _hour_label(w["end_h"] + 1, now_ontario)
        avg_ci = w["avg_intensity"]
        pct_delta = (current - avg_ci) / current * 100 if current > 0 else 0.0

        with preset_cols[i]:
            st.markdown(f"**{preset['name']}** ({preset['hours']:.4g}h)")
            st.metric(
                label=f"{start_label} – {end_label}",
                value=f"{avg_ci:.0f} gCO₂/kWh",
                delta=f"{pct_delta:+.0f}% vs now" if abs(pct_delta) > 0.5 else None,
                delta_color="inverse",
            )

    with preset_cols[-1]:
        st.markdown("**Custom**")
        custom_h = st.number_input(
            "Window (hours)", min_value=1, max_value=12, value=2, step=1, key="custom_window"
        )
        cw = best_window(predictions, float(custom_h))
        st.metric(
            label=f"{_hour_label(cw['start_h'], now_ontario)} – {_hour_label(cw['end_h'] + 1, now_ontario)}",
            value=f"{cw['avg_intensity']:.0f} gCO₂/kWh",
        )

    fuel_time_label = (
        pd.Timestamp(fuel_ts).tz_convert("America/Toronto").strftime("%I:%M %p ET").lstrip("0")
    )
    st.caption(f"Fuel mix data as of {fuel_time_label}. Forecast refreshes every 5 min.")


# ── Tab 2: Model Accuracy ─────────────────────────────────────────────────────
with tab_accuracy:
    st.subheader("Backtest metrics (walk-forward validation)")
    st.caption(
        "Expanding-window backtest, monthly folds 2023-07 → 2025-11. "
        "Baseline = 'same hour yesterday'."
    )

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        metrics_df = pd.DataFrame(
            [
                {
                    "Horizon": int(h),
                    "MAE (gCO₂/kWh)": v["mae_model"],
                    "Skill vs. yesterday": v["skill"],
                }
                for h, v in metrics.items()
            ]
        ).set_index("Horizon")

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            mae_chart = (
                alt.Chart(metrics_df.reset_index())
                .mark_line(point=True, color="#2980b9")
                .encode(
                    x=alt.X("Horizon:Q", title="Forecast horizon (hours ahead)"),
                    y=alt.Y(
                        "MAE (gCO₂/kWh):Q",
                        title="MAE (gCO₂/kWh)",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=["Horizon:Q", "MAE (gCO₂/kWh):Q", "Skill vs. yesterday:Q"],
                )
                .properties(title="Error by forecast horizon", height=280)
            )
            skill_chart = (
                alt.Chart(metrics_df.reset_index())
                .mark_line(point=True, color="#27ae60")
                .encode(
                    x=alt.X("Horizon:Q", title="Forecast horizon (hours ahead)"),
                    y=alt.Y(
                        "Skill vs. yesterday:Q",
                        title="Skill score (↑ better)",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=["Horizon:Q", "MAE (gCO₂/kWh):Q", "Skill vs. yesterday:Q"],
                )
                .properties(title="Skill vs. yesterday baseline", height=280)
            )
            st.altair_chart(mae_chart, use_container_width=True)
            st.altair_chart(skill_chart, use_container_width=True)

        with col_table:
            st.dataframe(
                metrics_df.style.format(
                    {"MAE (gCO₂/kWh)": "{:.2f}", "Skill vs. yesterday": "{:+.3f}"}
                ),
                height=480,
            )
            mean_skill = metrics_df["Skill vs. yesterday"].mean()
            st.metric(
                "Mean skill score",
                f"{mean_skill:+.3f}",
                help="Positive = better than 'same hour yesterday'. 0 = no better than naive baseline.",
            )
    else:
        st.info(
            "Backtest metrics not yet generated. "
            "Run `uv run python -m cleanwatt.model` to produce `models/eval_metrics.json`."
        )

    st.divider()
    st.subheader("Predicted vs. actual (live log)")

    if FORECAST_LOG_PATH.exists():
        log_df = pd.read_parquet(FORECAST_LOG_PATH)
        log_df["abs_error"] = (log_df["predicted_gco2"] - log_df["actual_gco2"]).abs()
        log_df["date"] = pd.to_datetime(log_df["forecast_made_at"]).dt.date
        rolling = (
            log_df.groupby("date")["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"date": "Date", "abs_error": "Daily MAE (gCO₂/kWh)"})
        )
        st.altair_chart(
            alt.Chart(rolling)
            .mark_line(point=True, color="#8e44ad")
            .encode(
                x="Date:T",
                y=alt.Y("Daily MAE (gCO₂/kWh):Q", scale=alt.Scale(zero=False)),
                tooltip=["Date:T", "Daily MAE (gCO₂/kWh):Q"],
            )
            .properties(title="Rolling daily MAE (live predictions)", height=260),
            use_container_width=True,
        )
        st.caption(f"{len(log_df):,} forecast records in log.")
    else:
        st.info(
            "No live forecast log yet. "
            "This will populate once the daily cron job (Phase 4) runs."
        )
