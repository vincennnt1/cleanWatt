"""
advisor.py — CleanWatt inference and advisory logic.

No Streamlit imports. All functions take plain inputs and return plain data.
Called by app.py; can be wrapped in a FastAPI route later without changes.
"""

from __future__ import annotations

import logging
import pickle
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd

from cleanwatt.carbon import add_intensity_column
from cleanwatt.ingest import fetch_fuel_mix

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
CARBON_PATH = DATA_DIR / "processed" / "carbon_intensity.parquet"
FORECAST_LOG_PATH = DATA_DIR / "forecast_log.parquet"

HORIZONS = list(range(1, 25))

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "carbon_intensity_lag_24h", "carbon_intensity_lag_48h",
    "pct_wind", "pct_gas", "pct_hydro", "pct_nuclear", "pct_solar", "pct_biofuel",
    "temperature_2m", "wind_speed_10m",
    "wind_speed_lead_6h", "wind_speed_lead_12h", "wind_speed_lead_24h",
    "demand_mw",
]

APPLIANCE_PRESETS = [
    {"name": "Dryer", "hours": 1},
    {"name": "Dishwasher", "hours": 1.5},
    {"name": "EV overnight", "hours": 6},
]

_ONTARIO_DEMAND_FALLBACK = 13_500.0  # MW — approximate Ontario average


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> dict[int, object]:
    """Load all 24 LightGBM models from disk."""
    models = {}
    for h in HORIZONS:
        with open(MODELS_DIR / f"model_h{h}.pkl", "rb") as f:
            models[h] = pickle.load(f)
    return models


# ── Live data fetching ────────────────────────────────────────────────────────

def _fetch_current_fuel_mix() -> dict:
    """Return the most recent IESO fuel mix fractions and current carbon intensity."""
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(hours=6)).date().isoformat()
    end = (now + pd.Timedelta(days=1)).date().isoformat()
    raw = fetch_fuel_mix(start, end)
    raw = add_intensity_column(raw)
    row = raw.sort_values("interval_start_utc").iloc[-1]
    fuels = ["nuclear", "hydro", "wind", "solar", "gas", "biofuel", "other"]
    total_mw = max(sum(float(row.get(f, 0) or 0) for f in fuels), 1.0)
    return {
        "carbon_intensity": float(row["carbon_intensity"]),
        "timestamp": row["interval_start_utc"],
        "pct_wind": float(row.get("wind", 0) or 0) / total_mw,
        "pct_gas": float(row.get("gas", 0) or 0) / total_mw,
        "pct_hydro": float(row.get("hydro", 0) or 0) / total_mw,
        "pct_nuclear": float(row.get("nuclear", 0) or 0) / total_mw,
        "pct_solar": float(row.get("solar", 0) or 0) / total_mw,
        "pct_biofuel": float(row.get("biofuel", 0) or 0) / total_mw,
    }


def _fetch_weather_forecast(lat: float = 43.7, lon: float = -79.4) -> pd.DataFrame:
    """Fetch 3-day weather forecast from Open-Meteo, UTC-indexed."""
    client = openmeteo_requests.Client()
    response = client.weather_api("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m"],
        "timezone": "UTC",
        "forecast_days": 3,
    })[0]
    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    return pd.DataFrame({
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
    }, index=times)


def _fetch_demand_forecast() -> pd.Series:
    """
    Parse IESO Adequacy Report XML for the Ontario demand forecast.
    Returns an empty Series on any failure — caller uses fallback.
    """
    try:
        with urllib.request.urlopen(
            "https://reports-public.ieso.ca/public/Adequacy2/PUB_Adequacy2.xml",
            timeout=10,
        ) as resp:
            root = ET.parse(resp).getroot()
    except Exception as exc:
        logger.warning("Adequacy report fetch failed: %s", exc)
        return pd.Series(dtype=float)

    def _local_tag(elem: ET.Element) -> str:
        tag = elem.tag
        return tag.split("}")[-1] if "}" in tag else tag

    rows: list[tuple[pd.Timestamp, float]] = []
    for elem in root.iter():
        if _local_tag(elem) in {"HourlyForecast", "Hourly"}:
            children = {_local_tag(c): c.text for c in elem}
            hour = children.get("Hour") or children.get("HourNum")
            demand = children.get("OntarioDemand") or children.get("Demand")
            date = children.get("DeliveryDate") or children.get("Date")
            if hour and demand and date:
                try:
                    ts = (
                        pd.Timestamp(f"{date} {int(hour) - 1}:00", tz="America/Toronto")
                        .tz_convert("UTC")
                    )
                    rows.append((ts, float(demand)))
                except Exception:
                    continue

    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(list(vals), index=list(idx)).sort_index()


def _get_lags(now_utc: pd.Timestamp) -> dict[str, float]:
    """Read 24h and 48h carbon intensity lags from the historical parquet."""
    df = pd.read_parquet(CARBON_PATH).set_index("interval_start_utc")
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")

    def _lookup(target: pd.Timestamp) -> float:
        if target in df.index:
            return float(df.loc[target, "carbon_intensity"])
        nearby = df.index[abs(df.index - target) <= pd.Timedelta(hours=2)]
        if len(nearby):
            return float(df.loc[nearby[0], "carbon_intensity"])
        return float(df["carbon_intensity"].median())

    h = now_utc.floor("h")
    return {
        "carbon_intensity_lag_24h": _lookup(h - pd.Timedelta(hours=24)),
        "carbon_intensity_lag_48h": _lookup(h - pd.Timedelta(hours=48)),
    }


# ── Feature assembly ──────────────────────────────────────────────────────────

def fetch_live_features() -> tuple[pd.DataFrame, dict]:
    """
    Build the single feature row for the current hour using live data.
    Returns (feature_row_df, metadata).
    """
    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    now_ontario = now_utc.tz_convert("America/Toronto")

    calendar = {
        "hour_of_day": int(now_ontario.hour),
        "day_of_week": int(now_ontario.dayofweek),
        "month": int(now_ontario.month),
        "is_weekend": int(now_ontario.dayofweek >= 5),
    }

    lags = _get_lags(now_utc)
    fuel_mix = _fetch_current_fuel_mix()
    current_intensity = fuel_mix.pop("carbon_intensity")
    fuel_ts = fuel_mix.pop("timestamp")

    weather_df = _fetch_weather_forecast()

    def _weather_at(offset_h: int, col: str) -> float:
        target = now_utc + pd.Timedelta(hours=offset_h)
        idx = weather_df.index.get_indexer([target], method="nearest")[0]
        return float(weather_df[col].iloc[idx])

    weather_features = {
        "temperature_2m": _weather_at(0, "temperature_2m"),
        "wind_speed_10m": _weather_at(0, "wind_speed_10m"),
        "wind_speed_lead_6h": _weather_at(6, "wind_speed_10m"),
        "wind_speed_lead_12h": _weather_at(12, "wind_speed_10m"),
        "wind_speed_lead_24h": _weather_at(24, "wind_speed_10m"),
    }

    demand_series = _fetch_demand_forecast()
    if not demand_series.empty:
        idx = demand_series.index.get_indexer([now_utc], method="nearest")
        demand_mw = float(demand_series.iloc[idx[0]]) if len(idx) else _ONTARIO_DEMAND_FALLBACK
    else:
        demand_mw = _ONTARIO_DEMAND_FALLBACK
        logger.info("Using fallback demand_mw=%.0f MW", demand_mw)

    row = pd.DataFrame([{**calendar, **lags, **fuel_mix, **weather_features, "demand_mw": demand_mw}])

    meta = {
        "current_intensity": current_intensity,
        "fuel_timestamp": fuel_ts,
        "now_utc": now_utc,
        "now_ontario": now_ontario,
    }
    return row, meta


# ── Advisory functions (pure — no I/O) ───────────────────────────────────────

def forecast_24h(models: dict, feature_row: pd.DataFrame) -> list[float]:
    """Run all 24 horizon models. Returns predictions for t+1 … t+24."""
    return [float(models[h].predict(feature_row[FEATURE_COLS])[0]) for h in HORIZONS]


def best_window(forecast: list[float], window_hours: float) -> dict:
    """
    Find the lowest-carbon contiguous window of window_hours (rounded to int hours)
    within the next 24h forecast.

    forecast[0] = t+1, ..., forecast[23] = t+24.
    start_h / end_h are 1-indexed offsets from now (start_h=1 means next hour).
    """
    n = max(1, round(window_hours))
    n = min(n, len(forecast))
    best_i = min(
        range(len(forecast) - n + 1),
        key=lambda i: float(np.mean(forecast[i : i + n])),
    )
    return {
        "start_h": best_i + 1,
        "end_h": best_i + n,
        "avg_intensity": float(np.mean(forecast[best_i : best_i + n])),
    }


def classify_intensity(current: float, forecast: list[float]) -> str:
    """
    Return green / yellow / red based on where current falls in today's 24h range.
    Bottom third → green, middle → yellow, top → red.
    """
    all_vals = [current] + list(forecast)
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    if span < 1:
        return "green"
    third = span / 3
    if current <= lo + third:
        return "green"
    elif current <= lo + 2 * third:
        return "yellow"
    return "red"


def get_forecast(models: dict) -> dict:
    """
    Main entry point for the dashboard.
    Returns all data the UI needs as a plain dict.
    """
    feature_row, meta = fetch_live_features()
    predictions = forecast_24h(models, feature_row)
    current = meta["current_intensity"]

    return {
        "current_intensity": current,
        "status": classify_intensity(current, predictions),
        "predictions": predictions,
        "windows": {p["name"]: best_window(predictions, p["hours"]) for p in APPLIANCE_PRESETS},
        "now_utc": meta["now_utc"],
        "now_ontario": meta["now_ontario"],
        "fuel_timestamp": meta["fuel_timestamp"],
    }
