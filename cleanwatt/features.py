from __future__ import annotations

import logging

import openmeteo_requests
import pandas as pd

# features.py — builds the feature matrix fed into model.py
#
# INPUT:  data/processed/carbon_intensity.parquet  (from ingest.py)
# OUTPUT: a DataFrame ready for LightGBM — one row per hour, columns below
#
# ─── FEATURE GROUPS ───────────────────────────────────────────────────────────
#
# 1. CALENDAR
#    hour_of_day (0–23), day_of_week (0–6), month (1–12), is_weekend (bool)
#    → derive from interval_start_utc, but convert to Ontario local time first
#      (America/Toronto) so "hour 9" means 9am Ontario, not 9am UTC
#    Library: pandas .dt accessor after .tz_convert("America/Toronto")
#
# 2. CARBON INTENSITY LAGS
#    carbon_intensity_lag_24h  — same hour yesterday
#    carbon_intensity_lag_48h  — same hour two days ago
#    → shift() by 24 and 48 on the sorted series; NaN rows at the head must be
#      dropped before training (don't fill with 0 — that's a false signal)
#    Research: pandas DataFrame.shift()
#
# 3. FUEL MIX FRACTIONS (pct_* columns)
#    pct_wind, pct_gas, pct_hydro, pct_nuclear, pct_solar, pct_biofuel
#    → divide each fuel's MW by total_mw for that hour
#    → these columns already exist in carbon_intensity.parquet (MW values)
#    → strong signal for near horizons; model will weight them automatically
#
# 4. WEATHER — Open-Meteo free API (no key needed)
#    Features: temperature_2m (°C), windspeed_10m (m/s)
#    → fetch hourly historical for Ontario (Toronto lat/lon: 43.7, -79.4)
#    → align on interval_start_utc with a merge; timezone-aware join
#    Library: openmeteo-requests (already in pyproject.toml)
#    Research: https://open-meteo.com/en/docs/historical-weather-api
#              look at "hourly" variables; response is JSON → pandas is easy
#    Note: for real-time/forecast features, switch to the Forecast API endpoint
#          but the Historical API is what you need for the training matrix
#
# 5. IESO DEMAND FORECAST (optional but valuable)
#    → gridstatus may expose this; check gridstatus.IESO() methods
#    → if not available via gridstatus, parse the Adequacy Report XML directly
#      (reports-public.ieso.ca/public/Adequacy2/)
#    → align on interval_start_utc same as weather
#    Research: look at what gridstatus.IESO() exposes (dir() or docs),
#              fall back to raw XML if needed
#
# ─── GAP HANDLING ─────────────────────────────────────────────────────────────
#
# ingest.py warns on gaps ≤3h and defers fill to here.
# forward-fill (ffill) those small gaps BEFORE computing lags and fractions.
# Use df.resample("1h").asfreq() to surface implicit gaps, then ffill(limit=3).
# Log how many rows were filled so it's auditable.
#
# ─── OUTPUT SCHEMA ────────────────────────────────────────────────────────────
#
# Target column:  carbon_intensity  (gCO2/kWh)
# Feature columns (everything else):
#   hour_of_day, day_of_week, month, is_weekend,
#   carbon_intensity_lag_24h, carbon_intensity_lag_48h,
#   pct_wind, pct_gas, pct_hydro, pct_nuclear, pct_solar, pct_biofuel,
#   temperature_2m, windspeed_10m,
#   [demand_forecast_mw — if IESO data is available]
#
# ─── FUNCTION SIGNATURES TO IMPLEMENT ────────────────────────────────────────
#
# load_carbon_series(path) -> pd.DataFrame
#   reads parquet, sets interval_start_utc as index, resamples to fill gaps
#
# add_calendar_features(df) -> pd.DataFrame
#   adds hour_of_day, day_of_week, month, is_weekend
#
# add_lag_features(df) -> pd.DataFrame
#   adds carbon_intensity_lag_24h and _lag_48h; does NOT drop NaN rows here
#
# add_fuel_fractions(df) -> pd.DataFrame
#   adds pct_* columns from raw MW columns
#
# fetch_weather(start, end, lat=43.7, lon=-79.4) -> pd.DataFrame
#   calls Open-Meteo historical API, returns hourly temp + windspeed UTC-indexed
#
# build_feature_matrix(carbon_path, drop_na=True) -> pd.DataFrame
#   orchestrates all of the above; drop_na=True drops rows where lags are NaN
#   this is the single entry point model.py will call


# load_carbon_series(path) -> pd.DataFrame
#   reads parquet, sets interval_start_utc as index, resamples to fill gaps
def load_carbon_series(path) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("interval_start_utc")
    df = df.resample("1h").asfreq()
    gap_rows = int(df.isna().any(axis=1).sum())
    df = df.ffill(limit=3)
    if gap_rows:
        logging.info("forward-filled %d gap row(s) ≤3h", gap_rows)
    return df

# add_calendar_features(df) -> pd.DataFrame
#   adds hour_of_day, day_of_week, month, is_weekend
def add_calendar_features(df) -> pd.DataFrame:
    local = df.index.tz_convert("America/Toronto")
    df = df.copy()
    df["hour_of_day"] = local.hour
    df["day_of_week"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    return df

# add_lag_features(df) -> pd.DataFrame
#   adds carbon_intensity_lag_24h and _lag_48h; does NOT drop NaN rows here
def add_lag_features(df) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()

    df["carbon_intensity_lag_24h"] = df["carbon_intensity"].shift(24)
    df["carbon_intensity_lag_48h"] = df["carbon_intensity"].shift(48)

    return df

# add_fuel_fractions(df) -> pd.DataFrame
#   adds pct_* columns from raw MW columns
def add_fuel_fractions(df) -> pd.DataFrame:
    # pct_wind, pct_gas, pct_hydro, pct_nuclear, pct_solar, pct_biofuel
    
    df = df.copy()
    df = df.sort_index()

    df["total"] = df[["wind", "gas", "hydro", "nuclear", "solar", "biofuel", "other"]].sum(axis=1) # only temp

    df["pct_wind"] = df["wind"] / df["total"]
    df["pct_gas"] = df["gas"] / df["total"]
    df["pct_hydro"] = df["hydro"] / df["total"]
    df["pct_nuclear"] = df["nuclear"] / df["total"]
    df["pct_solar"] = df["solar"] / df["total"]
    df["pct_biofuel"] = df["biofuel"] / df["total"]

    df = df.drop(columns=["total"])

    return df

# fetch_weather(start, end, lat=43.7, lon=-79.4) -> pd.DataFrame
#   calls Open-Meteo historical API, returns hourly temp + windspeed UTC-indexed
def fetch_weather(start, end, lat=43.7, lon=-79.4) -> pd.DataFrame:
    openmeteo = openmeteo_requests.Client()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m"],
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }

    response = openmeteo.weather_api(url, params=params)[0]
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

# fetch_ontario_demand(start, end) -> pd.DataFrame
#   fetches IESO historical Ontario demand (MW) via gridstatus, UTC-indexed.
#   used as demand_mw feature at training time; at inference time substitute
#   the official IESO demand forecast (get_resource_adequacy_report).
def fetch_ontario_demand(start: str, end: str) -> pd.DataFrame:
    import gridstatus as gs
    iso = gs.IESO()

    chunks = []
    current = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    while current < end_dt:
        next_cut = min(pd.Timestamp(f"{current.year + 1}-01-01"), end_dt)
        try:
            chunk = iso.get_load_zonal_hourly(
                date=current.strftime("%Y-%m-%d"),
                end=next_cut.strftime("%Y-%m-%d"),
            )
            chunk = (
                chunk[["Interval Start", "Ontario Demand"]]
                .rename(columns={"Interval Start": "ts", "Ontario Demand": "demand_mw"})
            )
            chunk["ts"] = chunk["ts"].dt.tz_convert("UTC")
            chunks.append(chunk.set_index("ts"))
        except Exception as exc:
            logging.warning("demand fetch failed for %s: %s", current.date(), exc)
        current = next_cut

    if not chunks:
        logging.warning("no demand data fetched — demand_mw feature will be absent")
        return pd.DataFrame(columns=["demand_mw"])

    return pd.concat(chunks).resample("1h").mean()


# build_feature_matrix(carbon_path, drop_na=True) -> pd.DataFrame
#   orchestrates all of the above; drop_na=True drops rows where lags are NaN
#   this is the single entry point model.py will call
def build_feature_matrix(carbon_path, drop_na=True) -> pd.DataFrame:
    df = load_carbon_series(carbon_path)

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_fuel_fractions(df)

    start = df.index.min().date().isoformat()
    end = df.index.max().date().isoformat()

    weather_df = fetch_weather(start, end)
    df = df.join(weather_df)

    # Wind speed lead features — observed future values as oracle proxy at training
    # time; at inference time replace with Open-Meteo forecast API values.
    for lead in [6, 12, 24]:
        df[f"wind_speed_lead_{lead}h"] = df["wind_speed_10m"].shift(-lead)

    demand_df = fetch_ontario_demand(start, end)
    if not demand_df.empty:
        df = df.join(demand_df)

    if drop_na:
        lag_cols = ["carbon_intensity_lag_24h", "carbon_intensity_lag_48h"]
        df = df.dropna(subset=lag_cols)

    return df