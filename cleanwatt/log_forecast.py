"""
log_forecast.py — Daily forecast log append and drift detection.

Run via cron (GitHub Actions) each morning:
    uv run python -m cleanwatt.log_forecast

Appends today's 24h predictions to data/forecast_log.parquet,
fills in actuals for yesterday's rows, and warns on drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from cleanwatt.advisor import load_models, get_forecast, HORIZONS
from cleanwatt.ingest import build_carbon_series, save_parquet

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
FORECAST_LOG_PATH = DATA_DIR / "forecast_log.parquet"
CARBON_PATH = DATA_DIR / "processed" / "carbon_intensity.parquet"
EVAL_METRICS_PATH = MODELS_DIR / "eval_metrics.json"

LOG_SCHEMA = {
    "forecast_made_at": "datetime64[us, UTC]",
    "target_hour": "datetime64[us, UTC]",
    "predicted_gco2": "float64",
    "actual_gco2": "float64",
    "horizon_hours": "int64",
}

# Rolling window for drift detection (days of complete rows)
_DRIFT_WINDOW_DAYS = 7
# Alert if rolling MAE exceeds training MAE by this multiplier
_DRIFT_THRESHOLD = 1.5


def refresh_carbon_series() -> None:
    """
    Fetch the last 3 days from IESO and merge into carbon_intensity.parquet.
    Preserves full history — only new/updated rows are written.
    """
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).date().isoformat()
    logger.info("Fetching IESO data from %s...", start)
    new_rows = build_carbon_series(start, None)

    if CARBON_PATH.exists():
        existing = pd.read_parquet(CARBON_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["interval_start_utc"], keep="last")
        combined = combined.sort_values("interval_start_utc").reset_index(drop=True)
    else:
        combined = new_rows

    save_parquet(combined, CARBON_PATH)
    logger.info("carbon_intensity.parquet updated — %d total rows", len(combined))


def make_forecast_rows(now_utc: pd.Timestamp, predictions: list[float]) -> pd.DataFrame:
    """Build 24 log rows from a fresh get_forecast() result."""
    rows = []
    for h, pred in zip(HORIZONS, predictions):
        rows.append({
            "forecast_made_at": now_utc,
            "target_hour": now_utc + pd.Timedelta(hours=h),
            "predicted_gco2": pred,
            "actual_gco2": float("nan"),
            "horizon_hours": h,
        })
    return pd.DataFrame(rows).astype(LOG_SCHEMA)


def fill_actuals(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match target_hours against carbon_intensity.parquet and fill actual_gco2
    for any rows where the target hour is in the past and actual is currently NaN.
    """
    if not CARBON_PATH.exists():
        logger.warning("carbon_intensity.parquet not found — skipping actuals fill")
        return log_df

    actuals = pd.read_parquet(CARBON_PATH)[["interval_start_utc", "carbon_intensity"]]
    actuals["interval_start_utc"] = pd.to_datetime(
        actuals["interval_start_utc"], utc=True
    ).dt.floor("h")
    actuals = actuals.set_index("interval_start_utc")["carbon_intensity"]

    now_utc = pd.Timestamp.now(tz="UTC")
    needs_fill = log_df["actual_gco2"].isna() & (log_df["target_hour"] < now_utc)

    def _lookup(ts: pd.Timestamp) -> float:
        ts = ts.floor("h")
        if ts in actuals.index:
            return float(actuals[ts])
        return float("nan")

    log_df = log_df.copy()
    log_df.loc[needs_fill, "actual_gco2"] = log_df.loc[needs_fill, "target_hour"].map(_lookup)

    filled = needs_fill.sum() - log_df.loc[needs_fill, "actual_gco2"].isna().sum()
    logger.info("Filled %d actual values", filled)
    return log_df


def detect_drift(log_df: pd.DataFrame) -> str | None:
    """
    Compare rolling 7-day MAE per horizon against training MAE.
    Returns a warning string if any horizon exceeds 1.5× training MAE, else None.
    """
    if not EVAL_METRICS_PATH.exists():
        logger.info("eval_metrics.json not found — skipping drift check")
        return None

    with open(EVAL_METRICS_PATH) as f:
        eval_metrics: dict = json.load(f)

    complete = log_df.dropna(subset=["actual_gco2"]).copy()
    if complete.empty:
        return None

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_DRIFT_WINDOW_DAYS)
    recent = complete[complete["target_hour"] >= cutoff]
    if recent.empty:
        return None

    warnings = []
    for h in HORIZONS:
        h_rows = recent[recent["horizon_hours"] == h]
        if len(h_rows) < 3:
            continue
        rolling_mae = (h_rows["predicted_gco2"] - h_rows["actual_gco2"]).abs().mean()
        train_mae = eval_metrics.get(str(h), {}).get("mae_model")
        if train_mae is None or train_mae <= 0:
            continue
        if rolling_mae > _DRIFT_THRESHOLD * train_mae:
            warnings.append(
                f"h={h}: rolling MAE {rolling_mae:.1f} > {_DRIFT_THRESHOLD}× "
                f"training MAE {train_mae:.1f}"
            )

    if warnings:
        return "Model drift detected:\n" + "\n".join(f"  {w}" for w in warnings)
    return None


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    refresh_carbon_series()

    logger.info("Loading models...")
    models = load_models()

    logger.info("Fetching live forecast...")
    result = get_forecast(models)
    now_utc: pd.Timestamp = result["now_utc"]
    predictions: list[float] = result["predictions"]
    logger.info("Forecast made at %s", now_utc)

    new_rows = make_forecast_rows(now_utc, predictions)

    if FORECAST_LOG_PATH.exists():
        existing = pd.read_parquet(FORECAST_LOG_PATH)
        existing["forecast_made_at"] = pd.to_datetime(existing["forecast_made_at"], utc=True)
        existing["target_hour"] = pd.to_datetime(existing["target_hour"], utc=True)
        existing["actual_gco2"] = pd.to_numeric(existing["actual_gco2"], errors="coerce")
        log_df = pd.concat([existing, new_rows], ignore_index=True)
    else:
        log_df = new_rows

    # Deduplicate — safe to re-run
    log_df = log_df.drop_duplicates(subset=["forecast_made_at", "target_hour"], keep="last")
    log_df = log_df.sort_values(["forecast_made_at", "horizon_hours"]).reset_index(drop=True)

    log_df = fill_actuals(log_df)

    FORECAST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_parquet(FORECAST_LOG_PATH, index=False)
    logger.info("Saved %d rows → %s", len(log_df), FORECAST_LOG_PATH)

    drift_warning = detect_drift(log_df)
    if drift_warning:
        print(f"\n⚠ DRIFT WARNING ⚠\n{drift_warning}\n")
    else:
        logger.info("No drift detected.")


if __name__ == "__main__":
    run()
