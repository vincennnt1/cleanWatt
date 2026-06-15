"""
Pulls IESO hourly generation-by-fuel data via gridstatus and writes:
  - data/processed/carbon_intensity.parquet  (canonical hourly series)

Run directly:
  uv run python -m cleanwatt.ingest
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import gridstatus

from cleanwatt.carbon import add_intensity_column, FUEL_COLUMNS

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Mapping from gridstatus column names → our canonical fuel names
# Adjust if gridstatus changes its schema.
GRIDSTATUS_FUEL_MAP: dict[str, str] = {
    "Nuclear": "nuclear",
    "Hydro": "hydro",
    "Wind": "wind",
    "Solar": "solar",
    "Gas": "gas",
    "Biofuel": "biofuel",
    "Other": "other",
}

REQUIRED_OUTPUT_COLUMNS = ["interval_start_utc"] + FUEL_COLUMNS + ["carbon_intensity"]


def fetch_fuel_mix(start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly generation-by-fuel from IESO via gridstatus.

    Args:
        start: ISO date string, e.g. "2024-01-01"
        end:   ISO date string, exclusive, e.g. "2024-02-01"

    Returns:
        DataFrame with columns: interval_start_utc, nuclear, hydro, wind,
        solar, gas, biofuel, other — all in MW.
    """
    ieso = gridstatus.IESO()
    logger.info("Fetching IESO fuel mix %s → %s", start, end)
    raw = ieso.get_fuel_mix(date=start, end=end)

    # gridstatus returns wide format with "Interval Start" / "Interval End";
    # older versions used "Fuel"/"MW" long format — pivot if needed.
    if "Fuel" in raw.columns and "MW" in raw.columns:
        raw = raw.pivot_table(
            index="Interval Start", columns="Fuel", values="MW", aggfunc="first"
        ).reset_index()
        raw.columns.name = None

    # Normalise timestamp column name (gridstatus ≥0.24 uses "Interval Start")
    raw = raw.rename(columns={"Interval Start": "interval_start_utc", "Time": "interval_start_utc"})
    raw = raw.rename(columns={k: v for k, v in GRIDSTATUS_FUEL_MAP.items() if k in raw.columns})

    # Ensure UTC-aware timestamps
    if not pd.api.types.is_datetime64_any_dtype(raw["interval_start_utc"]):
        raw["interval_start_utc"] = pd.to_datetime(raw["interval_start_utc"], utc=True)
    elif raw["interval_start_utc"].dt.tz is None:
        raw["interval_start_utc"] = raw["interval_start_utc"].dt.tz_localize("America/Toronto").dt.tz_convert("UTC")
    else:
        raw["interval_start_utc"] = raw["interval_start_utc"].dt.tz_convert("UTC")

    # Fill any missing fuel columns with 0; cast to float (some cols arrive as object)
    for col in FUEL_COLUMNS:
        if col not in raw.columns:
            raw[col] = 0.0
        else:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    raw = raw[["interval_start_utc"] + FUEL_COLUMNS].sort_values("interval_start_utc").reset_index(drop=True)
    return raw


def build_carbon_series(start: str, end: str) -> pd.DataFrame:
    """
    Fetch fuel mix and compute carbon intensity.

    Returns DataFrame with REQUIRED_OUTPUT_COLUMNS.
    """
    df = fetch_fuel_mix(start, end)
    df = add_intensity_column(df)
    return df[REQUIRED_OUTPUT_COLUMNS]


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)


def validate_output(df: pd.DataFrame) -> None:
    """Raise ValueError if the output DataFrame fails basic schema checks."""
    missing = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["interval_start_utc"].isnull().any():
        raise ValueError("Null timestamps found in output")

    if df[FUEL_COLUMNS].isnull().any().any():
        raise ValueError("Null values in fuel mix columns")

    if df["carbon_intensity"].isnull().any():
        raise ValueError("Null carbon_intensity values")

    # Check contiguous hourly series — raise on large gaps, warn on small ones
    diffs = df["interval_start_utc"].diff().dropna()
    non_hourly = diffs[diffs != pd.Timedelta("1h")]
    if not non_hourly.empty:
        large_gaps = non_hourly[non_hourly > pd.Timedelta("3h")]
        if not large_gaps.empty:
            gap_times = df.loc[large_gaps.index, "interval_start_utc"].tolist()
            raise ValueError(f"Gap(s) larger than 3 hours in fetched data: {gap_times}")
        gap_times = df.loc[non_hourly.index, "interval_start_utc"].tolist()
        logger.warning("%d small gap(s) ≤3h detected — will forward-fill in features.py: %s", len(non_hourly), gap_times)


def run(start: str = "2023-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Full ingestion pipeline. Writes carbon_intensity.parquet and returns the df.

    Args:
        start: start date (inclusive)
        end:   end date (exclusive). Defaults to today.
    """
    if end is None:
        end = pd.Timestamp.now(tz="UTC").date().isoformat()

    df = build_carbon_series(start, end)
    validate_output(df)
    save_parquet(df, PROCESSED_DIR / "carbon_intensity.parquet")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
