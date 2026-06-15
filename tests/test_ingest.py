import pandas as pd
import pytest
from cleanwatt.ingest import validate_output, REQUIRED_OUTPUT_COLUMNS, FUEL_COLUMNS


def _make_valid_df(n: int = 24) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = [start + pd.Timedelta(hours=i) for i in range(n)]
    rows = []
    for ts in timestamps:
        row = {"interval_start_utc": ts, "carbon_intensity": 50.0}
        for fuel in FUEL_COLUMNS:
            row[fuel] = 100.0
        rows.append(row)
    return pd.DataFrame(rows)[REQUIRED_OUTPUT_COLUMNS]


def test_validate_passes_on_valid_df():
    validate_output(_make_valid_df())


def test_validate_raises_on_missing_column():
    df = _make_valid_df().drop(columns=["nuclear"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_output(df)


def test_validate_raises_on_null_timestamp():
    df = _make_valid_df()
    df.loc[0, "interval_start_utc"] = pd.NaT
    with pytest.raises(ValueError, match="Null timestamps"):
        validate_output(df)


def test_validate_raises_on_null_fuel():
    df = _make_valid_df()
    df.loc[0, "gas"] = None
    with pytest.raises(ValueError, match="Null values in fuel"):
        validate_output(df)


def test_validate_raises_on_null_intensity():
    df = _make_valid_df()
    df.loc[5, "carbon_intensity"] = None
    with pytest.raises(ValueError, match="Null carbon_intensity"):
        validate_output(df)


def test_validate_warns_on_small_gap(caplog):
    import logging
    df = _make_valid_df(n=48)
    # Remove row 12 to introduce a 2h gap (≤3h → warn, not raise)
    df = df.drop(index=12).reset_index(drop=True)
    with caplog.at_level(logging.WARNING):
        validate_output(df)
    assert "gap" in caplog.text.lower()


def test_validate_raises_on_large_gap():
    df = _make_valid_df(n=48)
    # Remove rows 12-15 to introduce a 5h gap (>3h → raise)
    df = df.drop(index=[12, 13, 14, 15]).reset_index(drop=True)
    with pytest.raises(ValueError, match="larger than 3 hours"):
        validate_output(df)
