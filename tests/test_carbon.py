import pandas as pd
import pytest
from cleanwatt.carbon import intensity_from_mix, add_intensity_column, EMISSION_FACTORS


def test_pure_nuclear_is_near_zero():
    result = intensity_from_mix({"nuclear": 1000.0})
    assert result == EMISSION_FACTORS["nuclear"]


def test_pure_gas_returns_gas_factor():
    result = intensity_from_mix({"gas": 500.0})
    assert result == EMISSION_FACTORS["gas"]


def test_mixed_grid_is_between_extremes():
    mix = {"nuclear": 1000.0, "gas": 1000.0}
    result = intensity_from_mix(mix)
    assert EMISSION_FACTORS["nuclear"] < result < EMISSION_FACTORS["gas"]


def test_mixed_grid_weighted_correctly():
    # 75% nuclear, 25% gas → expect weighted average
    mix = {"nuclear": 750.0, "gas": 250.0}
    expected = (750 * EMISSION_FACTORS["nuclear"] + 250 * EMISSION_FACTORS["gas"]) / 1000
    assert abs(intensity_from_mix(mix) - expected) < 1e-6


def test_zero_generation_returns_zero():
    result = intensity_from_mix({})
    assert result == 0.0


def test_unknown_fuels_ignored():
    # Keys not in EMISSION_FACTORS should not raise
    result = intensity_from_mix({"nuclear": 100.0, "unicorn": 999.0})
    assert result == EMISSION_FACTORS["nuclear"]


def test_add_intensity_column_shape():
    df = pd.DataFrame([
        {"nuclear": 8000.0, "hydro": 4000.0, "wind": 1000.0, "solar": 200.0,
         "gas": 600.0, "biofuel": 100.0, "other": 0.0},
        {"nuclear": 7500.0, "hydro": 4200.0, "wind": 800.0, "solar": 0.0,
         "gas": 1200.0, "biofuel": 80.0, "other": 0.0},
    ])
    out = add_intensity_column(df)
    assert "carbon_intensity" in out.columns
    assert len(out) == 2
    assert out["carbon_intensity"].notna().all()


def test_add_intensity_column_does_not_mutate():
    df = pd.DataFrame([{"nuclear": 1000.0, "gas": 500.0}])
    original_cols = list(df.columns)
    add_intensity_column(df)
    assert list(df.columns) == original_cols


def test_high_gas_mix_higher_than_low_gas_mix():
    low_gas = intensity_from_mix({"nuclear": 9000.0, "gas": 100.0})
    high_gas = intensity_from_mix({"nuclear": 1000.0, "gas": 9000.0})
    assert high_gas > low_gas
