"""
Converts IESO hourly fuel mix (MW per fuel type) to carbon intensity (gCO2/kWh).

Emission factors from Electricity Maps / Environment Canada:
  - Gas: combined-cycle average for Ontario (~490 gCO2/kWh)
  - Nuclear, hydro, wind, solar: lifecycle near-zero
  - Biofuel: combustion emissions (~230 gCO2/kWh, lifecycle estimate)

These are average-intensity factors, not marginal. See CLAUDE.md for the caveat.
"""

import pandas as pd

# gCO2 per kWh of generation (lifecycle average, rounded to documented values)
EMISSION_FACTORS: dict[str, float] = {
    "nuclear": 12.0,
    "hydro": 24.0,
    "wind": 11.0,
    "solar": 41.0,
    "gas": 490.0,
    "biofuel": 230.0,
    "other": 490.0,  # conservative: treat unknowns as gas
}

# Canonical column names as returned by gridstatus IESO fuel mix
FUEL_COLUMNS = list(EMISSION_FACTORS.keys())


def intensity_from_mix(mix: dict[str, float]) -> float:
    """
    Compute gCO2/kWh from a single hour's fuel mix.

    Args:
        mix: dict mapping fuel name → generation in MW (or any proportional unit)

    Returns:
        Carbon intensity in gCO2/kWh. Returns 0.0 if total generation is zero.
    """
    total = sum(mix.get(f, 0.0) for f in EMISSION_FACTORS)
    if total == 0.0:
        return 0.0
    weighted = sum(mix.get(f, 0.0) * factor for f, factor in EMISSION_FACTORS.items())
    return weighted / total


def add_intensity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `carbon_intensity` column (gCO2/kWh) to a fuel-mix DataFrame.

    Expects one column per fuel type (MW). Missing fuel columns are treated as 0.
    Returns a copy with the new column appended.
    """
    df = df.copy()
    df["carbon_intensity"] = df.apply(
        lambda row: intensity_from_mix({f: row.get(f, 0.0) for f in EMISSION_FACTORS}),
        axis=1,
    )
    return df
