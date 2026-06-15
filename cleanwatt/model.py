"""
Train 24 direct-step LightGBM models (one per forecast horizon) and save artifacts.

Run directly:
  uv run python -m cleanwatt.model

Outputs:
  models/model_h{1..24}.pkl   — trained LightGBM regressors
  models/eval_metrics.json    — backtest MAE + skill per horizon
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

from cleanwatt.features import build_feature_matrix

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "carbon_intensity_lag_24h", "carbon_intensity_lag_48h",
    "pct_wind", "pct_gas", "pct_hydro", "pct_nuclear", "pct_solar", "pct_biofuel",
    "temperature_2m", "wind_speed_10m",
    "wind_speed_lead_6h", "wind_speed_lead_12h", "wind_speed_lead_24h",
    "demand_mw",
]
TARGET = "carbon_intensity"

HORIZONS = list(range(1, 25))

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)


def _lgb() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**LGB_PARAMS)


def backtest(df: pd.DataFrame) -> dict[int, dict]:
    """
    Expanding-window walk-forward backtest across all 24 horizons.
    Folds: monthly cutoffs 2023-07 → 2025-11 (~29 folds).
    Returns per-horizon results and writes models/eval_metrics.json.
    """
    cutoffs = pd.date_range("2023-07-01", "2025-11-01", freq="MS", tz="UTC")
    results: dict[int, dict] = {}

    for h in HORIZONS:
        targets = df[TARGET].shift(-h)
        baseline = df[TARGET].shift(24 - h)

        actuals, preds, bases = [], [], []

        for cutoff in cutoffs:
            test_end = cutoff + pd.DateOffset(months=1)

            train_mask = (df.index < cutoff) & targets.notna()
            test_mask = (
                (df.index >= cutoff) & (df.index < test_end)
                & targets.notna() & baseline.notna()
            )

            if train_mask.sum() < 200 or test_mask.sum() == 0:
                continue

            model = _lgb()
            model.fit(df.loc[train_mask, FEATURE_COLS], targets[train_mask])
            fold_preds = model.predict(df.loc[test_mask, FEATURE_COLS])

            actuals.extend(targets[test_mask].tolist())
            preds.extend(fold_preds.tolist())
            bases.extend(baseline[test_mask].tolist())

        mae_m = mean_absolute_error(actuals, preds)
        mae_b = mean_absolute_error(actuals, bases)
        skill = 1 - mae_m / mae_b
        results[h] = {"mae_model": round(mae_m, 4), "mae_baseline": round(mae_b, 4), "skill": round(skill, 4)}
        logger.info("h=%2d  MAE=%.2f  baseline=%.2f  skill=%+.3f", h, mae_m, mae_b, skill)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved backtest metrics → %s", metrics_path)

    return results


def train_and_save(df: pd.DataFrame) -> None:
    """
    Train 24 models on the full df and save to models/model_h{n}.pkl.
    Call this after backtest() has already validated the approach.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        targets = df[TARGET].shift(-h)
        mask = targets.notna()

        model = _lgb()
        model.fit(df.loc[mask, FEATURE_COLS], targets[mask])

        path = MODELS_DIR / f"model_h{h}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("h=%2d  trained on %d rows → %s", h, mask.sum(), path)


def run(carbon_path: str = "data/processed/carbon_intensity.parquet") -> None:
    """Train final models on full history and save artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache = Path("data/processed/feature_matrix.parquet")
    if cache.exists():
        logger.info("Loading feature matrix from cache...")
        df = pd.read_parquet(cache)
    else:
        logger.info("Building feature matrix (this takes ~2 min)...")
        df = build_feature_matrix(carbon_path, drop_na=True)
        df.to_parquet(cache)

    logger.info("Feature matrix: %d rows, %d cols", *df.shape)
    train_and_save(df)
    logger.info("Done. Models saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    run()
