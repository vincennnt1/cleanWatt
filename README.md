# CleanWatt

> **"Is now a good time to run it?"**

CleanWatt forecasts Ontario's grid carbon intensity 24 hours ahead and tells you the cleanest windows to run heavy appliances — dryer, dishwasher, EV charging, and more.

Ontario's residential electricity price is mostly regulated Time-of-Use (a fixed schedule, no model needed). Carbon intensity is different: it genuinely swings as the grid shifts between near-zero-carbon nuclear/hydro/wind and gas peakers. That variation is a real forecasting problem.

---

## How it works

1. **Ingest** — pulls hourly fuel mix from IESO's public XML reports via [`gridstatus`](https://github.com/kmax12/gridstatus)
2. **Carbon series** — converts each hour's fuel mix to gCO₂/kWh using standard emission factors (gas ≈ 490, nuclear/hydro/wind ≈ 0)
3. **Forecast** — 24 independent LightGBM models (one per horizon, t+1 through t+24) trained on history back to 2023
4. **Advise** — picks the lowest-carbon contiguous window for your appliance, shows green/yellow/red now-indicator
5. **Dashboard** — Streamlit app with live IESO + Open-Meteo data, refreshed every 5 minutes

---

## Features

- 24h carbon intensity forecast with per-hour confidence
- Appliance window cards: Dryer (1h), Dishwasher (1.5h), EV overnight (6h), custom
- Green / yellow / red badge relative to today's forecast range
- Model Accuracy tab: rolling MAE, skill score vs. "same hour yesterday" baseline
- Fully live: pulls current fuel mix from IESO and weather forecasts from Open-Meteo at runtime, no API key required

---

## Forecast accuracy

Walk-forward backtest, ~29 monthly folds (2023-07 → 2026-06), expanding window. Baseline = "same hour yesterday."

| Horizon | MAE (gCO₂/kWh) | Skill vs. baseline |
|---------|----------------|--------------------|
| h=1     | 4.98           | +0.771             |
| h=6     | 13.45          | +0.382             |
| h=12    | 16.74          | +0.231             |
| h=18    | 17.95          | +0.176             |
| h=24    | 17.77          | +0.185             |

**Mean skill across all 24 horizons: +0.301** (30% lower error than the naive baseline). Baseline MAE ≈ 21.76 gCO₂/kWh flat.

Notable: skill doesn't collapse monotonically — dips around h=15–16 then recovers to ~+0.19 at h=22–23 because `lag_24h` (current intensity) and a 24h wind speed lead give the model a strong signal at that pocket.

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data ingestion | `gridstatus`, `pandas`, `pyarrow` |
| Weather | `openmeteo-requests` (free, no key) |
| Modeling | `lightgbm`, `scikit-learn` |
| Dashboard | `streamlit`, `altair` |
| Storage | Parquet |
| Package manager | `uv` |

---

## Project layout

```
cleanwatt/
  ingest.py     # IESO fuel mix → data/processed/
  carbon.py     # fuel mix → gCO₂/kWh
  features.py   # feature matrix builder
  model.py      # train 24 LightGBM models, walk-forward backtest
  advisor.py    # window selection, green/yellow/red (UI-agnostic)
  app.py        # Streamlit dashboard
data/
  processed/    # carbon_intensity.parquet — canonical hourly series
  forecast_log.parquet  # predicted vs actual, appended daily
models/         # 24 .pkl artifacts + eval_metrics.json
tests/
notebooks/      # EDA and backtest analysis only
```

---

## Setup

Requires Python ≥ 3.13 and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd watt
uv sync
```

### Run the dashboard

```bash
uv run streamlit run cleanwatt/app.py
```

### Retrain models

```bash
uv run python -m cleanwatt.ingest      # refresh historical data
uv run python -m cleanwatt.model       # retrain all 24 models
```

### Tests

```bash
uv run pytest
```

## Live Demo

Try the live dashboard → [cleanwatt.streamlit.app](https://cleanwatt-ik2xgubtzhbwj3qvqwpv9f.streamlit.app/)

## Caveats

- **Emission factors are a choice.** Values follow the Electricity Maps methodology; documented in `carbon.py`.
- **Imports/exports excluded.** Ontario trades with neighbours; cross-border power carries its own carbon profile not captured here.
- **Average vs. marginal carbon.** This uses average grid mix, not the marginal generator.
