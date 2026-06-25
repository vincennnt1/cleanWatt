"""
api/index.py — FastAPI wrapper around cleanwatt.advisor.

Endpoints:
  GET /health              — liveness check
  GET /forecast            — full 24-hour forecast + appliance windows
  GET /forecast/window     — cheapest window for a custom appliance duration
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from cleanwatt.advisor import (
    APPLIANCE_PRESETS,
    best_window,
    fetch_live_features,
    forecast_24h,
    classify_intensity,
    load_models,
)

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # seconds — matches Streamlit's 5-min cache

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["models"] = load_models()
    _state["cache"] = {"data": None, "ts": 0.0}
    logger.info("Models loaded.")
    yield
    _state.clear()


app = FastAPI(title="CleanWatt API", version="1.0.0", lifespan=lifespan)


# ── Response models ───────────────────────────────────────────────────────────

class Window(BaseModel):
    start_h: int
    end_h: int
    avg_intensity: float


class ForecastResponse(BaseModel):
    current_intensity: float
    status: str
    predictions: list[float]
    windows: dict[str, Window]
    now_utc: str
    now_ontario: str
    fuel_timestamp: str


class WindowResponse(BaseModel):
    hours: float
    window: Window


# ── Caching helper ────────────────────────────────────────────────────────────

def _get_cached_forecast() -> dict:
    cache = _state["cache"]
    if cache["data"] is not None and (time.monotonic() - cache["ts"]) < _CACHE_TTL:
        return cache["data"]

    feature_row, meta = fetch_live_features()
    predictions = forecast_24h(_state["models"], feature_row)
    current = meta["current_intensity"]

    data = {
        "current_intensity": current,
        "status": classify_intensity(current, predictions),
        "predictions": predictions,
        "windows": {
            p["name"]: best_window(predictions, p["hours"]) for p in APPLIANCE_PRESETS
        },
        "now_utc": meta["now_utc"].isoformat(),
        "now_ontario": meta["now_ontario"].isoformat(),
        "fuel_timestamp": str(meta["fuel_timestamp"]),
    }
    cache["data"] = data
    cache["ts"] = time.monotonic()
    return data


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/forecast", response_model=ForecastResponse)
async def forecast():
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, _get_cached_forecast
        )
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ForecastResponse(**data)


@app.get("/forecast/window", response_model=WindowResponse)
async def forecast_window(
    hours: float = Query(default=1.0, gt=0, le=24, description="Appliance duration in hours"),
):
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, _get_cached_forecast
        )
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    w = best_window(data["predictions"], hours)
    return WindowResponse(hours=hours, window=Window(**w))
