"""Market regime tag for a closed bar (trend/range × volatility × direction)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modules.features_catalog import ensure_indicator
from modules.indicator_catalog import col_key

ADX_KEY = col_key("adx", {"period": 14})


def ensure_regime_columns(x: pd.DataFrame) -> pd.DataFrame:
    if ADX_KEY not in x.columns:
        x = ensure_indicator(x, "adx", {"period": 14})
    return x


def _num(row: pd.Series, key: str) -> float:
    try:
        v = float(row.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return v


def regime_of(row: pd.Series) -> dict[str, Any]:
    adx = _num(row, ADX_KEY)
    rank = _num(row, "atr_pct_rank")
    ema_slow, ema_long = _num(row, "ema_slow"), _num(row, "ema_long")
    trend = "mixed"
    if np.isfinite(adx):
        trend = "trend" if adx >= 25 else ("range" if adx < 20 else "mixed")
    vol = "mid_vol"
    if np.isfinite(rank):
        vol = "high_vol" if rank >= 0.7 else ("low_vol" if rank <= 0.3 else "mid_vol")
    direction = "flat"
    if np.isfinite(ema_slow) and np.isfinite(ema_long):
        direction = "up" if ema_slow > ema_long else "down"
    return {
        "trend": trend,
        "vol": vol,
        "dir": direction,
        "adx": round(adx, 1) if np.isfinite(adx) else None,
        "tag": f"{trend}_{direction}|{vol}",
    }
