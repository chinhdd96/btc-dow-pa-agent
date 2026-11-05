"""Closed catalog of tradeable indicators + allowed params (Strategy Spec phase-1)."""

from __future__ import annotations

from typing import Any

# id -> allowed param values (lists). Empty dict = no free params (uses market cols).
INDICATOR_CATALOG: dict[str, dict[str, list[Any]]] = {
    # Trend
    "sma": {"period": [20, 50, 100, 200]},
    "ema": {"period": [8, 12, 21, 50, 100, 200]},
    "ema_cross": {"fast": [8, 12, 20], "slow": [21, 50, 100]},
    "macd": {"fast": [8, 12], "slow": [17, 26], "signal": [9]},
    "adx": {"period": [14]},
    # Momentum
    "rsi": {"period": [6, 7, 14, 21]},
    "stoch_k": {"period": [14]},
    "cci": {"period": [20]},
    "ret": {"period": [5, 10, 20]},
    "williams_r": {"period": [14]},
    # Volatility
    "atr": {"period": [14]},
    "atr_pct": {"period": [14]},
    "bb_z": {"period": [20], "std": [2.0]},
    "donchian_high": {"period": [20, 55]},
    "donchian_low": {"period": [20, 55]},
    # Volume / flow
    "vol_ratio": {"period": [20]},
    "taker_buy_ratio": {},
    "obv_slope": {"period": [10, 20]},
    # Level
    "vwap_z": {"window": [24, 48, 96]},
    "dist_to_ema": {"period": [50, 200]},
    # Derivatives / HTF (from enriched frame)
    "funding_rate": {},
    "oi_change": {},
    "htf_1h_bull": {},
    "htf_4h_bull": {},
    # Price helpers
    "close": {},
}

OPS = ("lt", "gt", "lte", "gte", "eq", "between", "cross_up", "cross_down", "is_true", "is_false")


def catalog_for_prompt() -> dict[str, Any]:
    return {
        "indicators": {
            k: {"params": v} for k, v in INDICATOR_CATALOG.items()
        },
        "ops": list(OPS),
    }


def validate_indicator(ind: str, params: dict[str, Any] | None) -> tuple[bool, str]:
    if ind not in INDICATOR_CATALOG:
        return False, f"unknown indicator: {ind}"
    allowed = INDICATOR_CATALOG[ind]
    params = params or {}
    for key, choices in allowed.items():
        if key not in params:
            return False, f"{ind}: missing param {key}"
        val = params[key]
        # allow int/float match against catalog numbers
        ok = False
        for c in choices:
            if isinstance(c, float) or isinstance(val, float):
                if abs(float(val) - float(c)) < 1e-9:
                    ok = True
                    break
            elif val == c or int(val) == int(c):
                ok = True
                break
        if not ok:
            return False, f"{ind}.{key}={val} not in {choices}"
    for key in params:
        if key not in allowed and allowed:
            return False, f"{ind}: unexpected param {key}"
    return True, ""


def col_key(ind: str, params: dict[str, Any] | None = None) -> str:
    """Stable dataframe column name for an indicator instance."""
    params = params or {}
    if not params:
        return ind
    parts = [ind] + [f"{k}{params[k]}" for k in sorted(params)]
    return "_".join(str(p).replace(".", "p") for p in parts)
