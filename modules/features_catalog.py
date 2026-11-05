"""Compute catalog indicators with explicit params (wires Spec periods into series)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from modules.indicator_catalog import col_key
from modules.features import add_features as add_base_features


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _rsi(c: pd.Series, period: int) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(h, l, c, period)
    plus_di = 100 * pd.Series(plus_dm, index=c.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=c.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ensure_indicator(
    df: pd.DataFrame,
    ind: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add column for one indicator instance if missing. Mutates copy."""
    params = dict(params or {})
    key = col_key(ind, params if params else None)
    if key in df.columns and ind not in {"ema_cross"}:
        return df
    x = df
    c, h, l = x["close"], x["high"], x["low"]
    vol = x["volume"] if "volume" in x else pd.Series(np.nan, index=x.index)

    if ind == "close":
        x[key] = c
    elif ind == "sma":
        x[key] = _sma(c, int(params["period"]))
    elif ind == "ema":
        x[key] = _ema(c, int(params["period"]))
    elif ind == "ema_cross":
        fast = _ema(c, int(params["fast"]))
        slow = _ema(c, int(params["slow"]))
        x[key] = (fast - slow) / c.replace(0, np.nan)
        x[col_key("ema", {"period": int(params["fast"])})] = fast
        x[col_key("ema", {"period": int(params["slow"])})] = slow
    elif ind == "macd":
        fast_p, slow_p, sig_p = int(params["fast"]), int(params["slow"]), int(params["signal"])
        macd_line = _ema(c, fast_p) - _ema(c, slow_p)
        signal = macd_line.ewm(span=sig_p, adjust=False, min_periods=sig_p).mean()
        x[key] = macd_line - signal  # histogram
    elif ind == "adx":
        x[key] = _adx(h, l, c, int(params["period"]))
    elif ind == "rsi":
        x[key] = _rsi(c, int(params["period"]))
    elif ind == "stoch_k":
        n = int(params["period"])
        lowest = l.rolling(n, min_periods=n).min()
        highest = h.rolling(n, min_periods=n).max()
        x[key] = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    elif ind == "cci":
        n = int(params["period"])
        tp = (h + l + c) / 3
        sma = tp.rolling(n, min_periods=n).mean()
        mad = tp.rolling(n, min_periods=n).apply(
            lambda a: np.mean(np.abs(a - np.mean(a))), raw=True
        )
        x[key] = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    elif ind == "ret":
        x[key] = c.pct_change(int(params["period"]))
    elif ind == "williams_r":
        n = int(params["period"])
        highest = h.rolling(n, min_periods=n).max()
        lowest = l.rolling(n, min_periods=n).min()
        x[key] = -100 * (highest - c) / (highest - lowest).replace(0, np.nan)
    elif ind == "atr":
        x[key] = _atr(h, l, c, int(params["period"]))
    elif ind == "atr_pct":
        atr = _atr(h, l, c, int(params["period"]))
        x[key] = atr / c.replace(0, np.nan)
        x["atr"] = atr  # exit sizing convenience
    elif ind == "bb_z":
        n = int(params["period"])
        std_n = float(params.get("std", 2.0))
        mid = c.rolling(n, min_periods=n).mean()
        std = c.rolling(n, min_periods=n).std(ddof=0)
        x[key] = (c - mid) / (std * std_n / 2.0).replace(0, np.nan)  # z vs 1σ; value~±2 at bands
        # simpler: classic z-score
        x[key] = (c - mid) / std.replace(0, np.nan)
    elif ind == "donchian_high":
        n = int(params["period"])
        x[key] = h.shift(1).rolling(n, min_periods=n).max()
    elif ind == "donchian_low":
        n = int(params["period"])
        x[key] = l.shift(1).rolling(n, min_periods=n).min()
    elif ind == "vol_ratio":
        n = int(params["period"])
        sma = vol.rolling(n, min_periods=n).mean()
        x[key] = vol / sma.replace(0, np.nan)
    elif ind == "taker_buy_ratio":
        if "taker_buy_ratio" in x.columns:
            x[key] = x["taker_buy_ratio"]
        elif "taker_buy_quote" in x and "quote_volume" in x:
            x[key] = x["taker_buy_quote"] / x["quote_volume"].replace(0, np.nan)
        else:
            x[key] = np.nan
    elif ind == "obv_slope":
        n = int(params["period"])
        direction = np.sign(c.diff()).fillna(0)
        obv = (direction * vol).cumsum()
        x[key] = obv.diff(n) / n
    elif ind == "vwap_z":
        w = int(params["window"])
        typical = (h + l + c) / 3
        vwap = (typical * vol).rolling(w).sum() / vol.rolling(w).sum().replace(0, np.nan)
        atr = _atr(h, l, c, 14)
        x[key] = (c - vwap) / atr.replace(0, np.nan)
    elif ind == "dist_to_ema":
        ema = _ema(c, int(params["period"]))
        x[key] = (c - ema) / c.replace(0, np.nan)
    elif ind in {"funding_rate", "oi_change", "htf_1h_bull", "htf_4h_bull"}:
        if ind in x.columns:
            x[key] = x[ind]
        else:
            x[key] = np.nan
    else:
        raise ValueError(f"cannot compute indicator: {ind}")
    return x


def collect_requirements(conditions: Iterable[dict[str, Any]]) -> list[tuple[str, dict]]:
    req: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        ind = str(cond.get("ind") or "")
        params = dict(cond.get("params") or {})
        k = col_key(ind, params if params else None)
        if k not in seen:
            seen.add(k)
            req.append((ind, params))
        vs = cond.get("vs")
        if isinstance(vs, dict) and vs.get("ind"):
            vind = str(vs["ind"])
            vparams = dict(vs.get("params") or {})
            vk = col_key(vind, vparams if vparams else None)
            if vk not in seen:
                seen.add(vk)
                req.append((vind, vparams))
    return req


def ensure_features_for_spec(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Base features + all Spec-referenced catalog columns."""
    x = add_base_features(df)
    conds: list[dict] = []
    for key in ("entry_long", "entry_short", "filters"):
        block = spec.get(key) or []
        if isinstance(block, list):
            conds.extend([c for c in block if isinstance(c, dict)])
    # Always ensure atr for exits
    exit_cfg = spec.get("exit") or {}
    atr_period = int(exit_cfg.get("atr_period") or 14)
    conds.append({"ind": "atr", "params": {"period": atr_period}})
    conds.append({"ind": "atr_pct", "params": {"period": atr_period}})
    for ind, params in collect_requirements(conds):
        try:
            x = ensure_indicator(x, ind, params)
        except Exception:
            continue
    # Alias classic columns used by legacy code paths
    if "atr" not in x.columns:
        x = ensure_indicator(x, "atr", {"period": 14})
    atr_key = col_key("atr", {"period": atr_period})
    if atr_key in x.columns:
        x["atr"] = x[atr_key]
    atrp_key = col_key("atr_pct", {"period": atr_period})
    if atrp_key in x.columns:
        x["atr_pct"] = x[atrp_key]
    return x.replace([np.inf, -np.inf], np.nan)
