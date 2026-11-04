"""Feature engineering for Champion genomes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c, h, l = x.close, x.high, x.low
    x["ema_fast"] = _ema(c, 12)
    x["ema_mid"] = _ema(c, 26)
    x["ema_slow"] = _ema(c, 50)
    x["ema_long"] = _ema(c, 200)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    x["atr_pct"] = x.atr / c
    x["vol_sma"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume / x.vol_sma.replace(0, np.nan)
    if "taker_buy_quote" in x and "quote_volume" in x:
        x["taker_buy_ratio"] = x.taker_buy_quote / x.quote_volume.replace(0, np.nan)
    else:
        x["taker_buy_ratio"] = np.nan
    x["bb_mid"] = c.rolling(20).mean()
    x["bb_std"] = c.rolling(20).std(ddof=0)
    x["bb_z"] = (c - x.bb_mid) / x.bb_std.replace(0, np.nan)
    x["don_high"] = h.shift(1).rolling(20).max()
    x["don_low"] = l.shift(1).rolling(20).min()
    x["ret_5"] = c.pct_change(5)
    x["ret_20"] = c.pct_change(20)
    x["ema_spread"] = (x.ema_fast - x.ema_slow) / c
    x["trend_strength"] = (x.ema_slow - x.ema_long).abs() / c
    typical = (h + l + c) / 3
    x["vwap_rolling"] = (typical * x.volume).rolling(48).sum() / x.volume.rolling(
        48
    ).sum().replace(0, np.nan)
    x["vwap_z"] = (c - x.vwap_rolling) / x.atr.replace(0, np.nan)
    x["atr_pct_rank"] = x.atr_pct.rolling(240, min_periods=60).rank(pct=True)
    return x.replace([np.inf, -np.inf], np.nan)
