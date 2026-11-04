#!/usr/bin/env python3
"""BTCUSDT Autonomous Research & Paper Trading Agent v3.0

Research, LLM-assisted strategy discovery, derivatives-aware validation and PAPER trading only.
No private Binance endpoints and no live order execution are implemented.

Python 3.10+. Dependencies: requests, pandas, numpy.

Core design:
- Generates and evolves strategy genomes (not restricted to a fixed strategy list).
- Evaluates candidates with chronological train/validation/test windows.
- Uses conservative OHLC stop/target handling, fees and slippage.
- Selects a Champion only after independent validation.
- Paper-trades closed candles and persists state/trades in SQLite.
- Maximum initial margin allocation per position: 10% of current equity.
- Leverage is selected by the agent from a configurable candidate set.
- Live trading is intentionally absent; paper_manual means the user must
  separately approve any future live-trading implementation.

Important: profitable results are not guaranteed. This is an experimental
research framework, not investment advice or a production execution system.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
DB_DEFAULT = "btc_agent_v3.sqlite3"
REPORT_DEFAULT = "btc_agent_v3_report.json"


@dataclass
class Config:
    symbol: str = SYMBOL
    interval: str = "1h"
    history_days: int = 730
    initial_capital: float = 1000.0
    max_margin_fraction: float = 0.10  # user-approved: max 10% equity as margin per position
    leverage_candidates: tuple = (1, 2, 3, 5, 10, 20, 25, 50, 75, 100, 125)
    fee_rate: float = 0.0005            # verify actual Binance fee tier
    slippage_rate: float = 0.0002
    population_size: int = 36
    generations: int = 12
    elite_fraction: float = 0.20
    mutation_rate: float = 0.35
    train_fraction: float = 0.55
    validation_fraction: float = 0.20
    min_validation_trades: int = 12
    min_test_trades: int = 8
    paper_poll_seconds: int = 20
    research_every_hours: int = 24
    llm_enabled: bool = True
    llm_model: str = "gpt-5-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    max_funding_history: int = 1000
    maintenance_margin_rate: float = 0.005  # rough paper approximation; exchange tier is account-specific
    random_seed: int = 42
    db_path: str = DB_DEFAULT
    report_path: str = REPORT_DEFAULT


# ----------------------------- Data --------------------------------------
def _get(path: str, params: dict | None = None, timeout: int = 25):
    r = requests.get(BASE_URL + path, params=params, timeout=timeout,
                     headers={"User-Agent": "BTC-Autonomous-Research-Agent/2.0"})
    r.raise_for_status()
    return r.json()


def fetch_klines(interval: str, days: int, symbol: str = SYMBOL) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows: list = []
    while start_ms < end_ms:
        batch = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                           "startTime": start_ms, "endTime": end_ms, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 1
        if nxt <= start_ms:
            break
        start_ms = nxt
        if len(batch) < 1500:
            break
        time.sleep(0.10)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise RuntimeError("Binance returned no candles")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df.open_time), unit="ms", utc=True)
    df["close_time"] = pd.to_numeric(df.close_time)
    # Never use a still-forming candle.
    now_ms = int(time.time() * 1000)
    df = df[df.close_time < now_ms].drop_duplicates("open_time").sort_values("open_time")
    df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df["taker_buy_quote"] = pd.to_numeric(df["taker_buy_quote"], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]].reset_index(drop=True)


def fetch_market_snapshot(symbol: str = SYMBOL) -> dict:
    premium = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
    oi = _get("/fapi/v1/openInterest", {"symbol": symbol})
    ticker = _get("/fapi/v1/ticker/24hr", {"symbol": symbol})
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "mark_price": float(premium["markPrice"]), "index_price": float(premium["indexPrice"]),
            "funding_rate": float(premium["lastFundingRate"]),
            "next_funding_time": int(premium["nextFundingTime"]),
            "open_interest": float(oi["openInterest"]),
            "change_24h_pct": float(ticker["priceChangePercent"]),
            "volume_24h": float(ticker["volume"])}


# ---------------------- Derivatives / market data -----------------------
def fetch_funding_history(symbol: str, days: int, limit: int = 1000) -> pd.DataFrame:
    """Public historical funding events. Paginated; Binance returns up to 1000 per call."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows = []
    while start_ms < end_ms:
        batch = _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms,
                                                "endTime": end_ms, "limit": limit})
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= start_ms or len(batch) < limit:
            break
        start_ms = nxt
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_event_rate"])
    out = pd.DataFrame(rows)
    out["funding_time"] = pd.to_datetime(pd.to_numeric(out["fundingTime"]), unit="ms", utc=True)
    out["funding_event_rate"] = pd.to_numeric(out["fundingRate"], errors="coerce")
    return out[["funding_time", "funding_event_rate"]].drop_duplicates("funding_time").sort_values("funding_time")


def fetch_open_interest_history(symbol: str, period: str = "1h") -> pd.DataFrame:
    """Binance OI history is limited by endpoint retention; usually recent ~30 days."""
    try:
        rows = _get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": 500})
    except Exception as exc:
        logging.warning("OI history unavailable: %s", exc)
        return pd.DataFrame(columns=["open_time", "oi_change"])
    if not rows:
        return pd.DataFrame(columns=["open_time", "oi_change"])
    out = pd.DataFrame(rows)
    out["open_time"] = pd.to_datetime(pd.to_numeric(out["timestamp"]), unit="ms", utc=True)
    out["sum_open_interest"] = pd.to_numeric(out["sumOpenInterest"], errors="coerce")
    out["oi_change"] = out["sum_open_interest"].pct_change(3)
    return out[["open_time", "oi_change"]].dropna().sort_values("open_time")


def enrich_derivatives_data(candles: pd.DataFrame, symbol: str, days: int) -> pd.DataFrame:
    """Attach funding events exactly to their funding candle and recent OI changes."""
    x = candles.sort_values("open_time").copy()
    try:
        funding = fetch_funding_history(symbol, days)
        if not funding.empty:
            x = x.merge(funding, left_on="open_time", right_on="funding_time", how="left")
            x = x.drop(columns=["funding_time"])
            funding_curve = funding.rename(columns={"funding_time": "open_time", "funding_event_rate": "funding_rate"})
            x = pd.merge_asof(x.sort_values("open_time"), funding_curve.sort_values("open_time"), on="open_time", direction="backward")
        else:
            x["funding_event_rate"] = np.nan
            x["funding_rate"] = np.nan
    except Exception as exc:
        logging.warning("Funding history unavailable; research continues without it: %s", exc)
        x["funding_event_rate"] = np.nan
        x["funding_rate"] = np.nan
    try:
        oi = fetch_open_interest_history(symbol)
        if not oi.empty:
            x = pd.merge_asof(x.sort_values("open_time"), oi.sort_values("open_time"), on="open_time", direction="backward")
        else:
            x["oi_change"] = np.nan
    except Exception as exc:
        logging.warning("OI enrichment skipped: %s", exc)
        x["oi_change"] = np.nan
    return x.reset_index(drop=True)


def enrich_multi_timeframe(candles: pd.DataFrame, symbol: str, days: int) -> pd.DataFrame:
    """Causal 4h/1d trend context: higher timeframe values become available only at close."""
    x = candles.sort_values("open_time").copy()
    for interval, duration, prefix in [("4h", pd.Timedelta(hours=4), "htf4"),
                                       ("1d", pd.Timedelta(days=1), "htf1d")]:
        try:
            higher = fetch_klines(interval, days, symbol)
            higher["fast"] = higher.close.ewm(span=20, adjust=False, min_periods=20).mean()
            higher["slow"] = higher.close.ewm(span=50, adjust=False, min_periods=50).mean()
            higher["bull"] = higher.fast > higher.slow
            higher["available_at"] = higher.open_time + duration
            higher = higher[["available_at", "bull"]].dropna().sort_values("available_at")
            higher = higher.rename(columns={"bull": f"{prefix}_bull"})
            x = pd.merge_asof(x.sort_values("open_time"), higher,
                              left_on="open_time", right_on="available_at", direction="backward")
            x = x.drop(columns=["available_at"])
        except Exception as exc:
            logging.warning("%s context unavailable: %s", interval, exc)
            x[f"{prefix}_bull"] = np.nan
    return x.reset_index(drop=True)


# -------------------------- Feature engineering -------------------------
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
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
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
    x["vwap_rolling"] = (typical * x.volume).rolling(48).sum() / x.volume.rolling(48).sum().replace(0, np.nan)
    x["vwap_z"] = (c - x.vwap_rolling) / x.atr.replace(0, np.nan)
    x["atr_pct_rank"] = x.atr_pct.rolling(240, min_periods=60).rank(pct=True)
    return x.replace([np.inf, -np.inf], np.nan)


# -------------------------- Strategy genome -----------------------------
FAMILIES = ("trend_cross", "breakout", "pullback", "mean_revert", "momentum", "volatility_breakout", "vwap_reversion", "funding_contrarian", "oi_breakout", "range_compression", "orderflow_imbalance")

@dataclass
class Genome:
    family: str
    fast: int = 12
    slow: int = 50
    rsi_low: int = 30
    rsi_high: int = 70
    breakout_n: int = 20
    atr_stop: float = 2.0
    reward_r: float = 2.0
    volume_filter: float = 0.0
    trend_filter: bool = True
    max_atr_pct: float = 0.06
    leverage: int = 2
    funding_abs_max: float = 0.001
    oi_change_min: float = 0.0
    volatility_mode: str = "any"
    mtf_filter: bool = False

    @property
    def genome_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def random_genome(rng: random.Random, cfg: Config) -> Genome:
    fast = rng.choice([5, 8, 10, 12, 16, 20, 24])
    slow = rng.choice([30, 40, 50, 75, 100, 150, 200])
    if slow <= fast:
        slow = fast + 20
    low = rng.choice([20, 25, 30, 35, 40])
    high = rng.choice([60, 65, 70, 75, 80])
    return Genome(family=rng.choice(FAMILIES), fast=fast, slow=slow,
                  rsi_low=low, rsi_high=high,
                  breakout_n=rng.choice([10, 15, 20, 30, 40, 55]),
                  atr_stop=rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0]),
                  reward_r=rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]),
                  volume_filter=rng.choice([0.0, 1.0, 1.2, 1.5]),
                  trend_filter=rng.choice([True, False]),
                  max_atr_pct=rng.choice([0.025, 0.04, 0.06, 0.08]),
                  leverage=rng.choice(cfg.leverage_candidates),
                  funding_abs_max=rng.choice([0.0002, 0.0005, 0.001, 0.002]),
                  oi_change_min=rng.choice([0.0, 0.005, 0.01, 0.02]),
                  volatility_mode=rng.choice(["any", "high", "low"]),
                  mtf_filter=rng.choice([True, False]))


def mutate(g: Genome, rng: random.Random, cfg: Config) -> Genome:
    d = asdict(g)
    if rng.random() < .45: d["family"] = rng.choice(FAMILIES)
    if rng.random() < .50: d["fast"] = rng.choice([5, 8, 10, 12, 16, 20, 24, 30])
    if rng.random() < .50: d["slow"] = rng.choice([30, 40, 50, 75, 100, 150, 200])
    if d["slow"] <= d["fast"]: d["slow"] = d["fast"] + 20
    if rng.random() < .40: d["rsi_low"] = rng.choice([20, 25, 30, 35, 40])
    if rng.random() < .40: d["rsi_high"] = rng.choice([60, 65, 70, 75, 80])
    if rng.random() < .45: d["breakout_n"] = rng.choice([10, 15, 20, 30, 40, 55])
    if rng.random() < .45: d["atr_stop"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    if rng.random() < .45: d["reward_r"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0])
    if rng.random() < .35: d["volume_filter"] = rng.choice([0.0, 1.0, 1.2, 1.5])
    if rng.random() < .25: d["trend_filter"] = not d["trend_filter"]
    if rng.random() < .35: d["max_atr_pct"] = rng.choice([0.025, 0.04, 0.06, 0.08])
    if rng.random() < .30: d["leverage"] = rng.choice(cfg.leverage_candidates)
    if rng.random() < .35: d["funding_abs_max"] = rng.choice([0.0002, 0.0005, 0.001, 0.002])
    if rng.random() < .35: d["oi_change_min"] = rng.choice([0.0, 0.005, 0.01, 0.02])
    if rng.random() < .25: d["volatility_mode"] = rng.choice(["any", "high", "low"])
    if rng.random() < .25: d["mtf_filter"] = not d["mtf_filter"]
    return Genome(**d)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da, db = asdict(a), asdict(b)
    child = {k: (da[k] if rng.random() < .5 else db[k]) for k in da}
    if child["slow"] <= child["fast"]: child["slow"] = child["fast"] + 20
    return Genome(**child)


def make_signal(row: pd.Series, prev: pd.Series, g: Genome) -> Optional[int]:
    needed = ["atr", "rsi", "ema_fast", "ema_slow", "ema_long", "atr_pct"]
    if any(pd.isna(row.get(k)) for k in needed) or row.atr <= 0 or row.atr_pct > g.max_atr_pct:
        return None
    if g.volume_filter and (pd.isna(row.vol_ratio) or row.vol_ratio < g.volume_filter):
        return None
    bull = row.ema_fast > row.ema_slow
    bear = row.ema_fast < row.ema_slow
    long_trend = bull and row.close > row.ema_long
    short_trend = bear and row.close < row.ema_long
    if g.volatility_mode == "high" and row.atr_pct < 0.015: return None
    if g.volatility_mode == "low" and row.atr_pct > 0.015: return None
    if g.family == "trend_cross":
        if row.ema_fast > row.ema_slow and prev.ema_fast <= prev.ema_slow and (not g.trend_filter or long_trend): return 1
        if row.ema_fast < row.ema_slow and prev.ema_fast >= prev.ema_slow and (not g.trend_filter or short_trend): return -1
    elif g.family == "breakout":
        hh = row.get("don_high"); ll = row.get("don_low")
        if pd.notna(hh) and row.close > hh and (not g.trend_filter or long_trend): return 1
        if pd.notna(ll) and row.close < ll and (not g.trend_filter or short_trend): return -1
    elif g.family == "pullback":
        if long_trend and prev.close < prev.ema_fast and row.close > row.ema_fast and g.rsi_low <= row.rsi <= g.rsi_high: return 1
        if short_trend and prev.close > prev.ema_fast and row.close < row.ema_fast and (100-g.rsi_low) >= row.rsi >= (100-g.rsi_high): return -1
    elif g.family == "mean_revert":
        if row.get("bb_z", np.nan) <= -1.8 and row.rsi < g.rsi_low: return 1
        if row.get("bb_z", np.nan) >= 1.8 and row.rsi > g.rsi_high: return -1
    elif g.family == "momentum":
        if row.ret_5 > .012 and row.ret_20 > .02 and (not g.trend_filter or long_trend): return 1
        if row.ret_5 < -.012 and row.ret_20 < -.02 and (not g.trend_filter or short_trend): return -1
    elif g.family == "volatility_breakout":
        if row.close > row.don_high and row.atr_pct > .012 and (not g.trend_filter or long_trend): return 1
        if row.close < row.don_low and row.atr_pct > .012 and (not g.trend_filter or short_trend): return -1
    elif g.family == "vwap_reversion":
        if pd.notna(row.get("vwap_z")) and row.vwap_z < -1.5 and row.rsi < g.rsi_low: return 1
        if pd.notna(row.get("vwap_z")) and row.vwap_z > 1.5 and row.rsi > g.rsi_high: return -1
    elif g.family == "funding_contrarian":
        fr = row.get("funding_rate", np.nan)
        if pd.notna(fr) and fr >= g.funding_abs_max and row.rsi > 55: return -1
        if pd.notna(fr) and fr <= -g.funding_abs_max and row.rsi < 45: return 1
    elif g.family == "oi_breakout":
        oi = row.get("oi_change", np.nan)
        if pd.notna(oi) and oi >= g.oi_change_min and row.close > row.don_high: return 1
        if pd.notna(oi) and oi >= g.oi_change_min and row.close < row.don_low: return -1
    elif g.family == "range_compression":
        if row.atr_pct < 0.012 and row.close > row.don_high: return 1
        if row.atr_pct < 0.012 and row.close < row.don_low: return -1
    elif g.family == "orderflow_imbalance":
        flow = row.get("taker_buy_ratio", np.nan)
        if pd.notna(flow) and flow > 0.58 and row.vol_ratio > 1.1 and (not g.trend_filter or long_trend): return 1
        if pd.notna(flow) and flow < 0.42 and row.vol_ratio > 1.1 and (not g.trend_filter or short_trend): return -1
    return None


def mtf_side_allowed(row: pd.Series, side: Optional[int], enabled: bool) -> bool:
    if side is None or not enabled:
        return side is not None
    h4, d1 = row.get("htf4_bull", np.nan), row.get("htf1d_bull", np.nan)
    if pd.isna(h4) or pd.isna(d1):
        return False
    return bool(h4) and bool(d1) if side == 1 else (not bool(h4) and not bool(d1))


# -------------------------- Backtest -------------------------------------
def backtest(df: pd.DataFrame, g: Genome, cfg: Config, score_start: int = 201) -> dict:
    if len(df) < 250:
        return {"trades": [], "metrics": {"net_return_pct": -999, "profit_factor": 0,
                "max_drawdown_pct": 100, "trade_count": 0, "expectancy": -999, "score": -999}}
    x = df.reset_index(drop=True)
    equity = cfg.initial_capital
    peak = equity
    max_dd = 0.0
    pos = None
    trades = []
    fee, slip = cfg.fee_rate, cfg.slippage_rate
    for i in range(max(201, score_start), len(x)):
        row, prev = x.iloc[i], x.iloc[i-1]
        # Existing position first. If both stop and target are inside one candle,
        # assume the stop happened first (conservative OHLC-only approximation).
        if pos:
            # Funding is charged/credited at historical funding timestamps when available.
            fr = row.get("funding_event_rate", np.nan)
            if pd.notna(fr):
                funding_cashflow = -pos["side"] * float(fr) * pos["entry"] * pos["qty"]
                equity += funding_cashflow
                pos["funding_pnl"] = pos.get("funding_pnl", 0.0) + funding_cashflow
            liq_hit = (row.low <= pos["liq"] if pos["side"] == 1 else row.high >= pos["liq"])
            stop_hit = row.low <= pos["sl"] if pos["side"] == 1 else row.high >= pos["sl"]
            tp_hit = row.high >= pos["tp"] if pos["side"] == 1 else row.low <= pos["tp"]
            exit_reason, exit_px = None, None
            if liq_hit: exit_reason, exit_px = "LIQUIDATION_ESTIMATE", pos["liq"]
            elif stop_hit: exit_reason, exit_px = "SL", pos["sl"]
            elif tp_hit: exit_reason, exit_px = "TP", pos["tp"]
            elif i == len(x)-1: exit_reason, exit_px = "END", row.close
            if exit_px is not None:
                exit_fill = exit_px * (1 - slip * pos["side"])
                gross = (exit_fill - pos["entry"]) * pos["qty"] * pos["side"]
                exit_fee = abs(exit_fill * pos["qty"]) * fee
                trading_pnl = gross - pos["entry_fee"] - exit_fee
                pnl = trading_pnl + pos.get("funding_pnl", 0.0)
                equity += trading_pnl
                trades.append({"entry_time": str(pos["time"]), "exit_time": str(row.open_time),
                    "side": "LONG" if pos["side"] == 1 else "SHORT", "entry": pos["entry"],
                    "exit": exit_fill, "qty": pos["qty"], "leverage": pos["leverage"],
                    "margin": pos["margin"], "pnl": pnl, "funding_pnl": pos.get("funding_pnl", 0.0), "fees": pos["entry_fee"]+exit_fee,
                    "reason": exit_reason, "genome_id": g.genome_id})
                pos = None
        marked_equity = equity
        if pos:
            unrealized = (float(row.close) - pos["entry"]) * pos["qty"] * pos["side"]
            estimated_exit_fee = abs(float(row.close) * pos["qty"]) * fee
            marked_equity += unrealized - estimated_exit_fee
        peak = max(peak, marked_equity)
        max_dd = max(max_dd, (peak-marked_equity)/peak if peak else 0)
        if pos or i == len(x)-1:
            continue
        side = make_signal(row, prev, g)
        if not mtf_side_allowed(row, side, g.mtf_filter):
            continue
        # Signal uses the just-closed candle; fill is next candle OPEN.
        if i+1 >= len(x):
            continue
        nxt = x.iloc[i+1]
        entry = float(nxt.open) * (1 + slip*side)
        stop_dist = float(row.atr) * g.atr_stop
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            continue
        # Agent chooses leverage from candidate set; margin allocation is capped at 10% equity.
        leverage = int(g.leverage)
        margin = equity * cfg.max_margin_fraction
        notional = margin * leverage
        qty = notional / entry
        sl = entry - side*stop_dist
        tp = entry + side*stop_dist*g.reward_r
        mmr = cfg.maintenance_margin_rate
        liq = (entry * (1 - 1/leverage + mmr) if side == 1 else
               entry * (1 + 1/leverage - mmr)) if leverage > 1 else (entry * 0.01 if side == 1 else entry * 100)
        entry_fee = notional * fee
        # Reject setups whose nominal target cannot cover round-trip fees and slippage.
        expected_move = stop_dist*g.reward_r*qty
        expected_cost = entry_fee + abs(tp*qty)*fee + notional*slip + abs(tp*qty)*slip
        if expected_move <= expected_cost:
            continue
        pos = {"side": side, "entry": entry, "time": nxt.open_time, "qty": qty,
               "sl": sl, "tp": tp, "entry_fee": entry_fee, "leverage": leverage,
               "margin": margin, "funding_pnl": 0.0, "liq": liq}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    ret = (equity/cfg.initial_capital-1)*100
    pf = gp/gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    expectancy = float(np.mean(pnls)) if pnls else -999.0
    # Penalize drawdown and low sample count; no single metric can dominate.
    score = ret - max_dd*100*1.5 + min(pf, 4)*2 + min(len(trades), 100)*0.03
    if len(trades) < 8: score -= 25
    if not np.isfinite(pf): pf = 4.0
    return {"trades": trades, "metrics": {"net_return_pct": ret, "profit_factor": pf,
        "max_drawdown_pct": max_dd*100, "trade_count": len(trades),
        "win_rate_pct": len(wins)/len(trades)*100 if trades else 0,
        "expectancy": expectancy, "score": score, "ending_equity": equity}}


def evaluate_genome(df: pd.DataFrame, g: Genome, cfg: Config) -> dict:
    n = len(df)
    a = int(n*cfg.train_fraction)
    b = int(n*(cfg.train_fraction+cfg.validation_fraction))
    # Each slice retains pre-slice indicator warm-up, but trades are scored only
    # after its boundary. Simpler conservative implementation uses full causal
    # features then slices; no future data enters rolling indicators.
    train = backtest(df.iloc[:a], g, cfg)
    val_offset = a - max(0, a-250)
    test_offset = b - max(0, b-250)
    val = backtest(df.iloc[max(0,a-250):b], g, cfg, score_start=val_offset)
    test = backtest(df.iloc[max(0,b-250):], g, cfg, score_start=test_offset)
    vm, tm = val["metrics"], test["metrics"]
    qualified = (vm["trade_count"] >= cfg.min_validation_trades and
                 tm["trade_count"] >= cfg.min_test_trades and
                 vm["expectancy"] > 0 and tm["expectancy"] > 0 and
                 vm["profit_factor"] > 1.05 and tm["profit_factor"] > 1.05)
    # Robust score rewards validation AND test, not training fit.
    robust = min(vm["score"], tm["score"]) if qualified else min(vm["score"], tm["score"])-50
    return {"genome": asdict(g), "genome_id": g.genome_id,
            "train": train["metrics"], "validation": vm, "test": tm,
            "qualified": qualified, "robust_score": robust}


# --------------------------- LLM researcher ------------------------------
class LLMResearcher:
    """Optional OpenAI-compatible research interface. It proposes genomes only;
    all proposals are schema-checked and must pass deterministic validation."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", cfg.llm_base_url).rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", cfg.llm_model)

    def propose(self, market_summary: dict, count: int = 12) -> list[Genome]:
        if not self.cfg.llm_enabled or not self.api_key:
            logging.info("LLM researcher disabled: set OPENAI_API_KEY to enable external research proposals")
            return []
        allowed = {"family": list(FAMILIES), "fast": [5,8,10,12,16,20,24,30],
                   "slow": [30,40,50,75,100,150,200], "rsi_low": [20,25,30,35,40],
                   "rsi_high": [60,65,70,75,80], "breakout_n": [10,15,20,30,40,55],
                   "atr_stop": [1.2,1.5,1.8,2.0,2.5,3.0], "reward_r": [1.2,1.5,1.8,2,2.5,3,4],
                   "volume_filter": [0,1,1.2,1.5], "trend_filter": [True,False],
                   "max_atr_pct": [0.025,0.04,0.06,0.08], "leverage": list(self.cfg.leverage_candidates),
                   "funding_abs_max": [0.0002,0.0005,0.001,0.002], "oi_change_min": [0,0.005,0.01,0.02],
                   "volatility_mode": ["any","high","low"], "mtf_filter": [True,False]}
        prompt = {
            "task": "Propose diverse, testable BTCUSDT perpetual futures strategy hypotheses. Do not claim profitability. Return JSON only.",
            "market_summary": market_summary,
            "allowed_values": allowed,
            "requirements": ["diverse families and parameter combinations", "avoid lookahead", "prefer interpretable hypotheses", "do not optimize only for recent noise"],
            "output_schema": {"proposals": [{"family": "one allowed family", "fast": 12, "slow": 50,
                "rsi_low": 30, "rsi_high": 70, "breakout_n": 20, "atr_stop": 2.0, "reward_r": 2.0,
                "volume_filter": 0, "trend_filter": True, "max_atr_pct": 0.06, "leverage": 2,
                "funding_abs_max": 0.001, "oi_change_min": 0.005, "volatility_mode": "any", "mtf_filter": False}]}
        }
        try:
            r = requests.post(self.base_url + "/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "temperature": 0.8, "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": "You are a quantitative research scientist. Never assert a strategy is profitable without out-of-sample evidence."},
                                   {"role": "user", "content": json.dumps(prompt)}]}, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            obj = json.loads(content)
            proposals = []
            for item in obj.get("proposals", [])[:count]:
                try:
                    g = Genome(**item)
                    if g.family not in FAMILIES or g.slow <= g.fast or g.leverage not in self.cfg.leverage_candidates:
                        continue
                    proposals.append(g)
                except (TypeError, ValueError):
                    continue
            logging.info("LLM researcher returned %d validated strategy proposals", len(proposals))
            return proposals
        except Exception as exc:
            logging.warning("LLM research call failed; evolutionary search continues: %s", exc)
            return []


# ------------------------ Evolution and selection -----------------------
class StrategyEvolution:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)

    def research(self, df: pd.DataFrame) -> list[dict]:
        init_db(self.cfg.db_path)
        population = [random_genome(self.rng, self.cfg) for _ in range(self.cfg.population_size)]
        # LLM is a hypothesis generator, never a performance judge.
        summary = {"bars": len(df), "last_close": float(df.close.iloc[-1]),
                   "atr_pct": float(df.atr_pct.iloc[-1]) if pd.notna(df.atr_pct.iloc[-1]) else None,
                   "funding_rate": float(df.funding_rate.iloc[-1]) if "funding_rate" in df and pd.notna(df.funding_rate.iloc[-1]) else None}
        population.extend(LLMResearcher(self.cfg).propose(summary, count=max(8, self.cfg.population_size // 3)))
        persisted = db_get(self.cfg.db_path, "paper_state", {}) or {}
        if persisted.get("champion"):
            try: population.append(Genome(**persisted["champion"]))
            except Exception: pass
        all_results: dict[str, dict] = {}
        for gen in range(self.cfg.generations):
            results = []
            for g in population:
                if g.genome_id not in all_results:
                    all_results[g.genome_id] = evaluate_genome(df, g, self.cfg)
                results.append(all_results[g.genome_id])
            results.sort(key=lambda z: z["robust_score"], reverse=True)
            logging.info("Generation %d/%d | best=%s | qualified=%s", gen+1,
                         self.cfg.generations, results[0]["robust_score"],
                         sum(r["qualified"] for r in results))
            elite_n = max(2, int(self.cfg.population_size*self.cfg.elite_fraction))
            elites = [Genome(**r["genome"]) for r in results[:elite_n]]
            new_pop = elites[:]
            while len(new_pop) < self.cfg.population_size:
                if self.rng.random() < .25:
                    child = random_genome(self.rng, self.cfg)
                else:
                    p1, p2 = self.rng.choice(elites), self.rng.choice(elites)
                    child = crossover(p1, p2, self.rng)
                    if self.rng.random() < self.cfg.mutation_rate:
                        child = mutate(child, self.rng, self.cfg)
                new_pop.append(child)
            population = new_pop
        ranked = sorted(all_results.values(), key=lambda z: z["robust_score"], reverse=True)
        return ranked


# ----------------------------- Persistence -------------------------------
def init_db(path: str):
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS paper_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS research_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, payload TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS paper_state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")


def db_set(path: str, key: str, value: Any):
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)))


def db_get(path: str, key: str, default=None):
    with sqlite3.connect(path) as con:
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def save_research(path: str, report: dict):
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO research_runs(ts,payload) VALUES(?,?)",
                    (datetime.now(timezone.utc).isoformat(), json.dumps(report)))


# ---------------------- Paper execution realism -------------------------
def estimate_depth_fill(symbol: str, side: int, qty: float, fallback_price: float,
                        fallback_slippage: float) -> tuple[float, float, str]:
    """Estimate market fill from public order-book depth; never submits an order.
    side=+1 buys asks, side=-1 sells bids. Returns (VWAP, filled_qty, source)."""
    try:
        book = _get("/fapi/v1/depth", {"symbol": symbol, "limit": 100}, timeout=10)
        levels = book["asks"] if side == 1 else book["bids"]
        remaining, notional = qty, 0.0
        for px_s, q_s in levels:
            px, avail = float(px_s), float(q_s)
            take = min(remaining, avail)
            notional += take * px
            remaining -= take
            if remaining <= 1e-12:
                break
        filled = qty - remaining
        if filled > 0:
            vwap = notional / filled
            if remaining > 1e-12:
                # Unfilled tail receives an adverse fallback estimate, not a fake full depth fill.
                tail_px = fallback_price * (1 + fallback_slippage * side)
                vwap = (notional + remaining * tail_px) / qty
            return vwap, qty, "public_depth_vwap_with_tail_estimate"
    except Exception as exc:
        logging.debug("Depth snapshot unavailable, using fallback slippage: %s", exc)
    return fallback_price * (1 + fallback_slippage * side), qty, "fallback_slippage_model"


# ----------------------------- Paper trading -----------------------------
class PaperTrader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        init_db(cfg.db_path)
        self.state = db_get(cfg.db_path, "paper_state", None) or {
            "equity": cfg.initial_capital, "position": None, "last_candle": None,
            "champion": None, "trade_count": 0}

    def save(self):
        db_set(self.cfg.db_path, "paper_state", self.state)

    def set_champion(self, genome: Genome):
        self.state["champion"] = asdict(genome)
        self.save()

    def process(self, df: pd.DataFrame):
        if not self.state.get("champion"):
            logging.warning("No qualified Champion. Paper trader will not open positions.")
            return
        g = Genome(**self.state["champion"])
        x = add_features(df).reset_index(drop=True)
        last = x.iloc[-1]
        candle_id = str(last.open_time)
        if self.state.get("last_candle") == candle_id:
            return

        # A signal created on the previous closed candle is filled at this
        # candle's open. Process the pending fill BEFORE creating a new signal.
        pending = self.state.get("pending")
        if pending and pending["signal_time"] != candle_id and self.state.get("position") is None:
            reference_entry = float(last.open)
            stop_dist = pending["atr"] * pending.get("atr_stop", g.atr_stop)
            reward_r = pending.get("reward_r", g.reward_r)
            leverage = int(pending["leverage"])
            margin = self.state["equity"] * self.cfg.max_margin_fraction
            notional = margin * leverage
            qty = notional / reference_entry
            entry, qty, fill_source = estimate_depth_fill(self.cfg.symbol, pending["side"], qty,
                                                          reference_entry, self.cfg.slippage_rate)
            logging.info("Paper entry fill model: %s", fill_source)
            sl = entry - pending["side"] * stop_dist
            tp = entry + pending["side"] * stop_dist * reward_r
            mmr = self.cfg.maintenance_margin_rate
            liq = (entry * (1 - 1/leverage + mmr) if pending["side"] == 1 else
                   entry * (1 + 1/leverage - mmr)) if leverage > 1 else (entry * 0.01 if pending["side"] == 1 else entry * 100)
            entry_fee = notional * self.cfg.fee_rate
            self.state["position"] = {"side": pending["side"], "entry": entry,
                "entry_time": candle_id, "qty": qty, "margin": margin,
                "notional": notional, "leverage": leverage, "sl": sl, "tp": tp, "liq": liq,
                "entry_fee": entry_fee, "genome_id": pending["genome_id"], "funding_pnl": 0.0}
            self.state["pending"] = None
            logging.info("PAPER OPEN %s entry=%.2f lev=%sx margin=%.2f qty=%.6f SL=%.2f TP=%.2f",
                "LONG" if pending["side"] == 1 else "SHORT", entry, leverage, margin, qty, sl, tp)

        # Check the open position against this candle, including the entry candle.
        pos = self.state.get("position")
        if pos:
            fr = last.get("funding_event_rate", np.nan)
            if pd.notna(fr):
                funding_cashflow = -pos["side"] * float(fr) * pos["entry"] * pos["qty"]
                self.state["equity"] += funding_cashflow
                pos["funding_pnl"] = pos.get("funding_pnl", 0.0) + funding_cashflow
            liq_hit = last.low <= pos.get("liq", -np.inf) if pos["side"] == 1 else last.high >= pos.get("liq", np.inf)
            stop_hit = last.low <= pos["sl"] if pos["side"] == 1 else last.high >= pos["sl"]
            tp_hit = last.high >= pos["tp"] if pos["side"] == 1 else last.low <= pos["tp"]
            reason = "LIQUIDATION_ESTIMATE" if liq_hit else ("SL" if stop_hit else ("TP" if tp_hit else None))
            if reason:
                px = pos.get("liq") if reason == "LIQUIDATION_ESTIMATE" else (pos["sl"] if reason == "SL" else pos["tp"])
                fill, _, fill_source = estimate_depth_fill(self.cfg.symbol, -pos["side"], pos["qty"],
                                                           px, self.cfg.slippage_rate)
                logging.info("Paper exit fill model: %s", fill_source)
                gross = (fill - pos["entry"]) * pos["qty"] * pos["side"]
                exit_fee = abs(fill * pos["qty"]) * self.cfg.fee_rate
                trading_pnl = gross - pos["entry_fee"] - exit_fee
                pnl = trading_pnl + pos.get("funding_pnl", 0.0)
                self.state["equity"] += trading_pnl
                trade = {**pos, "exit": fill, "exit_reason": reason, "pnl": pnl,
                         "exit_time": candle_id, "exit_fee": exit_fee}
                with sqlite3.connect(self.cfg.db_path) as con:
                    con.execute("INSERT INTO paper_trades(payload) VALUES(?)", (json.dumps(trade),))
                self.state["trade_count"] += 1
                self.state["position"] = None
                logging.info("PAPER CLOSE %s pnl=%+.4f equity=%.2f", reason, pnl, self.state["equity"])

        # Only create a fresh signal after handling the previous pending order.
        # Never overwrite an unfilled pending signal.
        if self.state.get("position") is None and self.state.get("pending") is None and len(x) > 202:
            side = make_signal(last, x.iloc[-2], g)
            if mtf_side_allowed(last, side, g.mtf_filter):
                self.state["pending"] = {"side": side, "signal_time": candle_id,
                    "atr": float(last.atr), "genome_id": g.genome_id,
                    "leverage": int(g.leverage), "atr_stop": g.atr_stop, "reward_r": g.reward_r}
        self.state["last_candle"] = candle_id
        self.save()


def maybe_promote(ranked: list[dict], current: Optional[dict]) -> Optional[Genome]:
    qualified = [r for r in ranked if r["qualified"]]
    if not qualified:
        logging.warning("No candidate passed independent validation. Keep current Champion or stay flat.")
        return Genome(**current) if current else None
    best = qualified[0]
    if current:
        # Avoid needless champion churn: require clear robust-score improvement.
        old_id = Genome(**current).genome_id
        old = next((r for r in ranked if r["genome_id"] == old_id and r["qualified"]), None)
        if old and best["robust_score"] < old["robust_score"] + 3:
            return Genome(**current)
    return Genome(**best["genome"])


def run_research(cfg: Config) -> dict:
    logging.info("Fetching %s %s candles for %d days", cfg.symbol, cfg.interval, cfg.history_days)
    raw = fetch_klines(cfg.interval, cfg.history_days, cfg.symbol)
    raw = enrich_derivatives_data(raw, cfg.symbol, cfg.history_days)
    raw = enrich_multi_timeframe(raw, cfg.symbol, cfg.history_days)
    if len(raw) < 1500:
        raise RuntimeError(f"Insufficient history: {len(raw)} candles")
    data = add_features(raw)
    evolution = StrategyEvolution(cfg)
    ranked = evolution.research(data)
    trader = PaperTrader(cfg)
    current = trader.state.get("champion")
    selected = maybe_promote(ranked, current)
    if selected:
        trader.set_champion(selected)
    report = {"version": "3.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(cfg), "leverage_candidates": list(cfg.leverage_candidates)},
        "market_snapshot": fetch_market_snapshot(cfg.symbol),
        "bars": len(data), "champion": asdict(selected) if selected else None,
        "champion_evaluation": next((r for r in ranked if selected and r["genome_id"]==selected.genome_id), None),
        "top_candidates": ranked[:20],
        "qualification_note": "No candidate is promoted without positive validation and test expectancy, PF > 1.05, minimum trade counts; drawdown is included as a continuous score penalty rather than a separate hard capital stop.",
        "execution_mode": "PAPER_ONLY; live trading is not implemented"}
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    save_research(cfg.db_path, report)
    logging.info("Research complete. Champion=%s report=%s", selected.genome_id if selected else None, cfg.report_path)
    return report


def run_paper_loop(cfg: Config):
    trader = PaperTrader(cfg)
    logging.info("Starting PAPER loop only. Ctrl+C to stop. DB=%s", cfg.db_path)
    last_research = 0.0
    while True:
        try:
            # Continuous strategy evolution; never auto-enable live execution.
            if time.time() - last_research >= cfg.research_every_hours * 3600:
                report = run_research(cfg)
                last_research = time.time()
                trader = PaperTrader(cfg)  # reload newly promoted Champion from persistent state
                logging.info("Scheduled evolution completed; champion=%s", (report.get("champion") or {}).get("family"))
            bars = fetch_klines(cfg.interval, max(30, min(cfg.history_days, 90)), cfg.symbol)
            bars = enrich_derivatives_data(bars, cfg.symbol, min(cfg.history_days, 90))
            bars = enrich_multi_timeframe(bars, cfg.symbol, min(cfg.history_days, 90))
            # Replay every unseen closed candle in order instead of silently skipping
            # candles after downtime. process() is idempotent by candle timestamp.
            last_seen = trader.state.get("last_candle")
            if last_seen:
                unseen_idx = [i for i, ts in enumerate(bars.open_time.astype(str).tolist()) if ts > str(last_seen)]
                for idx in unseen_idx:
                    trader.process(bars.iloc[max(0, idx-205):idx+1])
            else:
                trader.process(bars)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logging.exception("Paper loop error: %s", exc)
        time.sleep(cfg.paper_poll_seconds)


def main():
    p = argparse.ArgumentParser(description="BTCUSDT autonomous strategy research + paper trading (no live orders)")
    p.add_argument("--mode", choices=["research", "paper", "snapshot"], default="research")
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--interval", default="1h")
    p.add_argument("--db", default=DB_DEFAULT)
    p.add_argument("--report", default=REPORT_DEFAULT)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--population", type=int, default=36)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--research-hours", type=int, default=24)
    p.add_argument("--no-llm", action="store_true", help="Disable optional LLM strategy hypothesis generation")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(initial_capital=args.capital, history_days=args.days, interval=args.interval,
                 db_path=args.db, report_path=args.report, generations=args.generations,
                 population_size=args.population, random_seed=args.seed,
                 research_every_hours=args.research_hours, llm_enabled=not args.no_llm)
    init_db(cfg.db_path)
    if args.mode == "snapshot":
        print(json.dumps(fetch_market_snapshot(cfg.symbol), indent=2))
    elif args.mode == "research":
        report = run_research(cfg)
        print(json.dumps({"champion": report["champion"],
                          "champion_evaluation": report["champion_evaluation"],
                          "top_candidates": report["top_candidates"][:5]}, indent=2, default=str))
    else:
        run_paper_loop(cfg)

if __name__ == "__main__":
    main()
