"""Strategy genome DNA, signals, mutation and crossover."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config

FAMILIES = (
    "trend_cross",
    "breakout",
    "pullback",
    "mean_revert",
    "momentum",
    "volatility_breakout",
    "vwap_reversion",
    "funding_contrarian",
    "oi_breakout",
    "range_compression",
    "orderflow_imbalance",
)


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


def random_genome(rng: random.Random) -> Genome:
    fast = rng.choice([5, 8, 10, 12, 16, 20, 24])
    slow = rng.choice([30, 40, 50, 75, 100, 150, 200])
    if slow <= fast:
        slow = fast + 20
    low = rng.choice([20, 25, 30, 35, 40])
    high = rng.choice([60, 65, 70, 75, 80])
    return Genome(
        family=rng.choice(FAMILIES),
        fast=fast,
        slow=slow,
        rsi_low=low,
        rsi_high=high,
        breakout_n=rng.choice([10, 15, 20, 30, 40, 55]),
        atr_stop=rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0]),
        reward_r=rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]),
        volume_filter=rng.choice([0.0, 1.0, 1.2, 1.5]),
        trend_filter=rng.choice([True, False]),
        max_atr_pct=rng.choice([0.025, 0.04, 0.06, 0.08]),
        leverage=rng.choice(config.LEVERAGE_CANDIDATES),
        funding_abs_max=rng.choice([0.0002, 0.0005, 0.001, 0.002]),
        oi_change_min=rng.choice([0.0, 0.005, 0.01, 0.02]),
        volatility_mode=rng.choice(["any", "high", "low"]),
        mtf_filter=rng.choice([True, False]),
    )


def mutate(g: Genome, rng: random.Random) -> Genome:
    d = asdict(g)
    if rng.random() < 0.45:
        d["family"] = rng.choice(FAMILIES)
    if rng.random() < 0.50:
        d["fast"] = rng.choice([5, 8, 10, 12, 16, 20, 24, 30])
    if rng.random() < 0.50:
        d["slow"] = rng.choice([30, 40, 50, 75, 100, 150, 200])
    if d["slow"] <= d["fast"]:
        d["slow"] = d["fast"] + 20
    if rng.random() < 0.40:
        d["rsi_low"] = rng.choice([20, 25, 30, 35, 40])
    if rng.random() < 0.40:
        d["rsi_high"] = rng.choice([60, 65, 70, 75, 80])
    if rng.random() < 0.45:
        d["breakout_n"] = rng.choice([10, 15, 20, 30, 40, 55])
    if rng.random() < 0.45:
        d["atr_stop"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    if rng.random() < 0.45:
        d["reward_r"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0])
    if rng.random() < 0.35:
        d["volume_filter"] = rng.choice([0.0, 1.0, 1.2, 1.5])
    if rng.random() < 0.25:
        d["trend_filter"] = not d["trend_filter"]
    if rng.random() < 0.35:
        d["max_atr_pct"] = rng.choice([0.025, 0.04, 0.06, 0.08])
    if rng.random() < 0.30:
        d["leverage"] = rng.choice(config.LEVERAGE_CANDIDATES)
    if rng.random() < 0.35:
        d["funding_abs_max"] = rng.choice([0.0002, 0.0005, 0.001, 0.002])
    if rng.random() < 0.35:
        d["oi_change_min"] = rng.choice([0.0, 0.005, 0.01, 0.02])
    if rng.random() < 0.25:
        d["volatility_mode"] = rng.choice(["any", "high", "low"])
    if rng.random() < 0.25:
        d["mtf_filter"] = not d["mtf_filter"]
    return Genome(**d)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da, db = asdict(a), asdict(b)
    child = {k: (da[k] if rng.random() < 0.5 else db[k]) for k in da}
    if child["slow"] <= child["fast"]:
        child["slow"] = child["fast"] + 20
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
    if g.volatility_mode == "high" and row.atr_pct < 0.015:
        return None
    if g.volatility_mode == "low" and row.atr_pct > 0.015:
        return None
    if g.family == "trend_cross":
        if (
            row.ema_fast > row.ema_slow
            and prev.ema_fast <= prev.ema_slow
            and (not g.trend_filter or long_trend)
        ):
            return 1
        if (
            row.ema_fast < row.ema_slow
            and prev.ema_fast >= prev.ema_slow
            and (not g.trend_filter or short_trend)
        ):
            return -1
    elif g.family == "breakout":
        hh = row.get("don_high")
        ll = row.get("don_low")
        if pd.notna(hh) and row.close > hh and (not g.trend_filter or long_trend):
            return 1
        if pd.notna(ll) and row.close < ll and (not g.trend_filter or short_trend):
            return -1
    elif g.family == "pullback":
        if (
            long_trend
            and prev.close < prev.ema_fast
            and row.close > row.ema_fast
            and g.rsi_low <= row.rsi <= g.rsi_high
        ):
            return 1
        if (
            short_trend
            and prev.close > prev.ema_fast
            and row.close < row.ema_fast
            and (100 - g.rsi_low) >= row.rsi >= (100 - g.rsi_high)
        ):
            return -1
    elif g.family == "mean_revert":
        if row.get("bb_z", np.nan) <= -1.8 and row.rsi < g.rsi_low:
            return 1
        if row.get("bb_z", np.nan) >= 1.8 and row.rsi > g.rsi_high:
            return -1
    elif g.family == "momentum":
        if row.ret_5 > 0.012 and row.ret_20 > 0.02 and (not g.trend_filter or long_trend):
            return 1
        if (
            row.ret_5 < -0.012
            and row.ret_20 < -0.02
            and (not g.trend_filter or short_trend)
        ):
            return -1
    elif g.family == "volatility_breakout":
        if (
            row.close > row.don_high
            and row.atr_pct > 0.012
            and (not g.trend_filter or long_trend)
        ):
            return 1
        if (
            row.close < row.don_low
            and row.atr_pct > 0.012
            and (not g.trend_filter or short_trend)
        ):
            return -1
    elif g.family == "vwap_reversion":
        if pd.notna(row.get("vwap_z")) and row.vwap_z < -1.5 and row.rsi < g.rsi_low:
            return 1
        if pd.notna(row.get("vwap_z")) and row.vwap_z > 1.5 and row.rsi > g.rsi_high:
            return -1
    elif g.family == "funding_contrarian":
        fr = row.get("funding_rate", np.nan)
        if pd.notna(fr) and fr >= g.funding_abs_max and row.rsi > 55:
            return -1
        if pd.notna(fr) and fr <= -g.funding_abs_max and row.rsi < 45:
            return 1
    elif g.family == "oi_breakout":
        oi = row.get("oi_change", np.nan)
        if pd.notna(oi) and oi >= g.oi_change_min and row.close > row.don_high:
            return 1
        if pd.notna(oi) and oi >= g.oi_change_min and row.close < row.don_low:
            return -1
    elif g.family == "range_compression":
        if row.atr_pct < 0.012 and row.close > row.don_high:
            return 1
        if row.atr_pct < 0.012 and row.close < row.don_low:
            return -1
    elif g.family == "orderflow_imbalance":
        flow = row.get("taker_buy_ratio", np.nan)
        if (
            pd.notna(flow)
            and flow > 0.58
            and row.vol_ratio > 1.1
            and (not g.trend_filter or long_trend)
        ):
            return 1
        if (
            pd.notna(flow)
            and flow < 0.42
            and row.vol_ratio > 1.1
            and (not g.trend_filter or short_trend)
        ):
            return -1
    return None


def mtf_side_allowed(row: pd.Series, side: Optional[int], enabled: bool) -> bool:
    if side is None or not enabled:
        return side is not None
    # Align with every available HTF bias column (15m→1h/4h/1d; 1h→4h/1d)
    biases: list[bool] = []
    for col in ("htf1_bull", "htf4_bull", "htf1d_bull"):
        v = row.get(col, np.nan)
        if pd.notna(v):
            biases.append(bool(v))
    if not biases:
        return False
    return all(biases) if side == 1 else all(not b for b in biases)


def soft_entry_genome(g: Genome) -> tuple[Genome, list[str]]:
    """Cap volume / floor max ATR for paper live only — genome_id unchanged in storage."""
    notes: list[str] = []
    if not config.PAPER_SOFT_ENTRY:
        return g, notes
    d = asdict(g)
    changed = False
    if d.get("volume_filter", 0) > config.PAPER_SOFT_VOLUME_CAP:
        notes.append(f"volf {d['volume_filter']}→{config.PAPER_SOFT_VOLUME_CAP}")
        d["volume_filter"] = config.PAPER_SOFT_VOLUME_CAP
        changed = True
    if float(d.get("max_atr_pct") or 0) < config.PAPER_SOFT_MAX_ATR_FLOOR:
        notes.append(
            f"maxatr {float(d['max_atr_pct'])*100:.1f}%→{config.PAPER_SOFT_MAX_ATR_FLOOR*100:.1f}%"
        )
        d["max_atr_pct"] = config.PAPER_SOFT_MAX_ATR_FLOOR
        changed = True
    if not changed:
        return g, notes
    return Genome(**d), notes


def diagnose_entry(
    row: pd.Series,
    prev: pd.Series,
    g: Genome,
) -> dict:
    """Short entry decision + reason for Telegram tick notify."""
    live_g, soft_notes = soft_entry_genome(g)
    needed = ["atr", "rsi", "ema_fast", "ema_slow", "ema_long", "atr_pct"]
    if any(pd.isna(row.get(k)) for k in needed):
        return {"side": None, "action": "SKIP", "why": "thiếu indicator (warm-up)"}
    if row.atr <= 0:
        return {"side": None, "action": "SKIP", "why": "ATR không hợp lệ"}
    if row.atr_pct > live_g.max_atr_pct:
        return {
            "side": None,
            "action": "SKIP",
            "why": f"ATR% {row.atr_pct*100:.2f}% > max {live_g.max_atr_pct*100:.1f}%",
        }
    if live_g.volume_filter and (
        pd.isna(row.vol_ratio) or row.vol_ratio < live_g.volume_filter
    ):
        return {
            "side": None,
            "action": "SKIP",
            "why": (
                f"vol_ratio {float(row.vol_ratio) if pd.notna(row.vol_ratio) else 0:.2f}"
                f" < {live_g.volume_filter}"
            ),
        }
    if live_g.volatility_mode == "high" and row.atr_pct < 0.015:
        return {"side": None, "action": "SKIP", "why": "vol mode=high nhưng ATR% thấp"}
    if live_g.volatility_mode == "low" and row.atr_pct > 0.015:
        return {"side": None, "action": "SKIP", "why": "vol mode=low nhưng ATR% cao"}

    side = make_signal(row, prev, live_g)
    if side is None:
        return {
            "side": None,
            "action": "SKIP",
            "why": f"family={live_g.family}: chưa đủ điều kiện setup",
        }
    if not mtf_side_allowed(row, side, live_g.mtf_filter):
        direction = "LONG" if side == 1 else "SHORT"
        if live_g.mtf_filter:
            bits = []
            for label, col in (("1h", "htf1_bull"), ("4h", "htf4_bull"), ("1d", "htf1d_bull")):
                v = row.get(col, np.nan)
                if pd.notna(v):
                    bits.append(f"{label}={'↑' if bool(v) else '↓'}")
            htf = " ".join(bits) if bits else "thiếu HTF"
            return {
                "side": None,
                "action": "SKIP",
                "why": f"signal {direction} bị chặn MTF ({htf})",
            }
        return {"side": None, "action": "SKIP", "why": "signal bị loại"}
    soft_tag = f" [soft:{','.join(soft_notes)}]" if soft_notes else ""
    return {
        "side": side,
        "action": "PENDING",
        "why": (
            f"signal {'LONG' if side == 1 else 'SHORT'} → chờ fill nến sau{soft_tag}"
        ),
    }


def genome_from_dict(d: dict) -> Genome | None:
    try:
        g = Genome(**{k: d[k] for k in Genome.__dataclass_fields__ if k in d})
        if g.family not in FAMILIES or g.slow <= g.fast:
            return None
        if g.leverage not in config.LEVERAGE_CANDIDATES:
            return None
        return g
    except (TypeError, ValueError, KeyError):
        return None
