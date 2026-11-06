"""Shared position mechanics for backtest, paper, live check and shadow Specs.

Every place that opens, manages or closes a simulated position goes through
these functions so backtest and paper can never drift apart.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

import config

# fill_fn(side_of_order, qty, reference_px) -> (fill_px, qty)
FillFn = Callable[[int, float, float], tuple[float, float]]


def slippage_fill(side: int, qty: float, px: float) -> tuple[float, float]:
    return px * (1 + config.SLIPPAGE_RATE * side), qty


def liquidation_price(entry: float, side: int, leverage: int) -> float:
    mmr = config.MAINTENANCE_MARGIN_RATE
    if leverage <= 1:
        return entry * 0.01 if side == 1 else entry * 100
    if side == 1:
        return entry * (1 - 1 / leverage + mmr)
    return entry * (1 + 1 / leverage - mmr)


def entry_viable(stop_dist: float, reward_r: float, qty: float, entry: float) -> bool:
    """Skip setups whose target cannot cover round-trip fees + slippage."""
    fee, slip = config.FEE_RATE, config.SLIPPAGE_RATE
    if not np.isfinite(stop_dist) or stop_dist <= 0:
        return False
    tp_px = entry + stop_dist * reward_r
    notional = entry * qty
    expected_move = stop_dist * reward_r * qty
    expected_cost = (
        notional * fee + abs(tp_px * qty) * fee + notional * slip + abs(tp_px * qty) * slip
    )
    return expected_move > expected_cost


def open_position(
    *,
    side: int,
    ref_open: float,
    atr: float,
    exit_cfg: dict[str, Any],
    equity: float,
    time: str,
    fill_fn: FillFn = slippage_fill,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    atr_stop = float(exit_cfg.get("atr_stop") or 2.0)
    reward_r = float(exit_cfg.get("reward_r") or 2.0)
    leverage = int(exit_cfg.get("leverage") or 1)
    stop_dist = float(atr) * atr_stop
    if not np.isfinite(stop_dist) or stop_dist <= 0 or ref_open <= 0:
        return None
    margin = equity * config.MAX_MARGIN_FRACTION
    notional = margin * leverage
    qty = notional / ref_open
    entry, qty = fill_fn(side, qty, ref_open)
    if not entry_viable(stop_dist, reward_r, qty, entry):
        return None
    sl = entry - side * stop_dist
    tp = entry + side * stop_dist * reward_r
    pos = {
        "side": side,
        "entry": entry,
        "entry_time": time,
        "qty": qty,
        "margin": margin,
        "notional": notional,
        "leverage": leverage,
        "sl": sl,
        "initial_sl": sl,
        "tp": tp,
        "liq": liquidation_price(entry, side, leverage),
        "entry_fee": notional * config.FEE_RATE,
        "risk_usd": stop_dist * qty,
        "funding_pnl": 0.0,
        "bars_held": 0,
    }
    if extra:
        pos.update(extra)
    return pos


def check_hit(pos: dict[str, Any], high: float, low: float) -> Optional[tuple[str, float]]:
    """Price-level exits; LIQ > SL > TP (conservative when one bar hits both)."""
    side = pos["side"]
    liq = pos.get("liq")
    if liq is not None:
        if (side == 1 and low <= liq) or (side == -1 and high >= liq):
            return "LIQUIDATION_ESTIMATE", float(liq)
    if (side == 1 and low <= pos["sl"]) or (side == -1 and high >= pos["sl"]):
        reason = "BE" if abs(pos["sl"] - pos["entry"]) < 1e-9 else (
            "TRAIL" if pos.get("sl_moved") else "SL"
        )
        return reason, float(pos["sl"])
    if (side == 1 and high >= pos["tp"]) or (side == -1 and low <= pos["tp"]):
        return "TP", float(pos["tp"])
    return None


def apply_funding(pos: dict[str, Any], funding_rate: float) -> float:
    if funding_rate is None or not np.isfinite(funding_rate):
        return 0.0
    cash = -pos["side"] * float(funding_rate) * pos["entry"] * pos["qty"]
    pos["funding_pnl"] = pos.get("funding_pnl", 0.0) + cash
    return cash


def close_position(
    pos: dict[str, Any],
    *,
    px: float,
    reason: str,
    time: str,
    fill_fn: FillFn = slippage_fill,
) -> dict[str, Any]:
    fill, _ = fill_fn(-pos["side"], pos["qty"], px)
    gross = (fill - pos["entry"]) * pos["qty"] * pos["side"]
    exit_fee = abs(fill * pos["qty"]) * config.FEE_RATE
    trading_pnl = gross - pos["entry_fee"] - exit_fee
    pnl = trading_pnl + pos.get("funding_pnl", 0.0)
    risk = float(pos.get("risk_usd") or 0)
    return {
        **pos,
        "exit": fill,
        "exit_reason": reason,
        "exit_time": time,
        "exit_fee": exit_fee,
        "trading_pnl": trading_pnl,
        "pnl": pnl,
        "r_multiple": (pnl / risk) if risk > 0 else None,
    }


def bar_close_exit(
    pos: dict[str, Any],
    *,
    close: float,
    exit_cfg: dict[str, Any],
    signal_side: Optional[int],
) -> Optional[tuple[str, float]]:
    """Exits decided only on a closed bar: opposite signal, then time stop."""
    if exit_cfg.get("exit_on_opposite") and signal_side == -pos["side"]:
        return "OPPOSITE", float(close)
    max_bars = int(exit_cfg.get("max_bars") or 0)
    if max_bars > 0 and int(pos.get("bars_held") or 0) >= max_bars:
        return "TIME", float(close)
    return None


def update_exit_levels(
    pos: dict[str, Any],
    *,
    high: float,
    low: float,
    close: float,
    atr: float,
    exit_cfg: dict[str, Any],
) -> None:
    """Move SL after a closed bar (breakeven, trailing). SL only ever tightens."""
    side = pos["side"]
    pos["bars_held"] = int(pos.get("bars_held") or 0) + 1
    risk_px = abs(pos["entry"] - pos.get("initial_sl", pos["sl"]))
    new_sl = pos["sl"]
    be_r = float(exit_cfg.get("breakeven_r") or 0)
    if be_r > 0 and risk_px > 0:
        best = high - pos["entry"] if side == 1 else pos["entry"] - low
        if best >= be_r * risk_px:
            new_sl = max(new_sl, pos["entry"]) if side == 1 else min(new_sl, pos["entry"])
    trail = float(exit_cfg.get("trail_atr") or 0)
    if trail > 0 and np.isfinite(atr) and atr > 0:
        cand = close - side * trail * atr
        new_sl = max(new_sl, cand) if side == 1 else min(new_sl, cand)
    if new_sl != pos["sl"]:
        pos["sl"] = new_sl
        pos["sl_moved"] = True
