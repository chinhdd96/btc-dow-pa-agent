"""Deterministic backtest and train/val/test evaluation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

import config
from modules.genome import Genome, make_signal, mtf_side_allowed


def backtest(df: pd.DataFrame, g: Genome, score_start: int = 201) -> dict:
    if len(df) < 250:
        return {
            "trades": [],
            "metrics": {
                "net_return_pct": -999,
                "profit_factor": 0,
                "max_drawdown_pct": 100,
                "trade_count": 0,
                "expectancy": -999,
                "score": -999,
            },
        }
    x = df.reset_index(drop=True)
    equity = config.INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    pos = None
    trades: list[dict] = []
    fee, slip = config.FEE_RATE, config.SLIPPAGE_RATE
    for i in range(max(201, score_start), len(x)):
        row, prev = x.iloc[i], x.iloc[i - 1]
        if pos:
            fr = row.get("funding_event_rate", np.nan)
            if pd.notna(fr):
                funding_cashflow = -pos["side"] * float(fr) * pos["entry"] * pos["qty"]
                equity += funding_cashflow
                pos["funding_pnl"] = pos.get("funding_pnl", 0.0) + funding_cashflow
            liq_hit = (
                row.low <= pos["liq"] if pos["side"] == 1 else row.high >= pos["liq"]
            )
            stop_hit = (
                row.low <= pos["sl"] if pos["side"] == 1 else row.high >= pos["sl"]
            )
            tp_hit = row.high >= pos["tp"] if pos["side"] == 1 else row.low <= pos["tp"]
            exit_reason, exit_px = None, None
            if liq_hit:
                exit_reason, exit_px = "LIQUIDATION_ESTIMATE", pos["liq"]
            elif stop_hit:
                exit_reason, exit_px = "SL", pos["sl"]
            elif tp_hit:
                exit_reason, exit_px = "TP", pos["tp"]
            elif i == len(x) - 1:
                exit_reason, exit_px = "END", row.close
            if exit_px is not None:
                exit_fill = exit_px * (1 - slip * pos["side"])
                gross = (exit_fill - pos["entry"]) * pos["qty"] * pos["side"]
                exit_fee = abs(exit_fill * pos["qty"]) * fee
                trading_pnl = gross - pos["entry_fee"] - exit_fee
                pnl = trading_pnl + pos.get("funding_pnl", 0.0)
                equity += trading_pnl
                trades.append(
                    {
                        "entry_time": str(pos["time"]),
                        "exit_time": str(row.open_time),
                        "side": "LONG" if pos["side"] == 1 else "SHORT",
                        "entry": pos["entry"],
                        "exit": exit_fill,
                        "qty": pos["qty"],
                        "leverage": pos["leverage"],
                        "margin": pos["margin"],
                        "pnl": pnl,
                        "funding_pnl": pos.get("funding_pnl", 0.0),
                        "fees": pos["entry_fee"] + exit_fee,
                        "reason": exit_reason,
                        "genome_id": g.genome_id,
                    }
                )
                pos = None
        marked_equity = equity
        if pos:
            unrealized = (float(row.close) - pos["entry"]) * pos["qty"] * pos["side"]
            estimated_exit_fee = abs(float(row.close) * pos["qty"]) * fee
            marked_equity += unrealized - estimated_exit_fee
        peak = max(peak, marked_equity)
        max_dd = max(max_dd, (peak - marked_equity) / peak if peak else 0)
        if pos or i == len(x) - 1:
            continue
        side = make_signal(row, prev, g)
        if not mtf_side_allowed(row, side, g.mtf_filter):
            continue
        if i + 1 >= len(x):
            continue
        nxt = x.iloc[i + 1]
        entry = float(nxt.open) * (1 + slip * side)
        stop_dist = float(row.atr) * g.atr_stop
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            continue
        leverage = int(g.leverage)
        margin = equity * config.MAX_MARGIN_FRACTION
        notional = margin * leverage
        qty = notional / entry
        sl = entry - side * stop_dist
        tp = entry + side * stop_dist * g.reward_r
        mmr = config.MAINTENANCE_MARGIN_RATE
        liq = (
            (
                entry * (1 - 1 / leverage + mmr)
                if side == 1
                else entry * (1 + 1 / leverage - mmr)
            )
            if leverage > 1
            else (entry * 0.01 if side == 1 else entry * 100)
        )
        entry_fee = notional * fee
        expected_move = stop_dist * g.reward_r * qty
        expected_cost = (
            entry_fee + abs(tp * qty) * fee + notional * slip + abs(tp * qty) * slip
        )
        if expected_move <= expected_cost:
            continue
        pos = {
            "side": side,
            "entry": entry,
            "time": nxt.open_time,
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "entry_fee": entry_fee,
            "leverage": leverage,
            "margin": margin,
            "funding_pnl": 0.0,
            "liq": liq,
        }
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    ret = (equity / config.INITIAL_CAPITAL - 1) * 100
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    expectancy = float(np.mean(pnls)) if pnls else -999.0
    score = ret - max_dd * 100 * 1.5 + min(pf, 4) * 2 + min(len(trades), 100) * 0.03
    if len(trades) < 8:
        score -= 25
    if not np.isfinite(pf):
        pf = 4.0
    return {
        "trades": trades,
        "metrics": {
            "net_return_pct": ret,
            "profit_factor": pf,
            "max_drawdown_pct": max_dd * 100,
            "trade_count": len(trades),
            "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0,
            "expectancy": expectancy,
            "score": score,
            "ending_equity": equity,
        },
    }


def evaluate_genome(df: pd.DataFrame, g: Genome) -> dict:
    n = len(df)
    a = int(n * config.TRAIN_FRACTION)
    b = int(n * (config.TRAIN_FRACTION + config.VALIDATION_FRACTION))
    train = backtest(df.iloc[:a], g)
    val_offset = a - max(0, a - 250)
    test_offset = b - max(0, b - 250)
    val = backtest(df.iloc[max(0, a - 250) : b], g, score_start=val_offset)
    test = backtest(df.iloc[max(0, b - 250) :], g, score_start=test_offset)
    vm, tm = val["metrics"], test["metrics"]
    qualified = (
        vm["trade_count"] >= config.MIN_VALIDATION_TRADES
        and tm["trade_count"] >= config.MIN_TEST_TRADES
        and vm["expectancy"] > 0
        and tm["expectancy"] > 0
        and vm["profit_factor"] > 1.05
        and tm["profit_factor"] > 1.05
    )
    robust = (
        min(vm["score"], tm["score"])
        if qualified
        else min(vm["score"], tm["score"]) - 50
    )
    return {
        "genome": asdict(g),
        "genome_id": g.genome_id,
        "train": train["metrics"],
        "validation": vm,
        "test": tm,
        "qualified": qualified,
        "robust_score": robust,
    }
