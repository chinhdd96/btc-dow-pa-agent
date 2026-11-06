"""Strategy Spec backtest + walk-forward evaluation (shares trade_sim with paper)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import config
from modules import trade_sim

SignalFn = Callable[[pd.Series, pd.Series], Optional[int]]

_EMPTY_METRICS = {
    "net_return_pct": -999,
    "profit_factor": 0,
    "max_drawdown_pct": 100,
    "trade_count": 0,
    "win_rate_pct": 0,
    "expectancy": -999,
    "score": -999,
}


def _score(ret: float, max_dd_frac: float, pf: float, n: int) -> float:
    score = ret - max_dd_frac * 100 * 1.5 + min(pf, 4) * 2 + min(n, 100) * 0.03
    if n < 8:
        score -= 25
    return score


def metrics_from_trades(trades: list[dict]) -> dict[str, Any]:
    """Closed-trade metrics for a subset of trades (fold / half / shadow)."""
    if not trades:
        return dict(_EMPTY_METRICS)
    pnls = [float(t["pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    gp, gl = sum(wins), abs(sum(p for p in pnls if p <= 0))
    pf = gp / gl if gl > 0 else (4.0 if gp > 0 else 0.0)
    eq = config.INITIAL_CAPITAL + np.cumsum(pnls)
    peaks = np.maximum.accumulate(np.concatenate([[config.INITIAL_CAPITAL], eq]))[1:]
    max_dd = float(np.max((peaks - eq) / peaks)) if len(eq) else 0.0
    ret = sum(pnls) / config.INITIAL_CAPITAL * 100
    return {
        "net_return_pct": ret,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd * 100,
        "trade_count": len(pnls),
        "win_rate_pct": len(wins) / len(pnls) * 100,
        "expectancy": float(np.mean(pnls)),
        "score": _score(ret, max_dd, pf, len(pnls)),
    }


def backtest_signal(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    exit_cfg: dict[str, Any],
    identity: str = "",
    score_start: int = 201,
) -> dict:
    """Bar-by-bar simulation: signal on bar i fills at bar i+1 open (same as paper)."""
    if len(df) < 250:
        return {"trades": [], "metrics": dict(_EMPTY_METRICS)}
    x = df.reset_index(drop=True)
    high = x["high"].to_numpy(float)
    low = x["low"].to_numpy(float)
    close = x["close"].to_numpy(float)
    opn = x["open"].to_numpy(float)
    atr = x["atr"].to_numpy(float) if "atr" in x else np.full(len(x), np.nan)
    fund = (
        x["funding_event_rate"].to_numpy(float)
        if "funding_event_rate" in x
        else np.full(len(x), np.nan)
    )
    times = x["open_time"].astype(str).tolist()
    use_opposite = bool(exit_cfg.get("exit_on_opposite"))

    equity = config.INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    pos: Optional[dict] = None
    trades: list[dict] = []
    start = max(201, score_start)
    prev = x.iloc[start - 1]
    last = len(x) - 1
    for i in range(start, len(x)):
        row = x.iloc[i]
        sig: Optional[int] = None
        sig_done = False
        if pos:
            equity += trade_sim.apply_funding(pos, fund[i])
            hit = trade_sim.check_hit(pos, high[i], low[i])
            if hit is None and use_opposite:
                sig, sig_done = signal_fn(row, prev), True
            if hit is None:
                hit = trade_sim.bar_close_exit(
                    pos, close=close[i], exit_cfg=exit_cfg, signal_side=sig
                )
            if hit is None and i == last:
                hit = ("END", float(close[i]))
            if hit:
                t = trade_sim.close_position(pos, px=hit[1], reason=hit[0], time=times[i])
                t["spec_id"] = identity
                equity += t["trading_pnl"]
                trades.append(t)
                pos = None
            else:
                trade_sim.update_exit_levels(
                    pos, high=high[i], low=low[i], close=close[i], atr=atr[i], exit_cfg=exit_cfg
                )
        marked = equity
        if pos:
            unreal = (close[i] - pos["entry"]) * pos["qty"] * pos["side"]
            marked += unreal - abs(close[i] * pos["qty"]) * config.FEE_RATE
        peak = max(peak, marked)
        max_dd = max(max_dd, (peak - marked) / peak if peak else 0)
        if pos is None and i < last:
            side = sig if sig_done else signal_fn(row, prev)
            if side is not None:
                pos = trade_sim.open_position(
                    side=side,
                    ref_open=float(opn[i + 1]),
                    atr=float(atr[i]),
                    exit_cfg=exit_cfg,
                    equity=equity,
                    time=times[i + 1],
                )
        prev = row

    m = metrics_from_trades(trades)
    ret = (equity / config.INITIAL_CAPITAL - 1) * 100
    m.update(
        {
            "net_return_pct": ret,
            "max_drawdown_pct": max_dd * 100,
            "score": _score(ret, max_dd, m["profit_factor"], len(trades)) if trades else -999,
            "ending_equity": equity,
        }
    )
    return {"trades": trades, "metrics": m}


def complexity_penalty(spec: dict[str, Any]) -> float:
    per_side = max(len(spec.get("entry_long") or []), len(spec.get("entry_short") or []))
    n = per_side + len(spec.get("filters") or [])
    return 2.5 * max(0, n - config.SPEC_MAX_CONDS_FREE)


def evaluate_spec(
    df: pd.DataFrame,
    spec: dict[str, Any],
    *,
    base: pd.DataFrame | None = None,
) -> dict:
    """One out-of-sample backtest after the train window, split into walk-forward folds."""
    from modules.features_catalog import ensure_features_for_spec
    from modules.strategy_spec import compile_signal, exit_params, normalize_spec, spec_id

    s = normalize_spec(spec)
    sid = spec_id(s)
    framed = ensure_features_for_spec(df, s, base=base).reset_index(drop=True)
    fn = compile_signal(s)
    ex = exit_params(s)
    n = len(framed)
    a = int(n * config.TRAIN_FRACTION)
    start = max(0, a - 250)
    bt = backtest_signal(
        framed.iloc[start:], fn, exit_cfg=ex, identity=sid, score_start=a - start
    )
    trades = bt["trades"]
    oos = bt["metrics"]

    n_folds = max(2, config.WF_FOLDS)
    edges = [
        pd.Timestamp(framed["open_time"].iloc[a + k * (n - a) // n_folds])
        for k in range(n_folds)
    ]
    buckets: list[list[dict]] = [[] for _ in range(n_folds)]
    for t in trades:
        ts = pd.Timestamp(t["entry_time"])
        k = max(i for i, e in enumerate(edges) if ts >= e) if ts >= edges[0] else 0
        buckets[k].append(t)
    folds = [metrics_from_trades(b) for b in buckets]
    half = n_folds // 2
    vm = metrics_from_trades([t for b in buckets[:half] for t in b])
    tm = metrics_from_trades([t for b in buckets[half:] for t in b])
    pos_folds = sum(1 for f in folds if f["trade_count"] > 0 and f["expectancy"] > 0)

    qualified = (
        vm["trade_count"] >= config.MIN_VALIDATION_TRADES
        and tm["trade_count"] >= config.MIN_TEST_TRADES
        and oos["expectancy"] > 0
        and oos["profit_factor"] > 1.05
        and pos_folds >= config.WF_MIN_POSITIVE_FOLDS
    )
    soft_qualified = (
        not qualified
        and vm["trade_count"] >= config.SOFT_MIN_VALIDATION_TRADES
        and tm["trade_count"] >= config.SOFT_MIN_TEST_TRADES
        and oos["profit_factor"] >= config.SOFT_MIN_PROFIT_FACTOR
        and (oos["expectancy"] > 0 or pos_folds >= half)
    )
    vs, ts_ = vm["score"], tm["score"]
    # Half conservative (worst half), half weighted toward the recent half
    robust = 0.5 * min(vs, ts_) + 0.5 * (0.35 * vs + 0.65 * ts_)
    robust += 3.0 * pos_folds / n_folds
    robust -= complexity_penalty(s)
    if int(ex["leverage"]) >= 5 and oos["max_drawdown_pct"] > 20:
        robust -= 5
    if not qualified:
        robust -= 25 if soft_qualified else 50
    return {
        "spec": s,
        "spec_id": sid,
        "name": s.get("name"),
        "train": {},
        "validation": vm,
        "test": tm,
        "oos": oos,
        "folds": [
            {
                "trades": f["trade_count"],
                "expectancy": round(f["expectancy"], 4) if f["trade_count"] else None,
                "pf": round(f["profit_factor"], 2),
            }
            for f in folds
        ],
        "positive_folds": pos_folds,
        "qualified": qualified,
        "soft_qualified": soft_qualified,
        "robust_score": robust,
    }
