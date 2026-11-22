"""Shared wallet and execution rules for the portfolio simulator."""

from __future__ import annotations

import math

import config
from modules import trade_sim


def new_state(symbols: tuple[str, ...], capital: float) -> dict:
    return {
        "symbols": list(symbols),
        "interval": config.TIMEFRAME_MAIN,
        "capital": capital,
        "cash": capital,
        "entries": 0,
        "champion": None,
        "version": 0,
        "markets": {symbol: {} for symbol in symbols},
    }


class Portfolio:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.opened: list[dict] = []
        self.closed: list[dict] = []

    @property
    def available(self) -> float:
        reserved = sum(
            m["position"]["margin"] for m in self.state["markets"].values() if m.get("position")
        )
        return self.state["cash"] - reserved

    def equity(self) -> float:
        equity = self.state["cash"]
        for market in self.state["markets"].values():
            if pos := market.get("position"):
                mark = market.get("mark", pos["entry"])
                equity += (mark - pos["entry"]) * pos["qty"] * pos["side"]
                equity -= abs(mark * pos["qty"]) * config.FEE_RATE
        return equity

    def signal(
        self, symbol: str, *, side: int | None, atr: float, spec: dict,
        signal_time: int, fill_time: int, context: dict | None = None,
    ) -> None:
        market = self.state["markets"][symbol]
        if side is None or not math.isfinite(atr) or atr <= 0:
            return
        if market.get("position") or market.get("pending"):
            return
        market["pending"] = {
            "side": side,
            "atr": atr,
            "signal_time": signal_time,
            "fill_time": fill_time,
            "exit_cfg": spec["exit"],
            "spec": spec,
            "version": self.state["version"],
            "entry_context": context,
        }

    def fill(self, symbol: str, time: int, price: float) -> bool:
        market = self.state["markets"][symbol]
        pending = market.get("pending")
        if not pending or time < pending["fill_time"]:
            return False
        market.pop("pending")
        if time != pending["fill_time"] or market.get("position"):
            return False
        leverage = pending["exit_cfg"]["leverage"]
        equity = max(0.0, self.equity())
        margin = equity * config.MAX_MARGIN_FRACTION
        if margin <= 0 or margin * (1 + leverage * config.FEE_RATE) > self.available + 1e-9:
            return False
        sequence = self.state["entries"] + 1
        pos = trade_sim.open_position(
            side=pending["side"], ref_open=price, atr=pending["atr"],
            exit_cfg=pending["exit_cfg"], equity=equity, time=time,
            extra={
                "entry_seq": sequence, "symbol": symbol, "spec": pending["spec"],
                "version": pending["version"], "exit_cfg": pending["exit_cfg"],
                "signal_time": pending["signal_time"], "entry_context": pending["entry_context"],
            },
        )
        if not pos:
            return False
        self.state["entries"] = sequence
        self.state["cash"] -= pos["entry_fee"]
        market.update(position=pos, mark=price)
        self.opened.append(dict(pos))
        return True

    def funding(self, symbol: str, time: int, rate: float) -> None:
        market = self.state["markets"][symbol]
        pos = market.get("position")
        if pos and pos["entry_time"] <= time and time > pos.get("last_funding", -1) and math.isfinite(rate):
            self.state["cash"] += trade_sim.apply_funding(pos, rate)
            pos["last_funding"] = time

    def touch(self, symbol: str, time: int, high: float, low: float, mark: float) -> None:
        market = self.state["markets"][symbol]
        market["mark"] = mark
        pos = market.get("position")
        if pos and pos["entry_time"] <= time:
            if hit := trade_sim.check_hit(pos, high, low):
                self.close(symbol, time, hit[0], hit[1])

    def close_bar(
        self, symbol: str, time: int, high: float, low: float, close: float,
        atr: float, opposite_signal: int | None,
    ) -> None:
        market = self.state["markets"][symbol]
        market["mark"] = close
        pos = market.get("position")
        if pos and pos["entry_time"] < time:
            ex = pos["exit_cfg"]
            hit = trade_sim.bar_close_exit(pos, close=close, exit_cfg=ex, signal_side=opposite_signal)
            if hit:
                self.close(symbol, time, hit[0], hit[1])
            else:
                trade_sim.update_exit_levels(pos, high=high, low=low, close=close, atr=atr, exit_cfg=ex)

    def close(self, symbol: str, time: int, reason: str, price: float) -> None:
        pos = self.state["markets"][symbol].pop("position")
        trade = trade_sim.close_position(pos, px=price, reason=reason, time=time)
        self.state["cash"] += trade["trading_pnl"] + pos["entry_fee"]
        self.closed.append(trade)
