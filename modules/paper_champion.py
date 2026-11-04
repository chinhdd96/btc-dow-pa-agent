"""Paper trader driven by Champion genome on live closed candles."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
from modules import champion_store
from modules.features import add_features
from modules.genome import Genome, diagnose_entry, make_signal, mtf_side_allowed
from modules.quant_data import estimate_depth_fill

logger = logging.getLogger(__name__)


class PaperChampion:
    def __init__(
        self,
        *,
        on_close: Callable[[dict], None] | None = None,
        on_open: Callable[[dict], None] | None = None,
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        champion_store.init_db()
        self.state = champion_store.get_paper_state()
        if "equity" not in self.state:
            self.state["equity"] = config.INITIAL_CAPITAL
        # Drop Champion trained on another TF (e.g. 1h → 15m switch)
        if self.state.get("timeframe") != config.TIMEFRAME_MAIN:
            if self.state.get("champion"):
                logger.warning(
                    "Clearing Champion: timeframe %s → %s",
                    self.state.get("timeframe"),
                    config.TIMEFRAME_MAIN,
                )
            self.state["champion"] = None
            self.state["pending"] = None
            self.state["position"] = None
            self.state["timeframe"] = config.TIMEFRAME_MAIN
            self.state["research_due"] = True
            champion_store.set_paper_state(self.state)
        self.on_close = on_close
        self.on_open = on_open
        self.on_tick = on_tick
        self._closed_this_tick: list[dict] = []
        self.last_tick_report: dict[str, Any] | None = None

    def save(self) -> None:
        champion_store.set_paper_state(self.state)

    def reload(self) -> None:
        self.state = champion_store.get_paper_state()

    def set_champion(self, genome: Genome, *, baseline_expectancy: float | None = None) -> None:
        self.state["champion"] = asdict(genome)
        self.state["timeframe"] = config.TIMEFRAME_MAIN
        if baseline_expectancy is not None:
            self.state["champion_baseline_expectancy"] = baseline_expectancy
        self.state["research_due"] = False
        self.save()

    def process(self, df: pd.DataFrame) -> list[dict]:
        """Process one window ending at latest closed candle. Returns newly closed trades."""
        self._closed_this_tick = []
        self.last_tick_report = None
        x = add_features(df).reset_index(drop=True)
        if x.empty:
            return []
        last = x.iloc[-1]
        candle_id = str(last.open_time)
        if self.state.get("last_candle") == candle_id:
            return []

        if not self.state.get("champion"):
            logger.warning("No qualified Champion. Paper trader will not open positions.")
            report = {
                "candle": candle_id,
                "close": float(last.close),
                "action": "SKIP",
                "why": "chưa có Champion (đợi research qualify)",
                "family": None,
                "equity": float(self.state.get("equity") or config.INITIAL_CAPITAL),
            }
            self.last_tick_report = report
            self.state["last_candle"] = candle_id
            self.save()
            if self.on_tick:
                try:
                    self.on_tick(report)
                except Exception as exc:
                    logger.warning("on_tick callback failed: %s", exc)
            return []
        g = Genome(**self.state["champion"])

        opened_this = False
        closed_reason: str | None = None
        pending = self.state.get("pending")
        if (
            pending
            and pending["signal_time"] != candle_id
            and self.state.get("position") is None
        ):
            reference_entry = float(last.open)
            stop_dist = pending["atr"] * pending.get("atr_stop", g.atr_stop)
            reward_r = pending.get("reward_r", g.reward_r)
            leverage = int(pending["leverage"])
            margin = self.state["equity"] * config.MAX_MARGIN_FRACTION
            notional = margin * leverage
            qty = notional / reference_entry
            entry, qty, fill_source = estimate_depth_fill(
                config.SYMBOL,
                pending["side"],
                qty,
                reference_entry,
                config.SLIPPAGE_RATE,
            )
            logger.info("Paper entry fill model: %s", fill_source)
            sl = entry - pending["side"] * stop_dist
            tp = entry + pending["side"] * stop_dist * reward_r
            mmr = config.MAINTENANCE_MARGIN_RATE
            liq = (
                (
                    entry * (1 - 1 / leverage + mmr)
                    if pending["side"] == 1
                    else entry * (1 + 1 / leverage - mmr)
                )
                if leverage > 1
                else (
                    entry * 0.01 if pending["side"] == 1 else entry * 100
                )
            )
            entry_fee = notional * config.FEE_RATE
            position = {
                "side": pending["side"],
                "entry": entry,
                "entry_time": candle_id,
                "qty": qty,
                "margin": margin,
                "notional": notional,
                "leverage": leverage,
                "sl": sl,
                "tp": tp,
                "liq": liq,
                "entry_fee": entry_fee,
                "genome_id": pending["genome_id"],
                "family": g.family,
                "funding_pnl": 0.0,
            }
            self.state["position"] = position
            self.state["pending"] = None
            opened_this = True
            logger.info(
                "PAPER OPEN %s entry=%.2f lev=%sx margin=%.2f qty=%.6f SL=%.2f TP=%.2f",
                "LONG" if pending["side"] == 1 else "SHORT",
                entry,
                leverage,
                margin,
                qty,
                sl,
                tp,
            )
            if self.on_open:
                try:
                    self.on_open(position)
                except Exception as exc:
                    logger.warning("on_open callback failed: %s", exc)

        pos = self.state.get("position")
        if pos:
            fr = last.get("funding_event_rate", np.nan)
            if pd.notna(fr):
                funding_cashflow = (
                    -pos["side"] * float(fr) * pos["entry"] * pos["qty"]
                )
                self.state["equity"] += funding_cashflow
                pos["funding_pnl"] = pos.get("funding_pnl", 0.0) + funding_cashflow
            liq_hit = (
                last.low <= pos.get("liq", -np.inf)
                if pos["side"] == 1
                else last.high >= pos.get("liq", np.inf)
            )
            stop_hit = (
                last.low <= pos["sl"] if pos["side"] == 1 else last.high >= pos["sl"]
            )
            tp_hit = (
                last.high >= pos["tp"] if pos["side"] == 1 else last.low <= pos["tp"]
            )
            reason = (
                "LIQUIDATION_ESTIMATE"
                if liq_hit
                else ("SL" if stop_hit else ("TP" if tp_hit else None))
            )
            if reason:
                px = (
                    pos.get("liq")
                    if reason == "LIQUIDATION_ESTIMATE"
                    else (pos["sl"] if reason == "SL" else pos["tp"])
                )
                fill, _, fill_source = estimate_depth_fill(
                    config.SYMBOL,
                    -pos["side"],
                    pos["qty"],
                    px,
                    config.SLIPPAGE_RATE,
                )
                logger.info("Paper exit fill model: %s", fill_source)
                gross = (fill - pos["entry"]) * pos["qty"] * pos["side"]
                exit_fee = abs(fill * pos["qty"]) * config.FEE_RATE
                trading_pnl = gross - pos["entry_fee"] - exit_fee
                pnl = trading_pnl + pos.get("funding_pnl", 0.0)
                self.state["equity"] += trading_pnl
                trade = {
                    **pos,
                    "exit": fill,
                    "exit_reason": reason,
                    "pnl": pnl,
                    "exit_time": candle_id,
                    "exit_fee": exit_fee,
                    "equity_after": self.state["equity"],
                }
                champion_store.insert_paper_trade(trade)
                self.state["trade_count"] = int(self.state.get("trade_count") or 0) + 1
                self.state["position"] = None
                self._closed_this_tick.append(trade)
                closed_reason = reason
                logger.info(
                    "PAPER CLOSE %s pnl=%+.4f equity=%.2f",
                    reason,
                    pnl,
                    self.state["equity"],
                )
                if self.on_close:
                    try:
                        self.on_close(trade)
                    except Exception as exc:
                        logger.warning("on_close callback failed: %s", exc)

        decision = {"side": None, "action": "HOLD", "why": ""}
        if self.state.get("position") is not None:
            pos = self.state["position"]
            side_txt = "LONG" if pos["side"] == 1 else "SHORT"
            decision = {
                "side": pos["side"],
                "action": "HOLD",
                "why": f"đang {side_txt} — chờ SL/TP (sl={float(pos['sl']):.0f} tp={float(pos['tp']):.0f})",
            }
        elif self.state.get("pending") is not None and not opened_this:
            pend = self.state["pending"]
            side_txt = "LONG" if pend["side"] == 1 else "SHORT"
            decision = {
                "side": pend["side"],
                "action": "PENDING",
                "why": f"đã có pending {side_txt} — chờ nến sau để fill",
            }
        elif len(x) <= 202:
            decision = {
                "side": None,
                "action": "SKIP",
                "why": f"chưa đủ bar warm-up ({len(x)}/203)",
            }
        else:
            decision = diagnose_entry(last, x.iloc[-2], g)
            if decision["action"] == "PENDING" and decision.get("side") is not None:
                side = int(decision["side"])
                if mtf_side_allowed(last, side, g.mtf_filter):
                    self.state["pending"] = {
                        "side": side,
                        "signal_time": candle_id,
                        "atr": float(last.atr),
                        "genome_id": g.genome_id,
                        "leverage": int(g.leverage),
                        "atr_stop": g.atr_stop,
                        "reward_r": g.reward_r,
                    }

        if opened_this:
            pos = self.state.get("position") or {}
            decision = {
                "side": pos.get("side"),
                "action": "OPEN",
                "why": f"fill {'LONG' if pos.get('side') == 1 else 'SHORT'} @ {float(pos.get('entry') or 0):.2f}",
            }
        elif closed_reason:
            decision = {
                "side": None,
                "action": f"CLOSE_{closed_reason}",
                "why": f"đóng lệnh vì {closed_reason}",
            }

        report = {
            "candle": candle_id,
            "close": float(last.close),
            "atr_pct": float(last.atr_pct) if pd.notna(last.atr_pct) else None,
            "rsi": float(last.rsi) if pd.notna(last.rsi) else None,
            "family": g.family,
            "genome_id": g.genome_id,
            "equity": float(self.state.get("equity") or 0),
            **decision,
        }
        self.last_tick_report = report
        self.state["last_candle"] = candle_id
        self.save()
        if self.on_tick:
            try:
                self.on_tick(report)
            except Exception as exc:
                logger.warning("on_tick callback failed: %s", exc)
        return list(self._closed_this_tick)

    def summary(self) -> dict[str, Any]:
        champ = self.state.get("champion") or {}
        return {
            "equity": float(self.state.get("equity") or config.INITIAL_CAPITAL),
            "initial_capital": config.INITIAL_CAPITAL,
            "pnl_total": float(self.state.get("equity") or 0) - config.INITIAL_CAPITAL,
            "trade_count": int(self.state.get("trade_count") or 0),
            "has_position": self.state.get("position") is not None,
            "has_pending": self.state.get("pending") is not None,
            "champion_family": champ.get("family"),
            "champion_id": Genome(**champ).genome_id if champ else None,
            "research_due": bool(self.state.get("research_due")),
        }
