"""Paper trader driven by Champion Strategy Spec (catalog indicators) on live candles."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
from modules import champion_store
from modules.features_catalog import ensure_features_for_spec
from modules.quant_data import estimate_depth_fill
from modules.strategy_spec import (
    base_name,
    diagnose_spec_entry,
    exit_params,
    genome_dict_to_spec,
    normalize_spec,
    spec_id,
)

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
        # Migrate legacy genome → champion_spec once
        if not self.state.get("champion_spec") and self.state.get("champion"):
            try:
                self.state["champion_spec"] = genome_dict_to_spec(self.state["champion"])
                self.state["champion_spec_id"] = spec_id(self.state["champion_spec"])
                self.state["trades_on_spec"] = 0
                logger.info(
                    "Migrated genome → Spec id=%s",
                    self.state["champion_spec_id"],
                )
                champion_store.set_paper_state(self.state)
            except Exception as exc:
                logger.warning("genome→spec migrate failed: %s", exc)
        if self.state.get("timeframe") != config.TIMEFRAME_MAIN:
            if self.state.get("champion_spec") or self.state.get("champion"):
                logger.warning(
                    "Clearing Champion: timeframe %s → %s",
                    self.state.get("timeframe"),
                    config.TIMEFRAME_MAIN,
                )
            self.state["champion"] = None
            self.state["champion_spec"] = None
            self.state["champion_spec_id"] = None
            self.state["trades_on_spec"] = 0
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

    def set_champion_spec(
        self,
        spec: dict,
        *,
        baseline_expectancy: float | None = None,
    ) -> None:
        s = normalize_spec(spec)
        s["name"] = base_name(s.get("name"))
        new_id = spec_id(s)
        old = self.state.get("champion_spec")
        old_id = spec_id(old) if old else None
        self.state["champion_spec"] = s
        self.state["champion_spec_id"] = new_id
        if new_id != old_id:
            self.state["trades_on_spec"] = 0
        self.state["research_mode"] = None
        self.state["timeframe"] = config.TIMEFRAME_MAIN
        # Keep a slim legacy mirror for old readers
        self.state["champion"] = {
            "family": s.get("legacy_family") or s.get("name") or "spec",
            "fast": 12,
            "slow": 50,
            "rsi_low": 30,
            "rsi_high": 70,
            "breakout_n": 20,
            "atr_stop": float(s["exit"]["atr_stop"]),
            "reward_r": float(s["exit"]["reward_r"]),
            "volume_filter": 0.0,
            "trend_filter": False,
            "max_atr_pct": 0.06,
            "leverage": int(s["exit"]["leverage"]),
            "funding_abs_max": 0.001,
            "oi_change_min": 0.0,
            "volatility_mode": "any",
            "mtf_filter": bool(s.get("mtf_filter")),
        }
        if baseline_expectancy is not None:
            self.state["champion_baseline_expectancy"] = baseline_expectancy
        self.state["research_due"] = False
        self.save()

    def process(self, df: pd.DataFrame) -> list[dict]:
        self._closed_this_tick = []
        self.last_tick_report = None
        spec = self.state.get("champion_spec")
        if not spec and self.state.get("champion"):
            try:
                spec = genome_dict_to_spec(self.state["champion"])
                self.state["champion_spec"] = spec
                self.state["champion_spec_id"] = spec_id(spec)
            except Exception:
                spec = None

        if not spec:
            # Still need candle id
            from modules.features import add_features

            x = add_features(df).reset_index(drop=True)
            if x.empty:
                return []
            last = x.iloc[-1]
            candle_id = str(last.open_time)
            if self.state.get("last_candle") == candle_id:
                return []
            report = {
                "candle": candle_id,
                "close": float(last.close),
                "action": "SKIP",
                "why": "chưa có Champion Spec (research đang tìm — PnL-first)",
                "family": None,
                "spec_id": None,
                "equity": float(self.state.get("equity") or config.INITIAL_CAPITAL),
            }
            self.last_tick_report = report
            self.state["last_candle"] = candle_id
            self.state["research_due"] = True
            self.save()
            if self.on_tick:
                try:
                    self.on_tick(report)
                except Exception as exc:
                    logger.warning("on_tick callback failed: %s", exc)
            return []

        s = normalize_spec(spec)
        sid = spec_id(s)
        x = ensure_features_for_spec(df, s).reset_index(drop=True)
        if x.empty:
            return []
        last = x.iloc[-1]
        candle_id = str(last.open_time)
        if self.state.get("last_candle") == candle_id:
            return []

        ex = exit_params(s)
        opened_this = False
        closed_reason: str | None = None
        pending = self.state.get("pending")
        if (
            pending
            and pending["signal_time"] != candle_id
            and self.state.get("position") is None
        ):
            reference_entry = float(last.open)
            stop_dist = pending["atr"] * pending.get("atr_stop", ex["atr_stop"])
            reward_r = pending.get("reward_r", ex["reward_r"])
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
                else (entry * 0.01 if pending["side"] == 1 else entry * 100)
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
                "genome_id": pending.get("genome_id") or sid,
                "spec_id": sid,
                "family": s.get("name") or s.get("legacy_family") or "spec",
                "funding_pnl": 0.0,
            }
            self.state["position"] = position
            self.state["pending"] = None
            opened_this = True
            logger.info(
                "PAPER OPEN %s entry=%.2f lev=%sx SL=%.2f TP=%.2f spec=%s",
                "LONG" if pending["side"] == 1 else "SHORT",
                entry,
                leverage,
                sl,
                tp,
                sid,
            )
            if self.on_open:
                try:
                    self.on_open(position)
                except Exception as exc:
                    logger.warning("on_open callback failed: %s", exc)

        pos = self.state.get("position")
        check_bars = [last]
        prev_candle = self.state.get("last_candle")
        if pos and not opened_this and prev_candle:
            times = x["open_time"].astype(str)
            hit = times.index[times == str(prev_candle)]
            if len(hit):
                check_bars = [x.iloc[j] for j in range(int(hit[0]) + 1, len(x))] or [last]
        for bar in check_bars if pos else []:
            fr = bar.get("funding_event_rate", np.nan)
            if pd.notna(fr):
                funding_cashflow = (
                    -pos["side"] * float(fr) * pos["entry"] * pos["qty"]
                )
                self.state["equity"] += funding_cashflow
                pos["funding_pnl"] = pos.get("funding_pnl", 0.0) + funding_cashflow
            liq_hit = (
                bar.low <= pos.get("liq", -np.inf)
                if pos["side"] == 1
                else bar.high >= pos.get("liq", np.inf)
            )
            stop_hit = (
                bar.low <= pos["sl"] if pos["side"] == 1 else bar.high >= pos["sl"]
            )
            tp_hit = (
                bar.high >= pos["tp"] if pos["side"] == 1 else bar.low <= pos["tp"]
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
                    "exit_time": str(bar.open_time),
                    "exit_fee": exit_fee,
                    "equity_after": self.state["equity"],
                }
                champion_store.insert_paper_trade(trade)
                self.state["trade_count"] = int(self.state.get("trade_count") or 0) + 1
                self.state["trades_on_spec"] = int(self.state.get("trades_on_spec") or 0) + 1
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
                break

        decision: dict[str, Any] = {"side": None, "action": "HOLD", "why": ""}
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
            decision = diagnose_spec_entry(last, x.iloc[-2], s)
            if decision["action"] == "PENDING" and decision.get("side") is not None:
                atr_v = float(last.atr) if pd.notna(last.get("atr")) else float("nan")
                if np.isfinite(atr_v) and atr_v > 0:
                    self.state["pending"] = {
                        "side": int(decision["side"]),
                        "signal_time": candle_id,
                        "atr": atr_v,
                        "genome_id": sid,
                        "leverage": int(ex["leverage"]),
                        "atr_stop": float(ex["atr_stop"]),
                        "reward_r": float(ex["reward_r"]),
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
            "atr_pct": float(last.atr_pct) if pd.notna(last.get("atr_pct")) else None,
            "rsi": float(last.rsi) if "rsi" in last.index and pd.notna(last.get("rsi")) else None,
            "family": s.get("name") or s.get("legacy_family"),
            "genome_id": sid,
            "spec_id": sid,
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
        spec = self.state.get("champion_spec") or {}
        sid = self.state.get("champion_spec_id")
        if not sid and spec:
            sid = spec_id(spec)
        return {
            "equity": float(self.state.get("equity") or config.INITIAL_CAPITAL),
            "initial_capital": config.INITIAL_CAPITAL,
            "pnl_total": float(self.state.get("equity") or 0) - config.INITIAL_CAPITAL,
            "trade_count": int(self.state.get("trade_count") or 0),
            "trades_on_spec": int(self.state.get("trades_on_spec") or 0),
            "has_position": self.state.get("position") is not None,
            "has_pending": self.state.get("pending") is not None,
            "champion_family": spec.get("name") or spec.get("legacy_family"),
            "champion_id": sid,
            "research_due": bool(self.state.get("research_due")),
        }
