"""Paper trader driven by a Strategy Spec on live closed candles.

The same class runs the real Champion (persisted, depth-aware fills, callbacks)
and shadow Specs (in-memory state, slippage-model fills, no callbacks).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import config
from modules import champion_store, trade_sim
from modules.features_catalog import ensure_features_for_spec
from modules.indicator_catalog import col_key
from modules.quant_data import estimate_depth_impact
from modules.regime import ensure_regime_columns, regime_of
from modules.strategy_spec import (
    base_name,
    compile_signal,
    diagnose_spec_entry,
    exit_params,
    normalize_spec,
    spec_id,
    spec_requirements,
)

logger = logging.getLogger(__name__)


def interval_td(interval: str | None = None) -> pd.Timedelta:
    iv = (interval or config.TIMEFRAME_MAIN).strip()
    unit = {"m": "min", "h": "h", "d": "D"}[iv[-1]]
    return pd.Timedelta(int(iv[:-1]), unit=unit)


def _ts_str(ms: int) -> str:
    return str(pd.Timestamp(int(ms), unit="ms", tz="UTC"))


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def entry_context(row: pd.Series, spec: dict) -> dict[str, Any]:
    """Indicator values the Spec looked at when it fired + market regime."""
    values: dict[str, Any] = {}
    for ind, params in spec_requirements(spec):
        key = col_key(ind, params if params else None)
        v = row.get(key, row.get(ind))
        try:
            fv = float(v)
            values[key] = round(fv, 6) if np.isfinite(fv) else None
        except (TypeError, ValueError):
            values[key] = None
    return {"indicators": values, "regime": regime_of(row), "close": float(row.close)}


class PaperChampion:
    def __init__(
        self,
        *,
        on_close: Callable[[dict], None] | None = None,
        on_open: Callable[[dict], None] | None = None,
        on_tick: Callable[[dict], None] | None = None,
        state: dict | None = None,
    ) -> None:
        self.shadow = state is not None
        self.on_close = on_close
        self.on_open = on_open
        self.on_tick = on_tick
        self._closed_this_tick: list[dict] = []
        self.last_tick_report: dict[str, Any] | None = None
        if self.shadow:
            self.state = state
            self.state.setdefault("equity", config.INITIAL_CAPITAL)
            return
        champion_store.init_db()
        self.state = champion_store.get_paper_state()
        if "equity" not in self.state:
            self.state["equity"] = config.INITIAL_CAPITAL
        changed = False
        spec = self.state.get("champion_spec")
        if spec and spec.get("name") != base_name(spec.get("name")):
            spec["name"] = base_name(spec.get("name"))
            changed = True
        if self.state.get("timeframe") != config.TIMEFRAME_MAIN:
            if spec or self.state.get("champion"):
                logger.warning(
                    "Clearing Champion: timeframe %s → %s",
                    self.state.get("timeframe"),
                    config.TIMEFRAME_MAIN,
                )
            for key in ("champion", "champion_spec", "champion_spec_id", "pending", "position", "family"):
                self.state[key] = None
            self.state["trades_on_spec"] = 0
            self.state["timeframe"] = config.TIMEFRAME_MAIN
            self.state["research_due"] = True
            changed = True
        if changed:
            self.save()

    # --- persistence / fills -------------------------------------------------

    def save(self) -> None:
        if not self.shadow:
            champion_store.set_paper_state(self.state)

    def reload(self) -> None:
        if not self.shadow:
            self.state = champion_store.get_paper_state()

    def _fill(self, side: int, qty: float, px: float) -> tuple[float, float]:
        if self.shadow:
            return trade_sim.slippage_fill(side, qty, px)
        impact = estimate_depth_impact(config.SYMBOL, side, qty)
        slip = max(config.SLIPPAGE_RATE, impact or 0.0)
        return px * (1 + slip * side), qty

    def _record_trade(self, trade: dict) -> None:
        self.state["trade_count"] = int(self.state.get("trade_count") or 0) + 1
        if self.shadow:
            keep = list(self.state.get("trades") or [])
            keep.append(
                {
                    "entry_time": trade.get("entry_time"),
                    "exit_time": trade.get("exit_time"),
                    "side": trade.get("side"),
                    "pnl": round(float(trade["pnl"]), 6),
                    "r": trade.get("r_multiple"),
                    "reason": trade.get("exit_reason"),
                }
            )
            self.state["trades"] = keep[-50:]
            return
        champion_store.insert_paper_trade(trade)
        self.state["trades_on_spec"] = int(self.state.get("trades_on_spec") or 0) + 1

    # --- champion management ---------------------------------------------------

    def set_champion_spec(
        self,
        spec: dict,
        *,
        baseline_expectancy: float | None = None,
    ) -> None:
        s = normalize_spec(spec)
        s.pop("variant", None)
        s["name"] = base_name(s.get("name"))
        new_id = spec_id(s)
        old = self.state.get("champion_spec")
        old_id = spec_id(old) if old else None
        self.state["champion_spec"] = s
        self.state["champion_spec_id"] = new_id
        if new_id != old_id:
            self.state["trades_on_spec"] = 0
            self.state["spec_since_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
        self.state["research_mode"] = None
        self.state["timeframe"] = config.TIMEFRAME_MAIN
        self.state["champion"] = {
            "family": s.get("name") or "spec",
            "atr_stop": float(s["exit"]["atr_stop"]),
            "reward_r": float(s["exit"]["reward_r"]),
            "leverage": int(s["exit"]["leverage"]),
            "mtf_filter": bool(s.get("mtf_filter")),
        }
        if baseline_expectancy is not None:
            self.state["champion_baseline_expectancy"] = baseline_expectancy
        self.state["research_due"] = False
        self.save()

    def _spec(self) -> Optional[dict]:
        spec = self.state.get("champion_spec")
        return normalize_spec(spec) if spec else None

    def _label(self, spec: dict) -> str:
        if self.shadow:
            return str(self.state.get("name") or spec.get("name") or "shadow")
        from modules.family import label

        return label(self.state) or spec.get("name") or "spec"

    # --- position mechanics ------------------------------------------------------

    def _open_from_pending(self, ref_open: float, time_str: str, spec: dict) -> bool:
        pending = self.state.get("pending") or {}
        ex = dict(pending.get("exit_cfg") or exit_params(spec))
        ex.setdefault("atr_stop", pending.get("atr_stop"))
        ex.setdefault("reward_r", pending.get("reward_r"))
        ex.setdefault("leverage", pending.get("leverage"))
        sid = pending.get("spec_id") or spec_id(spec)
        pos = trade_sim.open_position(
            side=int(pending["side"]),
            ref_open=ref_open,
            atr=float(pending["atr"]),
            exit_cfg=ex,
            equity=float(self.state["equity"]),
            time=time_str,
            fill_fn=self._fill,
            size_mult=(
                config.FAMILY_EXPLORE_SIZE_MULT
                if not self.shadow and self.state.get("explore_mode")
                else 1.0
            ),
            extra={
                "genome_id": sid,
                "spec_id": sid,
                "family": self._label(spec),
                "exit_cfg": ex,
                "entry_context": pending.get("entry_context"),
                "signal_time": pending.get("signal_time"),
            },
        )
        self.state["pending"] = None
        if pos is None:
            logger.info("Pending dropped: target does not cover costs at %.2f", ref_open)
            return False
        self.state["position"] = pos
        if not self.shadow:
            self.state["last_entry_ts"] = pd.Timestamp(time_str).isoformat()
            logger.info(
                "PAPER OPEN %s entry=%.2f lev=%sx SL=%.2f TP=%.2f spec=%s",
                "LONG" if pos["side"] == 1 else "SHORT",
                pos["entry"],
                pos["leverage"],
                pos["sl"],
                pos["tp"],
                sid,
            )
        if self.on_open:
            try:
                self.on_open(pos)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_open callback failed: %s", exc)
        return True

    def _close(self, reason: str, px: float, time_str: str, source: str) -> dict:
        pos = self.state["position"]
        trade = trade_sim.close_position(
            pos, px=px, reason=reason, time=time_str, fill_fn=self._fill
        )
        trade["exit_source"] = source
        self.state["equity"] = float(self.state["equity"]) + trade["trading_pnl"]
        trade["equity_after"] = self.state["equity"]
        self.state["position"] = None
        self._record_trade(trade)
        self._closed_this_tick.append(trade)
        if not self.shadow:
            logger.info(
                "PAPER CLOSE %s (%s) pnl=%+.4f equity=%.2f",
                reason,
                source,
                trade["pnl"],
                self.state["equity"],
            )
        if self.on_close:
            try:
                self.on_close(trade)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_close callback failed: %s", exc)
        return trade

    # --- live watch between candle closes -----------------------------------------

    def pending_fill_ms(self) -> Optional[int]:
        """Open time (ms) of the bar a pending signal fills on."""
        pend = self.state.get("pending")
        if not pend or self.state.get("position") is not None:
            return None
        return _ms(pend["signal_time"]) + int(interval_td().total_seconds() * 1000)

    def check_live(self, bars_1m: list[dict]) -> list[dict]:
        """Fill pending at the next bar open and close on SL/TP/LIQ touches (1m high/low)."""
        self._closed_this_tick = []
        spec = self._spec()
        if spec is None or not bars_1m:
            return []
        changed = False
        fill_ms = self.pending_fill_ms()
        if fill_ms is not None:
            bar = next((b for b in bars_1m if b["open_time_ms"] == fill_ms), None)
            if bar is not None:
                self._open_from_pending(bar["open"], _ts_str(fill_ms), spec)
                changed = True
        pos = self.state.get("position")
        if pos:
            entry_ms = _ms(pos["entry_time"])
            for bar in bars_1m:
                if bar["open_time_ms"] < entry_ms:
                    continue
                hit = trade_sim.check_hit(pos, bar["high"], bar["low"])
                if hit:
                    now = str(pd.Timestamp.now(tz="UTC").floor("s"))
                    self._close(hit[0], hit[1], now, "live")
                    changed = True
                    break
        if changed:
            self.save()
        return list(self._closed_this_tick)

    # --- closed-candle processing ------------------------------------------------------

    def process(self, df: pd.DataFrame) -> list[dict]:
        self._closed_this_tick = []
        self.last_tick_report = None
        s = self._spec()
        if not s:
            return self._process_no_spec(df)

        sid = spec_id(s)
        x = ensure_regime_columns(ensure_features_for_spec(df, s)).reset_index(drop=True)
        if x.empty:
            return []
        last = x.iloc[-1]
        candle_id = str(last.open_time)
        if self.state.get("last_candle") == candle_id:
            return []

        fn = compile_signal(s)
        opened_this = False
        closed_reason: str | None = None
        pending = self.state.get("pending")
        if pending and pending["signal_time"] != candle_id and self.state.get("position") is None:
            opened_this = self._open_from_pending(float(last.open), candle_id, s)

        pos = self.state.get("position")
        check_idx = [len(x) - 1]
        prev_candle = self.state.get("last_candle")
        if pos and not opened_this and prev_candle:
            times = x["open_time"].astype(str)
            hit_idx = times.index[times == str(prev_candle)]
            if len(hit_idx):
                check_idx = list(range(int(hit_idx[0]) + 1, len(x))) or check_idx
        for j in check_idx if pos else []:
            bar = x.iloc[j]
            pos = self.state["position"]
            ex = pos.get("exit_cfg") or exit_params(s)
            fr = bar.get("funding_event_rate", np.nan)
            self.state["equity"] = float(self.state["equity"]) + trade_sim.apply_funding(
                pos, float(fr) if pd.notna(fr) else np.nan
            )
            hit = trade_sim.check_hit(pos, float(bar.high), float(bar.low))
            if hit is None and ex.get("exit_on_opposite") and j >= 1:
                sig = fn(bar, x.iloc[j - 1])
                hit = trade_sim.bar_close_exit(pos, close=float(bar.close), exit_cfg=ex, signal_side=sig)
            elif hit is None:
                hit = trade_sim.bar_close_exit(pos, close=float(bar.close), exit_cfg=ex, signal_side=None)
            if hit:
                self._close(hit[0], hit[1], str(bar.open_time), "candle")
                closed_reason = hit[0]
                break
            atr_v = float(bar.atr) if pd.notna(bar.get("atr")) else float("nan")
            trade_sim.update_exit_levels(
                pos,
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                atr=atr_v,
                exit_cfg=ex,
            )

        decision: dict[str, Any]
        if self.state.get("position") is not None:
            pos = self.state["position"]
            side_txt = "LONG" if pos["side"] == 1 else "SHORT"
            decision = {
                "side": pos["side"],
                "action": "HOLD",
                "why": (
                    f"đang {side_txt} — chờ SL/TP (sl={float(pos['sl']):.0f} "
                    f"tp={float(pos['tp']):.0f})"
                ),
            }
        elif self.state.get("pending") is not None:
            pend = self.state["pending"]
            decision = {
                "side": pend["side"],
                "action": "PENDING",
                "why": f"đã có pending {'LONG' if pend['side'] == 1 else 'SHORT'} — chờ nến sau để fill",
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
                    ex = exit_params(s)
                    self.state["pending"] = {
                        "side": int(decision["side"]),
                        "signal_time": candle_id,
                        "atr": atr_v,
                        "spec_id": sid,
                        "genome_id": sid,
                        "leverage": int(ex["leverage"]),
                        "atr_stop": float(ex["atr_stop"]),
                        "reward_r": float(ex["reward_r"]),
                        "exit_cfg": ex,
                        "entry_context": entry_context(last, s),
                    }

        if opened_this and self.state.get("position") is not None:
            pos = self.state["position"]
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
            "regime": regime_of(last).get("tag"),
            "family": self._label(s),
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
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_tick callback failed: %s", exc)
        return list(self._closed_this_tick)

    def _process_no_spec(self, df: pd.DataFrame) -> list[dict]:
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
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_tick callback failed: %s", exc)
        return []

    def summary(self) -> dict[str, Any]:
        spec = self.state.get("champion_spec") or {}
        sid = self.state.get("champion_spec_id") or (spec_id(spec) if spec else None)
        return {
            "equity": float(self.state.get("equity") or config.INITIAL_CAPITAL),
            "initial_capital": config.INITIAL_CAPITAL,
            "pnl_total": float(self.state.get("equity") or 0) - config.INITIAL_CAPITAL,
            "trade_count": int(self.state.get("trade_count") or 0),
            "trades_on_spec": int(self.state.get("trades_on_spec") or 0),
            "has_position": self.state.get("position") is not None,
            "has_pending": self.state.get("pending") is not None,
            "champion_family": self._label(spec) if spec else None,
            "champion_id": sid,
            "research_due": bool(self.state.get("research_due")),
        }
