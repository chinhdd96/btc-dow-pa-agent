"""Paper trading account: simulate fill entry / SL / TP from live prices."""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperPortfolio:
    """
    Vốn ban đầu PAPER_BALANCE_USDT (1000).
    Mỗi lệnh dùng PAPER_MARGIN_PER_TRADE (100) * LEVERAGE → notional.
    Không gọi Binance order API — khớp entry/SL/TP theo giá live.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or config.PATHS["PAPER_ACCOUNT"])
        if not self.path.exists():
            self._write(self._default())

    def _default(self) -> dict[str, Any]:
        return {
            "started_at": _now_iso(),
            "initial_balance": config.PAPER_BALANCE_USDT,
            "balance": config.PAPER_BALANCE_USDT,
            "margin_per_trade": config.PAPER_MARGIN_PER_TRADE,
            "leverage": config.LEVERAGE,
            "pending": None,
            "position": None,
            "closed_trades": [],
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl": 0.0,
            },
        }

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = self._default()
            self._write(data)
        data.setdefault("stats", {"total_trades": 0, "wins": 0, "losses": 0, "realized_pnl": 0.0})
        data.setdefault("closed_trades", [])
        data.setdefault("pending", None)
        data.setdefault("position", None)
        data.setdefault("balance", config.PAPER_BALANCE_USDT)
        data.setdefault("initial_balance", config.PAPER_BALANCE_USDT)
        data.setdefault("started_at", _now_iso())
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self.path.parent),
            suffix=".tmp",
        ) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp_name = tmp.name
        Path(tmp_name).replace(self.path)

    def balance(self) -> float:
        return float(self._read()["balance"])

    def summary(self) -> dict[str, Any]:
        data = self._read()
        started = data.get("started_at")
        days = 0.0
        try:
            t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - t0).total_seconds() / 86400
        except (TypeError, ValueError, AttributeError):
            pass
        stats = data["stats"]
        return {
            "balance": float(data["balance"]),
            "initial_balance": float(data["initial_balance"]),
            "pnl_total": float(data["balance"]) - float(data["initial_balance"]),
            "days_elapsed": round(days, 2),
            "total_trades": int(stats.get("total_trades", 0)),
            "wins": int(stats.get("wins", 0)),
            "losses": int(stats.get("losses", 0)),
            "pending": data.get("pending"),
            "position": data.get("position"),
        }

    def has_exposure(self) -> bool:
        data = self._read()
        return bool(data.get("pending") or data.get("position"))

    def get_open_position(self) -> dict[str, Any] | None:
        return self._read().get("position")

    def place_signal(
        self,
        action: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        reasoning: str = "",
        regime: str = "",
        order_type: str = "MARKET",
        mark_price: float | None = None,
        signal_level: int | None = None,
        win_probability: float | None = None,
    ) -> dict[str, Any]:
        """Đăng ký lệnh paper. MARKET khớp ngay nếu giá đã chạm entry; LIMIT chờ."""
        action = action.upper()
        data = self._read()
        if data.get("pending") or data.get("position"):
            return {"ok": False, "reason": "Already have pending/open paper position"}

        margin = float(config.PAPER_MARGIN_PER_TRADE)
        if float(data["balance"]) < margin:
            return {
                "ok": False,
                "reason": f"Balance {data['balance']:.2f} < margin {margin}",
            }

        notional = margin * float(config.LEVERAGE)
        qty = notional / entry if entry > 0 else 0.0
        pending = {
            "action": action,
            "entry": float(entry),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "margin": margin,
            "leverage": int(config.LEVERAGE),
            "quantity": qty,
            "notional": notional,
            "order_type": order_type.upper(),
            "reasoning": reasoning,
            "regime": regime,
            "signal_level": signal_level,
            "win_probability": win_probability,
            "created_at": _now_iso(),
        }
        data["pending"] = pending
        self._write(data)

        # MARKET: thử khớp ngay với mark price
        mark = mark_price if mark_price is not None else entry
        events = self.on_price(mark, high=mark, low=mark)
        filled = any(e.get("event") == "filled" for e in events)
        return {
            "ok": True,
            "pending": pending,
            "filled_immediately": filled,
            "events": events,
            "balance": self.balance(),
        }

    def on_price(
        self,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Cập nhật theo giá live (và high/low nến gần nhất nếu có).
        Trả về list event: filled / closed.
        """
        high = high if high is not None else price
        low = low if low is not None else price
        events: list[dict[str, Any]] = []
        data = self._read()

        # 1) Fill pending nếu giá khớp entry
        pending = data.get("pending")
        if pending:
            filled = self._entry_touched(pending, price, high, low)
            if filled:
                pos = {
                    **pending,
                    "filled_at": _now_iso(),
                    "fill_price": float(pending["entry"]),
                }
                data["pending"] = None
                data["position"] = pos
                self._write(data)
                events.append(
                    {
                        "event": "filled",
                        "action": pos["action"],
                        "entry": pos["entry"],
                        "quantity": pos["quantity"],
                        "margin": pos["margin"],
                        "sl": pos["stop_loss"],
                        "tp": pos["take_profit"],
                        "balance": float(data["balance"]),
                    }
                )
                logger.info("Paper FILLED %s @ %s", pos["action"], pos["entry"])

        # 2) Close position nếu chạm SL/TP
        data = self._read()
        position = data.get("position")
        if position:
            exit_info = self._exit_hit(position, price, high, low)
            if exit_info:
                exit_price, reason = exit_info
                closed = self._close_position(data, position, exit_price, reason)
                events.append(closed)
                logger.info(
                    "Paper CLOSED %s %s pnl=%.4f bal=%.2f",
                    reason,
                    position["action"],
                    closed["pnl"],
                    closed["balance"],
                )

        return events

    @staticmethod
    def _entry_touched(
        pending: dict[str, Any],
        price: float,
        high: float,
        low: float,
    ) -> bool:
        entry = float(pending["entry"])
        action = pending["action"]
        order_type = pending.get("order_type", "MARKET")

        # MARKET: coi như khớp nếu giá hiện tại gần entry (±0.15%) hoặc nến đã quét qua
        if order_type == "MARKET":
            if low <= entry <= high:
                return True
            return abs(price - entry) / entry <= 0.0015

        # LIMIT BUY: giá xuống chạm entry; LIMIT SELL: giá lên chạm entry
        if action == "BUY":
            return low <= entry
        return high >= entry

    @staticmethod
    def _exit_hit(
        position: dict[str, Any],
        price: float,
        high: float,
        low: float,
    ) -> tuple[float, str] | None:
        action = position["action"]
        sl = float(position["stop_loss"])
        tp = float(position["take_profit"])

        if action == "BUY":
            # SL trước nếu cùng nến quét cả hai (giả định xấu hơn)
            if low <= sl:
                return sl, "SL"
            if high >= tp:
                return tp, "TP"
            if price <= sl:
                return sl, "SL"
            if price >= tp:
                return tp, "TP"
        else:
            if high >= sl:
                return sl, "SL"
            if low <= tp:
                return tp, "TP"
            if price >= sl:
                return sl, "SL"
            if price <= tp:
                return tp, "TP"
        return None

    def _close_position(
        self,
        data: dict[str, Any],
        position: dict[str, Any],
        exit_price: float,
        reason: str,
    ) -> dict[str, Any]:
        qty = float(position["quantity"])
        entry = float(position["fill_price"] or position["entry"])
        action = position["action"]
        if action == "BUY":
            pnl = qty * (exit_price - entry)
        else:
            pnl = qty * (entry - exit_price)

        data["balance"] = float(data["balance"]) + pnl
        stats = data["stats"]
        stats["total_trades"] = int(stats.get("total_trades", 0)) + 1
        stats["realized_pnl"] = float(stats.get("realized_pnl", 0)) + pnl
        if pnl >= 0:
            stats["wins"] = int(stats.get("wins", 0)) + 1
            result = "WIN"
        else:
            stats["losses"] = int(stats.get("losses", 0)) + 1
            result = "LOSS"

        trade = {
            "action": action,
            "entry": entry,
            "exit": exit_price,
            "sl": position["stop_loss"],
            "tp": position["take_profit"],
            "quantity": qty,
            "margin": position["margin"],
            "pnl": round(pnl, 4),
            "result": result,
            "reason": reason,
            "regime": position.get("regime", ""),
            "reasoning": position.get("reasoning", ""),
            "signal_level": position.get("signal_level"),
            "win_probability": position.get("win_probability"),
            "opened_at": position.get("filled_at"),
            "closed_at": _now_iso(),
        }
        data["closed_trades"].append(trade)
        # giữ tối đa 200 lệnh gần nhất
        data["closed_trades"] = data["closed_trades"][-200:]
        data["position"] = None
        self._write(data)

        return {
            "event": "closed",
            "result": result,
            "reason": reason,
            "pnl": round(pnl, 4),
            "balance": round(float(data["balance"]), 4),
            "trade": trade,
        }
