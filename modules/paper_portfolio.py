"""Paper trading account: simulate fill entry / SL / TP from live prices."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_ms(value: Any) -> int | None:
    """Parse ISO timestamp → unix ms UTC. None if invalid."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            v = int(value)
            return v if v > 10_000_000_000 else v * 1000
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (TypeError, ValueError, OSError):
        return None


def _paper_path() -> Path:
    return Path(config.PATHS["PAPER_ACCOUNT"])


def _slim_for_remote(data: dict[str, Any]) -> dict[str, Any]:
    """Keep account state; trim closed_trades tail for GitHub size."""
    out = dict(data)
    trades = list(out.get("closed_trades") or [])
    out["closed_trades"] = trades[-50:]
    return out


def _github_headers() -> dict[str, str] | None:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return None
    return {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "btc-dow-pa-agent-paper",
    }


def backup_remote(data: dict[str, Any] | None = None) -> bool:
    """Upsert paper_account.json to history-data branch (Blitz rebuild-safe)."""
    headers = _github_headers()
    if not headers:
        return False
    path_local = _paper_path()
    if data is None:
        if not path_local.exists():
            return False
        try:
            data = json.loads(path_local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
    payload = _slim_for_remote(data)
    repo = config.HISTORY_GITHUB_REPO
    remote_path = config.PAPER_GITHUB_PATH
    branch = config.HISTORY_GITHUB_BRANCH or "history-data"
    content_b64 = base64.b64encode(
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    ).decode("ascii")
    try:
        from modules.decision_history import (
            _ensure_history_branch,
            _github_contents_url,
        )

        if not _ensure_history_branch(headers, repo, branch):
            return False
        meta = requests.get(
            _github_contents_url(repo, remote_path, branch),
            headers=headers,
            timeout=30,
        )
        sha = None
        if meta.status_code == 200:
            sha = meta.json().get("sha")
        elif meta.status_code != 404:
            logger.warning("paper remote GET HTTP %s", meta.status_code)
            return False
        body: dict[str, Any] = {
            "message": "chore: sync paper_account for Blitz persistence",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        put = requests.put(
            _github_contents_url(repo, remote_path),
            headers=headers,
            json=body,
            timeout=60,
        )
        if put.status_code not in {200, 201}:
            logger.warning(
                "paper remote PUT HTTP %s: %s", put.status_code, put.text[:200]
            )
            return False
        logger.info(
            "paper_account backed up to GitHub branch=%s bal=%.2f",
            branch,
            float(payload.get("balance") or 0),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("paper remote backup failed: %s", exc)
        return False


def restore_from_remote() -> bool:
    """If local paper_account missing, pull from GitHub. Returns True if restored."""
    path = _paper_path()
    if path.exists():
        return False
    headers = _github_headers()
    if not headers:
        logger.info("No HISTORY_GITHUB_TOKEN — skip paper restore")
        return False
    repo = config.HISTORY_GITHUB_REPO
    remote_path = config.PAPER_GITHUB_PATH
    branch = config.HISTORY_GITHUB_BRANCH or "history-data"
    try:
        from modules.decision_history import _github_contents_url

        def _load(br: str) -> dict[str, Any] | None:
            resp = requests.get(
                _github_contents_url(repo, remote_path, br),
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                logger.warning("paper remote GET %s HTTP %s", br, resp.status_code)
                return None
            raw = base64.b64decode(resp.json().get("content") or "").decode("utf-8")
            return json.loads(raw)

        data = _load(branch)
        source = branch
        if data is None:
            data = _load("main")
            source = "main"
        if not data or not isinstance(data, dict):
            logger.info("No remote paper_account yet (branch=%s)", branch)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "paper_account restored from GitHub branch=%s bal=%.2f started=%s",
            source,
            float(data.get("balance") or 0),
            data.get("started_at"),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("paper remote restore failed: %s", exc)
        return False


def ensure_restored() -> bool:
    """Call once at process start before PaperPortfolio() — never overwrites local."""
    path = _paper_path()
    if path.exists():
        logger.info("paper_account local exists — keep as-is")
        return False
    return restore_from_remote()


class PaperPortfolio:
    """
    Vốn ban đầu PAPER_BALANCE_USDT (1000).
    Mỗi lệnh dùng PAPER_MARGIN_PER_TRADE (100) * LEVERAGE → notional.
    Không gọi Binance order API — khớp entry/SL/TP theo giá live.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or config.PATHS["PAPER_ACCOUNT"])
        if not self.path.exists():
            # Don't backup fresh default — would overwrite remote after failed restore
            self._write(self._default(), backup=False)

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
            self._write(data, backup=False)
        data.setdefault("stats", {"total_trades": 0, "wins": 0, "losses": 0, "realized_pnl": 0.0})
        data.setdefault("closed_trades", [])
        data.setdefault("pending", None)
        data.setdefault("position", None)
        data.setdefault("balance", config.PAPER_BALANCE_USDT)
        data.setdefault("initial_balance", config.PAPER_BALANCE_USDT)
        data.setdefault("started_at", _now_iso())
        return data

    def _write(self, data: dict[str, Any], *, backup: bool = True) -> None:
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
        if backup:
            try:
                backup_remote(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("paper backup after write failed: %s", exc)

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
        trigger: str = "",
        invalidation: str = "",
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
            "initial_stop_loss": float(stop_loss),
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
            "trigger": trigger,
            "invalidation": invalidation,
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
        path_candles: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Cập nhật theo giá live (và high/low nến gần nhất nếu có).
        path_candles: nến 1m từ lúc fill → nay; quét tuần tự SL/TP (ưu tiên hơn 1 nến 1H).
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
                    "initial_stop_loss": float(
                        pending.get("initial_stop_loss") or pending["stop_loss"]
                    ),
                    "stop_armed": True,  # classic SL beyond entry is live immediately
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

        # 2) Close position nếu chạm SL/TP (path từ fill nếu có)
        data = self._read()
        position = data.get("position")
        if position:
            exit_info: tuple[float, str] | None
            if path_candles:
                exit_info, armed = self._exit_hit_on_path(
                    position, path_candles, mark=price
                )
                if bool(position.get("stop_armed", True)) != armed:
                    position = dict(position)
                    position["stop_armed"] = armed
                    data["position"] = position
                    self._write(data)
                    data = self._read()
                    position = data.get("position")
            else:
                exit_info = self._exit_hit(position, price, high, low)
            if exit_info and position:
                exit_price, reason = exit_info
                closed = self._close_position(data, position, exit_price, reason)
                events.append(closed)
                logger.info(
                    "Paper CLOSED %s %s pnl=%.4f bal=%.2f path_bars=%s",
                    reason,
                    position["action"],
                    closed["pnl"],
                    closed["balance"],
                    len(path_candles or []),
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

    @classmethod
    def _exit_hit_on_path(
        cls,
        position: dict[str, Any],
        candles: list[dict[str, Any]],
        *,
        mark: float,
    ) -> tuple[tuple[float, str] | None, bool]:
        """
        Quét nến từ lúc fill theo thứ tự thời gian.
        Trả (exit|None, stop_armed).
        """
        fill_ms = _parse_iso_ms(position.get("filled_at")) or 0
        entry = float(position.get("fill_price") or position.get("entry") or 0)
        action = str(position.get("action") or "").upper()
        sl = float(position["stop_loss"])
        # Profit-lock stop (SELL SL < entry / BUY SL > entry) needs arming first
        armed = bool(position.get("stop_armed", True))
        if action == "SELL" and sl < entry:
            armed = bool(position.get("stop_armed", False))
        elif action == "BUY" and sl > entry:
            armed = bool(position.get("stop_armed", False))

        ordered = sorted(candles, key=lambda c: int(c.get("open_time") or 0))
        for c in ordered:
            ot = int(c.get("open_time") or 0)
            if fill_ms and ot + 60_000 < fill_ms:
                # bar fully before fill — skip
                continue
            hi = float(c.get("high", c.get("close", mark)))
            lo = float(c.get("low", c.get("close", mark)))
            cl = float(c.get("close", mark))
            was_armed = armed
            armed = cls._update_stop_arm(action, entry, sl, lo, hi, cl, armed)
            # Profit-lock: không đóng SL ngay nến vừa arm (OHLC không biết thứ tự wick)
            hit = cls._exit_hit_bar(
                action,
                entry,
                sl,
                float(position["take_profit"]),
                lo,
                hi,
                cl,
                was_armed,
            )
            if hit:
                return hit, armed

        # Final mark tick — dùng trạng thái armed sau path
        armed = cls._update_stop_arm(action, entry, sl, mark, mark, mark, armed)
        hit = cls._exit_hit_bar(
            action,
            entry,
            sl,
            float(position["take_profit"]),
            mark,
            mark,
            mark,
            armed,
        )
        return hit, armed

    @staticmethod
    def _update_stop_arm(
        action: str,
        entry: float,
        sl: float,
        low: float,
        high: float,
        close: float,
        armed: bool,
    ) -> bool:
        """Arm profit-lock stop only after price trades through the stop level."""
        if armed:
            return True
        if action == "SELL" and sl < entry:
            # Short lock: arm when price has been at/below SL
            return low <= sl or close <= sl
        if action == "BUY" and sl > entry:
            return high >= sl or close >= sl
        return True

    @staticmethod
    def _exit_hit_bar(
        action: str,
        entry: float,
        sl: float,
        tp: float,
        low: float,
        high: float,
        price: float,
        armed: bool,
    ) -> tuple[float, str] | None:
        if action == "BUY":
            # SL before TP if both touched (adverse assumption)
            if sl <= entry:
                if low <= sl or price <= sl:
                    return sl, "SL"
            elif armed and (low <= sl or price <= sl):
                # Profit-lock SL above entry: only after armed
                return sl, "SL"
            if high >= tp or price >= tp:
                return tp, "TP"
        else:
            # Adverse first: SL (price up) before TP (price down) if both in bar
            if sl >= entry:
                if high >= sl or price >= sl:
                    return sl, "SL"
            elif armed and (high >= sl or price >= sl):
                return sl, "SL"
            if low <= tp or price <= tp:
                return tp, "TP"
        return None

    @staticmethod
    def _exit_hit(
        position: dict[str, Any],
        price: float,
        high: float,
        low: float,
    ) -> tuple[float, str] | None:
        action = str(position.get("action") or "").upper()
        entry = float(position.get("fill_price") or position.get("entry") or 0)
        sl = float(position["stop_loss"])
        tp = float(position["take_profit"])
        armed = bool(position.get("stop_armed", True))
        if action == "SELL" and sl < entry:
            armed = bool(position.get("stop_armed", False))
            # Without path: refuse SL if stop already below mark (breached / not armed)
            if not armed and price > sl:
                armed = False
        elif action == "BUY" and sl > entry:
            armed = bool(position.get("stop_armed", False))
            if not armed and price < sl:
                armed = False
        return PaperPortfolio._exit_hit_bar(
            action, entry, sl, tp, low, high, price, armed
        )

    def cancel_pending(self) -> dict[str, Any]:
        """Hủy lệnh chờ khớp (manage CANCEL_PENDING)."""
        data = self._read()
        pending = data.get("pending")
        if not pending:
            return {"ok": False, "reason": "No pending order"}
        data["pending"] = None
        self._write(data)
        logger.info("Paper CANCEL_PENDING %s @ %s", pending.get("action"), pending.get("entry"))
        return {
            "ok": True,
            "event": "cancelled",
            "pending": pending,
            "balance": self.balance(),
        }

    def close_now(
        self,
        mark_price: float,
        reason: str = "MANAGE_CLOSE",
    ) -> dict[str, Any]:
        """Đóng position đang mở theo giá mark (manage CLOSE)."""
        data = self._read()
        position = data.get("position")
        if not position:
            return {"ok": False, "reason": "No open position"}
        closed = self._close_position(data, position, float(mark_price), reason)
        logger.info(
            "Paper CLOSE_NOW %s pnl=%.4f bal=%.2f",
            reason,
            closed["pnl"],
            closed["balance"],
        )
        return {"ok": True, **closed}

    def update_stop_loss(
        self,
        new_sl: float,
        mark_price: float | None = None,
    ) -> dict[str, Any]:
        """Cập nhật SL (manage TRAIL). Caller phải đã validate chặt hơn."""
        data = self._read()
        position = data.get("position")
        if not position:
            return {"ok": False, "reason": "No open position"}
        old_sl = float(position["stop_loss"])
        new_sl_f = float(new_sl)
        side = str(position.get("action") or "").upper()
        # Defense: refuse SL already breached by mark (would instant-close)
        if mark_price is not None:
            mark = float(mark_price)
            if side == "BUY" and new_sl_f >= mark:
                return {
                    "ok": False,
                    "reason": f"BUY SL {new_sl_f} >= mark {mark} — already breached",
                }
            if side == "SELL" and new_sl_f <= mark:
                return {
                    "ok": False,
                    "reason": f"SELL SL {new_sl_f} <= mark {mark} — already breached",
                }
        if position.get("initial_stop_loss") is None:
            position["initial_stop_loss"] = old_sl
        entry = float(position.get("fill_price") or position.get("entry") or 0)
        position["stop_loss"] = new_sl_f
        position["sl_updated_at"] = _now_iso()
        # Profit-lock: arm ngay nếu mark đã đi qua SL (đúng phía lãi); else chờ path
        mark_f = float(mark_price) if mark_price is not None else None
        if side == "SELL" and new_sl_f < entry:
            position["stop_armed"] = bool(mark_f is not None and mark_f <= new_sl_f)
        elif side == "BUY" and new_sl_f > entry:
            position["stop_armed"] = bool(mark_f is not None and mark_f >= new_sl_f)
        else:
            position["stop_armed"] = True
        data["position"] = position
        self._write(data)
        logger.info("Paper TRAIL SL %s → %s", old_sl, new_sl_f)
        return {
            "ok": True,
            "event": "trail",
            "old_sl": old_sl,
            "new_sl": new_sl_f,
            "position": position,
        }

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
