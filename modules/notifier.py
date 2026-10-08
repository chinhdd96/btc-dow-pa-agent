"""Telegram notifier (nhãn tiếng Việt)."""

from __future__ import annotations

import logging
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

ACTION_VI = {
    "BUY": "MUA (Long)",
    "SELL": "BÁN (Short)",
    "HOLD": "ĐỨNG NGOÀI (HOLD)",
    "CLOSE": "ĐÓNG LỆNH",
}

REGIME_VI = {
    "REGIME_UPTREND": "Xu hướng tăng",
    "REGIME_DOWNTREND": "Xu hướng giảm",
    "REGIME_SIDEWAY_CHOP": "Đi ngang / chop",
    "REGIME_RANGE": "Range",
    "REGIME_RANGE_BREAKOUT_ATTEMPT": "Range — thử breakout",
    "REGIME_RANGE_BREAKOUT_CONFIRMED": "Range — breakout xác nhận",
    "REGIME_RANGE_FAILED_BREAKOUT": "Range — breakout thất bại",
    "REGIME_RANGE_RETEST": "Range — retest",
    "REGIME_RANGE_REJECTION": "Range — rejection",
    "REGIME_RANGE_CONSOLIDATION": "Range — consolidation",
    "REGIME_UPTREND_RETEST": "Uptrend — retest",
    "REGIME_DOWNTREND_RETEST": "Downtrend — retest",
    "REGIME_HIGH_VOLATILITY_NEWS": "Biến động mạnh / tin tức",
}

RESULT_VI = {
    "WIN": "THẮNG",
    "LOSS": "THUA",
}


def _action_vi(action: Any) -> str:
    key = str(action or "").upper()
    return ACTION_VI.get(key, str(action))


def _regime_vi(regime: Any) -> str:
    key = str(regime or "")
    return REGIME_VI.get(key, key or "—")


class Notifier:
    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.token = token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID

    def send(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured; skip notify")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4000],
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("Telegram HTTP %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Telegram send failed: %s", exc)
            return False

    def notify_manage(
        self,
        decision: dict[str, Any],
        mark: float | None = None,
        extra: str = "",
    ) -> None:
        action = str(decision.get("manage_action") or decision.get("action") or "HOLD")
        lines = [
            "BTC Dow/PA — Quản trị lệnh (paper)",
            f"Hành động: {action} | Thesis: {decision.get('thesis_status')} | "
            f"Điểm: {decision.get('manage_score')}",
        ]
        if mark is not None:
            lines.append(f"Giá mark: {mark:.2f}")
        new_sl = decision.get("new_stop_loss")
        if new_sl is not None:
            lines.append(f"SL mới (TRAIL): {new_sl}")
        lines.append(f"Lý do: {str(decision.get('reasoning') or '')[:500]}")
        if extra:
            lines.append(extra)
        self.send("\n".join(lines))

    def notify_decision(
        self,
        decision: dict[str, Any],
        balance: float | None = None,
        extra: str = "",
    ) -> None:
        action = _action_vi(decision.get("action"))
        lines = [
            "BTC Dow/PA — Quyết định giao dịch",
            f"Chế độ: {_regime_vi(decision.get('market_regime'))} | "
            f"State: {decision.get('state', '')} | Bias: {decision.get('short_term_bias', '')}",
            f"Level L{decision.get('signal_level', '?')} | "
            f"Điểm: {decision.get('setup_score')} | "
            f"P(win): {decision.get('win_probability')} | Hành động: {action}",
            f"Trigger: {str(decision.get('trigger', ''))[:200]}",
            f"Invalidation: {str(decision.get('invalidation', ''))[:200]}",
            (
                f"Vào lệnh: {decision.get('entry_price')} | "
                f"Cắt lỗ (SL): {decision.get('stop_loss_price')} | "
                f"Chốt lời (TP): {decision.get('take_profit_price')}"
            ),
            f"Tỷ lệ R:R: {decision.get('risk_reward_ratio')}",
            f"Phân tích Dow: {decision.get('dow_structure_analysis', '')[:400]}",
            f"Tín hiệu Price Action: {decision.get('price_action_signal', '')[:400]}",
            f"Lý do: {decision.get('reasoning', '')[:500]}",
        ]
        if balance is not None:
            lines.append(f"Số dư (USDT): {balance:.4f}")
        if extra:
            lines.append(extra)
        self.send("\n".join(lines))

    def notify_post_mortem(self, lesson: dict[str, Any], result: str, pnl: float) -> None:
        result_u = str(result or "").upper()
        text = (
            "BTC Dow/PA — Hậu phẫu lệnh\n"
            f"Kết quả: {RESULT_VI.get(result_u, result)} | PnL: {pnl:+.4f} USDT\n"
            f"Chế độ lúc vào: {_regime_vi(lesson.get('regime'))}\n"
            f"Nhận định: {lesson.get('mistake_or_insight')}\n"
            f"Bài học: {lesson.get('actionable_rule')}\n"
            f"Trọng số: {lesson.get('weight')}"
        )
        self.send(text)

    def notify_error(self, where: str, error: str) -> None:
        self.send(f"BTC Dow/PA — LỖI\n[{where}]\n{error[:1500]}")
