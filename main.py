"""
Mono-Agent BTC Dow & PA Trader — main orchestrator.

Runs 24/7 locally:
- Trade decision every TIMEFRAME_MAIN (1h)
- Learner + news crawlers every 24h
- Memory compaction Sundays 00:00 UTC
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import schedule

import config
from modules.binance_client import BinanceClient, BinanceClientError
from modules import candle_store
from modules import decision_history
from modules.learner_crawler import run_learner_crawler
from modules.llm_agent import LLMAgent
from modules.memory_manager import MemoryManager
from modules.news_crawler import run_news_crawler
from modules.notifier import Notifier
from modules import paper_portfolio as paper_portfolio_mod
from modules.paper_portfolio import PaperPortfolio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _load_state() -> dict[str, Any]:
    path = Path(config.PATHS["STATE"])
    if not path.exists():
        return {
            "had_position": False,
            "last_decision": None,
            "position_snapshot": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "had_position": False,
            "last_decision": None,
            "position_snapshot": None,
        }


def _save_state(state: dict[str, Any]) -> None:
    path = Path(config.PATHS["STATE"])
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_news_context() -> dict[str, Any]:
    path = Path(config.PATHS["CONTEXT_NEWS"])
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def effective_min_score(memory: MemoryManager) -> float:
    base = memory.get_min_score()
    news = _load_news_context()
    if str(news.get("news_risk_level", "")).upper() == "HIGH":
        return max(base, config.PENALTY_MIN_SCORE)
    return base


def validate_decision_hard(
    decision: dict[str, Any],
    min_score: float,
    binance: BinanceClient,
    primary_bias: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Apply Python hard guards (incl. Dow primary bias)."""
    action = str(decision.get("action", "HOLD")).upper()
    if action == "HOLD":
        return decision, None

    if config.DOW_DIRECTION_GATE:
        bias = str(
            primary_bias or decision.get("primary_bias") or ""
        ).upper()
        if "SIDEWAY" in bias:
            decision = dict(decision)
            decision["action"] = "HOLD"
            reason = "Chặn Dow: primary SIDEWAY → HOLD"
            decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
            return decision, reason
        if bias == "UPTREND" and action == "SELL":
            decision = dict(decision)
            decision["action"] = "HOLD"
            reason = "Chặn Dow: primary UPTREND — không SELL"
            decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
            return decision, reason
        if bias == "DOWNTREND" and action == "BUY":
            decision = dict(decision)
            decision["action"] = "HOLD"
            reason = "Chặn Dow: primary DOWNTREND — không BUY"
            decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
            return decision, reason

    score = float(decision.get("setup_score") or 0)
    if score < min_score:
        decision = dict(decision)
        decision["action"] = "HOLD"
        reason = f"Chặn cứng: điểm {score} < ngưỡng {min_score}"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    entry = decision.get("entry_price")
    sl = decision.get("stop_loss_price")
    tp = decision.get("take_profit_price")
    if entry is None:
        entry = None  # will use mark price later
        decision = dict(decision)

    try:
        mark = binance.get_mark_price()
    except BinanceClientError:
        mark = float(entry or 0)

    use_entry = float(entry) if entry else mark
    decision["entry_price"] = use_entry

    ok, reason = binance.validate_stop_loss(action, use_entry, sl)
    if not ok:
        decision = dict(decision)
        decision["action"] = "HOLD"
        decision["reasoning"] = f"Chặn SL: {reason}. {decision.get('reasoning', '')}"
        return decision, reason

    if tp is None or float(tp) <= 0:
        decision = dict(decision)
        decision["action"] = "HOLD"
        reason = "Chặn cứng: thiếu giá chốt lời (TP)"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    # Directional TP check
    if action == "BUY" and float(tp) <= use_entry:
        decision = dict(decision)
        decision["action"] = "HOLD"
        reason = "Chặn cứng: BUY thì TP phải > giá vào"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason
    if action == "SELL" and float(tp) >= use_entry:
        decision = dict(decision)
        decision["action"] = "HOLD"
        reason = "Chặn cứng: SELL thì TP phải < giá vào"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    # Minimum R:R 1.5
    risk = abs(use_entry - float(sl))
    reward = abs(float(tp) - use_entry)
    if risk <= 0 or reward / risk < config.MIN_RR:
        decision = dict(decision)
        decision["action"] = "HOLD"
        reason = f"Chặn cứng: R:R {reward / risk if risk else 0:.2f} < {config.MIN_RR}"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    decision["risk_reward_ratio"] = round(reward / risk, 2)
    return decision, None


def validate_manage_hard(
    decision: dict[str, Any],
    *,
    position: dict[str, Any] | None,
    pending: dict[str, Any] | None,
    mark: float,
) -> tuple[dict[str, Any], str | None]:
    """Python hard gates for paper manage mode."""
    decision = dict(decision)
    action = str(decision.get("manage_action") or "HOLD").upper()
    thesis = str(decision.get("thesis_status") or "INTACT").upper()
    try:
        score = float(decision.get("manage_score") or 0)
    except (TypeError, ValueError):
        score = 0.0

    if position:
        allowed = {"HOLD", "CLOSE", "TRAIL"}
    elif pending:
        allowed = {"HOLD", "CANCEL_PENDING"}
    else:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        return decision, "Không có exposure"

    if action not in allowed:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = f"Chặn manage: {action} không hợp lệ khi {'position' if position else 'pending'}"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    if action == "HOLD":
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        return decision, None

    if score < config.MANAGE_MIN_SCORE:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = f"Chặn manage: điểm {score} < {config.MANAGE_MIN_SCORE}"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    if action == "CLOSE":
        if thesis not in {"INVALIDATED", "WEAKENING"}:
            decision["manage_action"] = "HOLD"
            decision["action"] = "MANAGE_HOLD"
            reason = f"Chặn CLOSE: thesis_status={thesis} (cần WEAKENING/INVALIDATED)"
            decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
            return decision, reason
        decision["manage_action"] = "CLOSE"
        decision["action"] = "MANAGE_CLOSE"
        return decision, None

    if action == "CANCEL_PENDING":
        decision["manage_action"] = "CANCEL_PENDING"
        decision["action"] = "MANAGE_CANCEL_PENDING"
        return decision, None

    # TRAIL
    new_sl = decision.get("new_stop_loss")
    if new_sl is None:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = "Chặn TRAIL: thiếu new_stop_loss"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    pos = position or {}
    side = str(pos.get("action") or "").upper()
    entry = float(pos.get("fill_price") or pos.get("entry") or 0)
    old_sl = float(pos.get("stop_loss") or 0)
    initial_sl = float(pos.get("initial_stop_loss") or old_sl)
    qty = float(pos.get("quantity") or 0)
    new_sl_f = float(new_sl)

    if side == "BUY":
        upnl = qty * (mark - entry)
        tighter = new_sl_f > old_sl
        at_or_beyond_be = new_sl_f >= entry
        r_dist = abs(entry - initial_sl) if initial_sl else 0.0
        locked_r = ((new_sl_f - entry) / r_dist) if r_dist > 0 else 0.0
    elif side == "SELL":
        upnl = qty * (entry - mark)
        tighter = new_sl_f < old_sl
        at_or_beyond_be = new_sl_f <= entry
        r_dist = abs(entry - initial_sl) if initial_sl else 0.0
        locked_r = ((entry - new_sl_f) / r_dist) if r_dist > 0 else 0.0
    else:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = "Chặn TRAIL: side không hợp lệ"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    if upnl <= 0:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = f"Chặn TRAIL: chưa lãi (uPnL={upnl:+.4f})"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason
    if not tighter:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = f"Chặn TRAIL: SL mới không chặt hơn (old={old_sl} new={new_sl_f})"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason
    if not at_or_beyond_be:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = f"Chặn TRAIL: SL mới chưa về/khóa BE (entry={entry} new={new_sl_f})"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason
    # Prefer lock ≥ TRAIL_MIN_LOCK_R; still allow pure breakeven (locked_r >= 0)
    if locked_r < 0:
        decision["manage_action"] = "HOLD"
        decision["action"] = "MANAGE_HOLD"
        reason = f"Chặn TRAIL: locked_r={locked_r:.2f} < 0"
        decision["reasoning"] = f"{reason}. {decision.get('reasoning', '')}"
        return decision, reason

    decision["manage_action"] = "TRAIL"
    decision["action"] = "MANAGE_TRAIL"
    decision["new_stop_loss"] = new_sl_f
    decision["locked_r"] = round(locked_r, 3)
    return decision, None


def handle_closed_position(
    binance: BinanceClient,
    memory: MemoryManager,
    llm: LLMAgent,
    notifier: Notifier,
    state: dict[str, Any],
) -> None:
    snap = state.get("position_snapshot") or {}
    last = state.get("last_decision") or {}
    action = snap.get("side") or last.get("action") or "BUY"
    entry = float(snap.get("entry_price") or last.get("entry_price") or 0)
    sl = last.get("stop_loss_price")
    tp = last.get("take_profit_price")
    regime = last.get("market_regime", "REGIME_SIDEWAY_CHOP")
    reasoning = last.get("reasoning", "")

    # Approximate PnL from last unrealized if available; else 0
    pnl = float(snap.get("unrealized_pnl") or 0)
    # Prefer mark vs entry for sign if pnl missing
    try:
        mark = binance.get_mark_price()
        if entry > 0 and pnl == 0:
            amt = float(snap.get("amount") or 0)
            if action == "BUY":
                pnl = (mark - entry) * amt
            else:
                pnl = (entry - mark) * amt
    except BinanceClientError:
        pass

    result = "WIN" if pnl >= 0 else "LOSS"
    try:
        candles = binance.get_klines(config.TIMEFRAME_MAIN, limit=20)
    except BinanceClientError:
        candles = []

    lesson = llm.run_post_mortem(
        action=action,
        entry=entry,
        sl=sl,
        tp=tp,
        result=result,
        pnl=pnl,
        regime=regime,
        entry_reasoning=reasoning,
        post_trade_candles=candles,
    )
    memory.add_lesson(
        regime=lesson.get("regime", regime),
        mistake_or_insight=lesson.get("mistake_or_insight", ""),
        actionable_rule=lesson.get("actionable_rule", ""),
        weight=int(lesson.get("weight", 1) or 1),
        classification=str(lesson.get("classification") or ""),
    )
    stats = memory.record_closed_trade(result)
    notifier.notify_post_mortem(lesson, result, pnl)
    notifier.send(
        f"Cập nhật thống kê: thua liên tiếp={stats.get('consecutive_losses')} | "
        f"điểm tối thiểu={stats.get('current_min_score')} | "
        f"tỷ lệ thắng={stats.get('win_rate')}"
    )


def _paper_status_text(paper: PaperPortfolio) -> str:
    s = paper.summary()
    return (
        f"Paper: số dư={s['balance']:.2f}u (đầu {s['initial_balance']:.0f}u) "
        f"PnL={s['pnl_total']:+.2f}u | ngày={s['days_elapsed']:.1f}/30 "
        f"lệnh={s['total_trades']} Thắng{s['wins']}/Thua{s['losses']}"
    )


def handle_paper_closed(
    memory: MemoryManager,
    llm: LLMAgent,
    notifier: Notifier,
    closed_event: dict[str, Any],
    binance: BinanceClient,
) -> None:
    trade = closed_event.get("trade") or {}
    result = closed_event.get("result", "LOSS")
    pnl = float(closed_event.get("pnl", 0))
    try:
        candles = binance.get_klines(config.TIMEFRAME_MAIN, limit=20)
    except BinanceClientError:
        candles = []
    lesson = llm.run_post_mortem(
        action=trade.get("action", ""),
        entry=float(trade.get("entry") or 0),
        sl=trade.get("sl"),
        tp=trade.get("tp"),
        result=result,
        pnl=pnl,
        regime=trade.get("regime", "REGIME_SIDEWAY_CHOP"),
        entry_reasoning=trade.get("reasoning", ""),
        post_trade_candles=candles,
    )
    memory.add_lesson(
        regime=lesson.get("regime", trade.get("regime", "REGIME_SIDEWAY_CHOP")),
        mistake_or_insight=lesson.get("mistake_or_insight", ""),
        actionable_rule=lesson.get("actionable_rule", ""),
        weight=int(lesson.get("weight", 1) or 1),
        classification=str(lesson.get("classification") or ""),
    )
    stats = memory.record_closed_trade(result)
    notifier.notify_post_mortem(lesson, result, pnl)
    result_vi = "THẮNG" if str(result).upper() == "WIN" else "THUA"
    notifier.send(
        f"PAPER ĐÓNG LỆNH [{closed_event.get('reason')}] {result_vi} "
        f"PnL={pnl:+.4f}u\n"
        f"{_paper_status_text(PaperPortfolio())}\n"
        f"điểm tối thiểu (memory)={stats.get('current_min_score')}"
    )


def job_trade(
    binance: BinanceClient,
    memory: MemoryManager,
    llm: LLMAgent,
    notifier: Notifier,
    paper: PaperPortfolio | None = None,
) -> None:
    logger.info("=== Trade job start ===")
    state = _load_state()

    # --- PAPER MODE: giả lập khớp entry / SL / TP theo giá live ---
    if binance.paper_mode:
        paper = paper or PaperPortfolio()
        try:
            price = binance.get_mark_price()
            hist = candle_store.sync_all(binance)
            candles_1d = hist["1d"]
            candles_4h = hist["4h"]
            candles_1h = hist["1h"]
        except BinanceClientError as exc:
            notifier.notify_error("job_trade.market_data", str(exc))
            return

        last = candles_1h[-1] if candles_1h else {}
        high = float(last.get("high", price))
        low = float(last.get("low", price))
        events = paper.on_price(price, high=high, low=low)

        for ev in events:
            if ev.get("event") == "filled":
                notifier.send(
                    "PAPER — ĐÃ KHỚP ENTRY\n"
                    f"{ev.get('action')} khối lượng={ev.get('quantity'):.6f}\n"
                    f"Vào={ev.get('entry')} | SL={ev.get('sl')} | TP={ev.get('tp')}\n"
                    f"Margin={config.PAPER_MARGIN_PER_TRADE}u x{config.LEVERAGE}\n"
                    f"{_paper_status_text(paper)}"
                )
            elif ev.get("event") == "closed":
                try:
                    handle_paper_closed(memory, llm, notifier, ev, binance)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("paper post-mortem failed")
                    notifier.notify_error("paper_post_mortem", str(exc))
                    notifier.send(
                        f"PAPER ĐÓNG [{ev.get('reason')}] PnL={ev.get('pnl')} "
                        f"số dư={ev.get('balance')}"
                    )

        if paper.has_exposure():
            pos = paper.get_open_position()
            pending = paper.summary().get("pending")
            if not config.PAPER_MANAGE_ENABLED:
                if pos:
                    entry = float(pos["entry"])
                    qty = float(pos["quantity"])
                    if pos["action"] == "BUY":
                        upnl = qty * (price - entry)
                    else:
                        upnl = qty * (entry - price)
                    notifier.send(
                        f"PAPER ĐANG MỞ {pos['action']} @ {entry}\n"
                        f"Giá mark={price:.2f} | lãi/lỗ tạm={upnl:+.4f}u\n"
                        f"SL={pos['stop_loss']} | TP={pos['take_profit']}\n"
                        f"{_paper_status_text(paper)}\n"
                        "— bỏ qua lệnh mới (manage tắt)"
                    )
                elif pending:
                    notifier.send(
                        f"PAPER CHỜ KHỚP {pending['action']} entry={pending['entry']}\n"
                        f"Giá mark={price:.2f}\n{_paper_status_text(paper)}\n"
                        "— bỏ qua lệnh mới (manage tắt)"
                    )
                return

            manage = llm.get_manage_decision(
                candles_4h=candles_4h,
                candles_1h=candles_1h,
                candles_1d=candles_1d,
                current_price=price,
                position=pos,
                pending=pending if not pos else None,
                manage_min_score=config.MANAGE_MIN_SCORE,
            )
            manage, reject = validate_manage_hard(
                manage,
                position=pos,
                pending=pending if not pos else None,
                mark=price,
            )
            extra = (
                f"manage_score≥{config.MANAGE_MIN_SCORE} | "
                f"primary={manage.get('primary_bias')} | "
                f"{_paper_status_text(paper)}"
            )
            if reject:
                extra += f" | bị chặn: {reject}"
            state["last_manage"] = manage
            _save_state(state)
            try:
                sum_fn = (
                    None
                    if manage.get("llm_error") == "rate_limit"
                    else llm.summarize_decision_for_history
                )
                decision_history.append(manage, price, summarize_fn=sum_fn)
            except Exception as exc:  # noqa: BLE001
                logger.warning("decision_history manage append failed: %s", exc)
            notifier.notify_manage(manage, mark=price, extra=extra)

            action = str(manage.get("manage_action") or "HOLD").upper()
            if action == "CLOSE" and pos:
                closed = paper.close_now(price, reason="MANAGE_CLOSE")
                if closed.get("ok"):
                    try:
                        handle_paper_closed(memory, llm, notifier, closed, binance)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("manage close post-mortem failed")
                        notifier.notify_error("paper_manage_close", str(exc))
            elif action == "CANCEL_PENDING" and pending and not pos:
                cancelled = paper.cancel_pending()
                if cancelled.get("ok"):
                    notifier.send(
                        "PAPER — HỦY LỆNH CHỜ (manage)\n"
                        f"{cancelled.get('pending', {}).get('action')} "
                        f"entry={cancelled.get('pending', {}).get('entry')}\n"
                        f"{_paper_status_text(paper)}"
                    )
            elif action == "TRAIL" and pos:
                new_sl = float(manage["new_stop_loss"])
                trailed = paper.update_stop_loss(new_sl)
                if trailed.get("ok"):
                    notifier.send(
                        "PAPER — TRAIL SL\n"
                        f"{pos.get('action')} SL {trailed.get('old_sl')} → "
                        f"{trailed.get('new_sl')} (locked_r="
                        f"{manage.get('locked_r', '?')})\n"
                        f"{_paper_status_text(paper)}"
                    )
            logger.info(
                "=== Paper manage done action=%s thesis=%s ===",
                action,
                manage.get("thesis_status"),
            )
            return

        balance = paper.balance()
        if balance < config.PAPER_MARGIN_PER_TRADE:
            notifier.send(
                f"PAPER DỪNG: số dư {balance:.2f}u < margin "
                f"{config.PAPER_MARGIN_PER_TRADE}u — không vào lệnh mới\n"
                f"{_paper_status_text(paper)}"
            )
            return

        min_score = effective_min_score(memory)
        hist_txt = decision_history.format_for_prompt(
            decision_history.load_recent(config.DECISION_HISTORY_MAX)
        )
        decision = llm.get_decision(
            candles_4h=candles_4h,
            candles_1h=candles_1h,
            candles_1d=candles_1d,
            current_price=price,
            current_position=None,
            memory_lessons=memory.get_all_lessons_summary(),
            min_score_required=min_score,
            decision_history=hist_txt,
        )
        decision, reject = validate_decision_hard(
            decision,
            min_score,
            binance,
            primary_bias=str(decision.get("primary_bias") or ""),
        )
        extra = (
            f"điểm tối thiểu={min_score} | primary={decision.get('primary_bias')} | "
            f"bars 1D={len(candles_1d)} 4H={len(candles_4h)} 1H={len(candles_1h)} | "
            f"{_paper_status_text(paper)}"
        )
        if reject:
            extra += f" | bị chặn: {reject}"
        state["last_decision"] = decision
        _save_state(state)
        try:
            sum_fn = (
                None
                if decision.get("llm_error") == "rate_limit"
                else llm.summarize_decision_for_history
            )
            decision_history.append(decision, price, summarize_fn=sum_fn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("decision_history append failed: %s", exc)
        notifier.notify_decision(decision, balance=balance, extra=extra)

        action = decision.get("action")
        if action in {"BUY", "SELL"}:
            placed = paper.place_signal(
                action=action,
                entry=float(decision["entry_price"]),
                stop_loss=float(decision["stop_loss_price"]),
                take_profit=float(decision["take_profit_price"]),
                reasoning=str(decision.get("reasoning", "")),
                regime=str(decision.get("market_regime", "")),
                order_type="MARKET",
                mark_price=price,
                signal_level=decision.get("signal_level"),
                win_probability=decision.get("win_probability"),
                trigger=str(decision.get("trigger") or ""),
                invalidation=str(decision.get("invalidation") or ""),
            )
            if not placed.get("ok"):
                notifier.send(f"PAPER từ chối: {placed.get('reason')}")
            else:
                for ev in placed.get("events") or []:
                    if ev.get("event") == "filled":
                        notifier.send(
                            "PAPER — KHỚP NGAY\n"
                            f"{action} @ {decision['entry_price']}\n"
                            f"SL={decision['stop_loss_price']} | "
                            f"TP={decision['take_profit_price']}\n"
                            f"Margin={config.PAPER_MARGIN_PER_TRADE}u\n"
                            f"{_paper_status_text(paper)}"
                        )
                if not placed.get("filled_immediately"):
                    notifier.send(
                        "PAPER — CHỜ GIÁ KHỚP ENTRY\n"
                        f"{action} entry={decision['entry_price']}\n"
                        f"SL={decision['stop_loss_price']} | "
                        f"TP={decision['take_profit_price']}\n"
                        f"Margin={config.PAPER_MARGIN_PER_TRADE}u x{config.LEVERAGE}\n"
                        f"{_paper_status_text(paper)}"
                    )
        logger.info("=== Paper trade job done action=%s ===", decision.get("action"))
        return

    # --- LIVE MODE (Binance thật) ---
    try:
        position = binance.get_position()
    except BinanceClientError as exc:
        notifier.notify_error("job_trade.get_position", str(exc))
        return

    had_position = bool(state.get("had_position"))
    if had_position and position is None:
        logger.info("Position closed detected — running post-mortem")
        try:
            handle_closed_position(binance, memory, llm, notifier, state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("post-mortem failed")
            notifier.notify_error("post_mortem", str(exc))
        state["had_position"] = False
        state["position_snapshot"] = None
        _save_state(state)

    if position is not None:
        state["had_position"] = True
        state["position_snapshot"] = {
            "side": position["side"],
            "amount": position["amount"],
            "entry_price": position["entry_price"],
            "unrealized_pnl": position["unrealized_pnl"],
        }
        _save_state(state)
        notifier.send(
            f"Đang giữ {position['side']} khối lượng={position['amount']} "
            f"lãi/lỗ tạm={position['unrealized_pnl']:.4f} — bỏ qua lệnh mới"
        )
        logger.info("Open position exists — skip entry")
        return

    try:
        hist = candle_store.sync_all(binance)
        candles_1d = hist["1d"]
        candles_4h = hist["4h"]
        candles_1h = hist["1h"]
        price = binance.get_mark_price()
        balance = binance.get_balance_usdt()
    except BinanceClientError as exc:
        notifier.notify_error("job_trade.market_data", str(exc))
        return

    min_score = effective_min_score(memory)
    lessons = memory.get_all_lessons_summary()
    hist_txt = decision_history.format_for_prompt(
        decision_history.load_recent(config.DECISION_HISTORY_MAX)
    )
    decision = llm.get_decision(
        candles_4h=candles_4h,
        candles_1h=candles_1h,
        candles_1d=candles_1d,
        current_price=price,
        current_position=None,
        memory_lessons=lessons,
        min_score_required=min_score,
        decision_history=hist_txt,
    )
    decision, reject = validate_decision_hard(
        decision,
        min_score,
        binance,
        primary_bias=str(decision.get("primary_bias") or ""),
    )
    extra = (
        f"điểm tối thiểu={min_score} | primary={decision.get('primary_bias')} | "
        f"bars 1D={len(candles_1d)} 4H={len(candles_4h)} 1H={len(candles_1h)}"
    )
    if reject:
        extra += f" | bị chặn: {reject}"

    state["last_decision"] = decision
    _save_state(state)
    try:
        sum_fn = (
            None
            if decision.get("llm_error") == "rate_limit"
            else llm.summarize_decision_for_history
        )
        decision_history.append(decision, price, summarize_fn=sum_fn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("decision_history append failed: %s", exc)
    notifier.notify_decision(decision, balance=balance, extra=extra)

    action = decision.get("action")
    if action in {"BUY", "SELL"}:
        try:
            result = binance.open_bracket(
                action=action,
                entry_price=float(decision["entry_price"]),
                stop_loss_price=float(decision["stop_loss_price"]),
                take_profit_price=float(decision["take_profit_price"]),
                balance=balance,
            )
            state["had_position"] = True
            state["position_snapshot"] = {
                "side": action,
                "amount": result.get("quantity"),
                "entry_price": decision["entry_price"],
                "unrealized_pnl": 0.0,
            }
            _save_state(state)
            notifier.send(
                f"ĐÃ ĐẶT LỆNH {action} khối lượng={result.get('quantity')} "
                f"SL={result.get('stop_loss')} | TP={result.get('take_profit')}"
            )
        except BinanceClientError as exc:
            logger.exception("open_bracket failed")
            notifier.notify_error("open_bracket", str(exc))

    logger.info("=== Trade job done action=%s ===", decision.get("action"))


def job_learn(notifier: Notifier) -> None:
    try:
        msg = run_learner_crawler()
        logger.info(msg)
        notifier.send(f"Learner: {msg}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("learner job failed")
        notifier.notify_error("learner", str(exc))


def job_news(notifier: Notifier) -> None:
    try:
        msg = run_news_crawler()
        logger.info(msg)
        notifier.send(f"Tin tức: {msg}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("news job failed")
        notifier.notify_error("news", str(exc))


def job_compact(memory: MemoryManager, notifier: Notifier) -> None:
    """Hourly probe; only compact at Sunday 00:00–00:59 UTC."""
    now = datetime.now(timezone.utc)
    if now.weekday() != 6 or now.hour != 0:
        return
    state = _load_state()
    stamp = now.strftime("%Y-%W")
    if state.get("last_compact_week") == stamp:
        return
    try:
        msg = memory.compact_memory()
        logger.info(msg)
        notifier.send(f"Memory: {msg}")
        state["last_compact_week"] = stamp
        _save_state(state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compact failed")
        notifier.notify_error("compact", str(exc))


def safe_trade_job(
    binance: BinanceClient,
    memory: MemoryManager,
    llm: LLMAgent,
    notifier: Notifier,
    paper: PaperPortfolio | None = None,
) -> None:
    try:
        job_trade(binance, memory, llm, notifier, paper=paper)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled trade job error: %s\n%s", exc, traceback.format_exc())
        notifier.notify_error("job_trade", f"{exc}\n{traceback.format_exc()[-800:]}")


def main() -> None:
    logger.info("Starting BTC Dow/PA Mono-Agent")
    # Restore decision history + paper account from GitHub if local empty
    # (Blitz rebuild wipes generated/ — keep balance/days across rebuilds)
    try:
        restored = decision_history.ensure_restored()
        if restored:
            logger.info("Restored %d decision_history rows from remote", restored)
    except Exception as exc:  # noqa: BLE001
        logger.warning("decision_history restore skipped: %s", exc)
    try:
        if config.PAPER_MODE and paper_portfolio_mod.ensure_restored():
            logger.info("Restored paper_account from remote")
    except Exception as exc:  # noqa: BLE001
        logger.warning("paper_account restore skipped: %s", exc)
    memory = MemoryManager()
    llm = LLMAgent()
    notifier = Notifier()
    paper = PaperPortfolio() if config.PAPER_MODE else None

    try:
        binance = BinanceClient()
    except BinanceClientError as exc:
        logger.error("Cannot init Binance client: %s", exc)
        notifier.notify_error("startup", str(exc))
        raise SystemExit(1) from exc

    # Schedule jobs
    schedule.every(config.DECISION_INTERVAL_MINUTES).minutes.do(
        safe_trade_job, binance, memory, llm, notifier, paper
    )
    schedule.every(24).hours.do(job_learn, notifier)
    schedule.every(24).hours.do(job_news, notifier)
    schedule.every().hour.do(job_compact, memory, notifier)

    start_msg = (
        f"Bot khởi động lúc {datetime.now(timezone.utc).isoformat()} UTC\n"
        f"Cặp={config.SYMBOL} | Đòn bẩy={config.LEVERAGE}x | "
        f"Chu kỳ quyết định={config.DECISION_INTERVAL_MINUTES} phút\n"
    )
    if config.PAPER_MODE and paper:
        s = paper.summary()
        start_msg += (
            "Chế độ=PAPER (không đặt lệnh Binance)\n"
            f"Vốn đầu={s['initial_balance']:.0f}u | Margin/lệnh={config.PAPER_MARGIN_PER_TRADE:.0f}u\n"
            f"Số dư hiện tại={s['balance']:.2f}u | PnL={s['pnl_total']:+.2f}u\n"
            f"Ngày chạy={s['days_elapsed']:.1f}/30 — theo dõi số dư sau 1 tháng"
        )
    else:
        start_msg += "Chế độ=LIVE"
    notifier.send(start_msg)

    # Run once immediately on startup
    job_learn(notifier)
    job_news(notifier)
    safe_trade_job(binance, memory, llm, notifier, paper)

    logger.info("Entering schedule loop")
    while True:
        try:
            schedule.run_pending()
        except Exception as exc:  # noqa: BLE001
            logger.error("schedule loop error: %s", exc)
            notifier.notify_error("schedule_loop", str(exc))
        time.sleep(1)


if __name__ == "__main__":
    main()
