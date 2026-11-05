"""
Champion Research Agent — live Binance data, PAPER fills only.

Flow:
  LLM propose → evolution/backtest/validation → Champion promote
  → paper on closed live candles → post-mortem / decay → re-research
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import schedule

import config
from modules import champion_backup
from modules import champion_store
from modules import strategy_decay
from modules.archive_state import archive_learn_state
from modules.learner_crawler import run_learner_crawler
from modules.llm_researcher import post_mortem_paper_trade
from modules.memory_manager import MemoryManager
from modules.news_crawler import run_news_crawler
from modules.notifier import Notifier
from modules.paper_champion import PaperChampion
from modules.quant_data import (
    enrich_derivatives_data,
    enrich_multi_timeframe,
    fetch_klines,
)
from modules.research_pipeline import run_research
from modules import tick_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _load_runtime() -> dict[str, Any]:
    path = Path(config.PATHS["STATE"])
    if not path.exists():
        return {"last_research_ts": 0.0, "last_compact_week": None}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_research_ts": 0.0, "last_compact_week": None}


def _save_runtime(state: dict[str, Any]) -> None:
    import json

    path = Path(config.PATHS["STATE"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def job_research(notifier: Notifier, memory: MemoryManager) -> dict | None:
    try:
        logger.info("=== Research job start ===")
        report = run_research(use_llm=config.LLM_RESEARCH_ENABLED)
        champ_spec = report.get("champion_spec") or {}
        review = report.get("ai_review") or {}
        ev = report.get("champion_evaluation") or {}
        msg = (
            "RESEARCH Spec xong (PAPER only | PnL-first)\n"
            f"Spec={champ_spec.get('name') or 'NONE'} "
            f"id={ev.get('spec_id') or report.get('champion_spec_id') or '—'}\n"
            f"mode={report.get('config', {}).get('research_mode') or '—'} "
            f"promote={report.get('promote_mode')}\n"
            f"Qualified top={sum(1 for c in report.get('top_candidates') or [] if c.get('qualified'))}\n"
            f"AI review: {(review.get('summary') or '')[:400]}"
        )
        notifier.send(msg)
        rt = _load_runtime()
        rt["last_research_ts"] = time.time()
        _save_runtime(rt)
        # Clear decay flag after research
        st = champion_store.get_paper_state()
        st["research_due"] = False
        champion_store.set_paper_state(st)
        try:
            champion_backup.backup_remote(force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("champion backup after research failed: %s", exc)
        logger.info("=== Research job done ===")
        return report
    except Exception as exc:  # noqa: BLE001
        logger.exception("research job failed")
        notifier.notify_error("job_research", str(exc))
        return None


def _handle_closed_trade(
    trade: dict,
    notifier: Notifier,
    memory: MemoryManager,
) -> None:
    st = champion_store.get_paper_state()
    genome = st.get("champion")
    pm = post_mortem_paper_trade(trade, genome=genome)
    lesson = pm.get("lesson") or ""
    if lesson:
        try:
            memory.add_lesson(
                regime="CHAMPION_PAPER",
                mistake_or_insight=lesson,
                actionable_rule=f"genome={trade.get('genome_id')} {pm.get('classification')}",
                classification=str(pm.get("classification") or ""),
                setup_tag=str(trade.get("family") or ""),
                close_reason=str(trade.get("exit_reason") or ""),
                pnl=float(trade.get("pnl") or 0),
            )
        except Exception as exc:
            logger.warning("add_lesson failed: %s", exc)

    hints = pm.get("challenger_hints") or []
    if hints:
        existing = list(st.get("challenger_hints") or [])
        existing.extend(hints)
        st["challenger_hints"] = existing[-16:]
        champion_store.set_paper_state(st)

    if pm.get("decay_signal"):
        st = champion_store.get_paper_state()
        st["research_due"] = True
        st["research_mode"] = "replace"
        champion_store.set_paper_state(st)

    # Spec patch/replace cadence by trades on current Spec (PnL learning)
    st_n = champion_store.get_paper_state()
    trades_on = int(st_n.get("trades_on_spec") or 0)
    last_sr = float(st_n.get("last_spec_research_ts") or 0)
    cooled = (time.time() - last_sr) >= config.SPEC_PATCH_COOLDOWN_HOURS * 3600
    if cooled and trades_on >= config.SPEC_REPLACE_MIN_TRADES:
        st_n["research_due"] = True
        st_n["research_mode"] = "replace"
        champion_store.set_paper_state(st_n)
    elif cooled and trades_on >= config.SPEC_PATCH_MIN_TRADES:
        st_n["research_due"] = True
        st_n["research_mode"] = "patch"
        champion_store.set_paper_state(st_n)

    decay = strategy_decay.check_decay()
    side = "LONG" if trade.get("side") == 1 else "SHORT"
    notifier.send(
        f"PAPER CLOSE {side} {trade.get('exit_reason')}\n"
        f"PnL={float(trade.get('pnl') or 0):+.4f}u | "
        f"Equity={float(trade.get('equity_after') or st.get('equity') or 0):.2f}u\n"
        f"Genome={trade.get('genome_id')} | {pm.get('classification')}\n"
        f"{(lesson or '')[:300]}\n"
        f"Decay={decay.get('decay')} {decay.get('reason') or ''}"
    )
    try:
        champion_backup.backup_remote(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("champion backup after close failed: %s", exc)


def job_paper_tick(notifier: Notifier, memory: MemoryManager) -> None:
    """One paper tick on live closed candles (idempotent by last_candle)."""
    try:
        rt = _load_runtime()
        st = champion_store.get_paper_state()
        due = bool(st.get("research_due"))
        last_ts = float(rt.get("last_research_ts") or 0)
        if due or (time.time() - last_ts >= config.RESEARCH_EVERY_HOURS * 3600):
            job_research(notifier, memory)

        # 15m: keep enough bars for indicators (~14d min); cap fetch for poll latency
        days = max(14, min(config.HISTORY_DAYS, 45 if config.TIMEFRAME_MAIN.endswith("m") else 90))
        bars = fetch_klines(config.TIMEFRAME_MAIN, days, config.SYMBOL)
        bars = enrich_derivatives_data(bars, config.SYMBOL, days)
        bars = enrich_multi_timeframe(bars, config.SYMBOL, days)

        def on_open(pos: dict) -> None:
            side = "LONG" if pos.get("side") == 1 else "SHORT"
            notifier.send(
                f"PAPER OPEN {side}\n"
                f"entry={float(pos['entry']):.2f} lev={pos.get('leverage')}x "
                f"margin={float(pos.get('margin') or 0):.2f}\n"
                f"SL={float(pos['sl']):.2f} TP={float(pos['tp']):.2f}\n"
                f"genome={pos.get('genome_id')} family={pos.get('family')}"
            )

        def on_tick(report: dict) -> None:
            # Research context (not AI entry). Keep even OPEN/CLOSE for flat-vs-active debug.
            try:
                tick_history.append(report)
            except Exception as exc:
                logger.warning("tick_history append failed: %s", exc)
            # Short per-candle analysis (15m close). OPEN/CLOSE still have dedicated msgs.
            action = str(report.get("action") or "SKIP")
            if action in {"OPEN"} or action.startswith("CLOSE_"):
                return  # avoid duplicate with on_open / on_close
            close = report.get("close")
            close_s = f"{float(close):.1f}" if close is not None else "—"
            atr = report.get("atr_pct")
            rsi = report.get("rsi")
            metrics = []
            if atr is not None:
                metrics.append(f"ATR%{float(atr)*100:.2f}")
            if rsi is not None:
                metrics.append(f"RSI{float(rsi):.0f}")
            metrics_s = " · ".join(metrics)
            candle = str(report.get("candle") or "")[-19:]
            family = report.get("family") or "—"
            why = str(report.get("why") or "")[:160]
            lines = [
                f"15m {candle}",
                f"close={close_s}" + (f" | {metrics_s}" if metrics_s else ""),
                f"{action} · {family}",
                why,
            ]
            notifier.send("\n".join(lines))

        trader = PaperChampion(
            on_close=lambda t: _handle_closed_trade(t, notifier, memory),
            on_open=on_open,
            on_tick=on_tick,
        )
        last_seen = trader.state.get("last_candle")
        closed_all: list[dict] = []
        if last_seen:
            unseen_idx = [
                i
                for i, ts in enumerate(bars.open_time.astype(str).tolist())
                if ts > str(last_seen)
            ]
            for idx in unseen_idx:
                closed_all.extend(
                    trader.process(bars.iloc[max(0, idx - 205) : idx + 1])
                )
        else:
            closed_all.extend(trader.process(bars))

        # Chronic filter SKIP → flag early research (genome change), not live soft-entry
        st2 = champion_store.get_paper_state()
        diag = tick_history.maybe_flag_research_due(st2)
        if diag and st2.get("research_due"):
            champion_store.set_paper_state(st2)
            try:
                champion_backup.backup_remote(force=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("champion backup after starve failed: %s", exc)
            if diag.get("should_reresearch"):
                notifier.send(
                    "Champion STARVING (chronic SKIP)\n"
                    f"{diag.get('summary')}\n"
                    + "\n".join((diag.get("focus") or [])[:3])
                    + "\n→ research sớm (đổi genome, không nới filter live)"
                )

        if not trader.state.get("champion"):
            logger.info("Flat: no Champion qualified yet")
    except Exception as exc:  # noqa: BLE001
        logger.error("paper tick error: %s\n%s", exc, traceback.format_exc())
        notifier.notify_error("job_paper_tick", str(exc))


def job_learn(notifier: Notifier) -> None:
    try:
        msg = run_learner_crawler()
        logger.info(msg)
        notifier.send(f"Learner (research context): {msg}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("learner job failed")
        notifier.notify_error("learner", str(exc))


def job_news(notifier: Notifier) -> None:
    try:
        msg = run_news_crawler()
        logger.info(msg)
        notifier.send(f"News (research context): {msg}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("news job failed")
        notifier.notify_error("news", str(exc))


def job_compact(memory: MemoryManager, notifier: Notifier) -> None:
    now = datetime.now(timezone.utc)
    if now.weekday() != 6 or now.hour != 0:
        return
    state = _load_runtime()
    stamp = now.strftime("%Y-%W")
    if state.get("last_compact_week") == stamp:
        return
    try:
        msg = memory.compact_memory()
        logger.info(msg)
        notifier.send(f"Memory: {msg}")
        state["last_compact_week"] = stamp
        _save_runtime(state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compact failed")
        notifier.notify_error("compact", str(exc))


def main() -> None:
    logger.info("Starting Champion Research Agent (live data, PAPER fills only)")
    if not config.PAPER_MODE or not config.FORCE_PAPER_ONLY:
        logger.warning(
            "PAPER_MODE/FORCE_PAPER_ONLY expected true — enforcing paper-only semantics"
        )

    try:
        archived = archive_learn_state()
        if archived:
            logger.info("Archived legacy learn state → %s", archived)
    except Exception as exc:  # noqa: BLE001
        logger.warning("archive_learn_state skipped: %s", exc)

    champion_store.init_db()
    try:
        restored = champion_backup.ensure_restored()
        if restored:
            logger.info("Restored Champion snapshot from GitHub")
    except Exception as exc:  # noqa: BLE001
        logger.warning("champion restore skipped: %s", exc)

    memory = MemoryManager()
    notifier = Notifier()

    start_msg = (
        f"Champion Spec Agent khởi động {datetime.now(timezone.utc).isoformat()} UTC\n"
        f"Mode=PAPER ONLY | Mission=PnL dương (accept risk in bounds)\n"
        f"Symbol={config.SYMBOL} interval={config.TIMEFRAME_MAIN}\n"
        f"Research every {config.RESEARCH_EVERY_HOURS}h | "
        f"patch≥{config.SPEC_PATCH_MIN_TRADES} replace≥{config.SPEC_REPLACE_MIN_TRADES} trades\n"
        f"Paper poll {config.PAPER_POLL_SECONDS}s\n"
        f"Capital={config.INITIAL_CAPITAL:.0f}u | "
        f"margin≤{config.MAX_MARGIN_FRACTION*100:.0f}% equity | "
        f"lev candidates={list(config.LEVERAGE_CANDIDATES)}\n"
        f"DB={config.RESEARCH_DB_PATH}"
    )
    trader = PaperChampion()
    s = trader.summary()
    start_msg += (
        f"\nEquity={s['equity']:.2f}u PnL={s['pnl_total']:+.2f}u "
        f"trades={s['trade_count']} on_spec={s.get('trades_on_spec', 0)} "
        f"spec={s['champion_family'] or 'NONE'} id={s['champion_id'] or '—'}"
    )
    notifier.send(start_msg)

    # Immediate context + research + first paper tick
    job_learn(notifier)
    job_news(notifier)
    job_research(notifier, memory)
    job_paper_tick(notifier, memory)

    schedule.every(24).hours.do(job_learn, notifier)
    schedule.every(24).hours.do(job_news, notifier)
    schedule.every().hour.do(job_compact, memory, notifier)
    schedule.every(1).hours.do(
        lambda: champion_backup.backup_remote(force=True)
    )

    logger.info(
        "Entering loop: paper tick every %ss; research every %sh or on decay",
        config.PAPER_POLL_SECONDS,
        config.RESEARCH_EVERY_HOURS,
    )
    while True:
        try:
            schedule.run_pending()
            job_paper_tick(notifier, memory)
        except Exception as exc:  # noqa: BLE001
            logger.error("main loop error: %s", exc)
            notifier.notify_error("main_loop", str(exc))
        time.sleep(config.PAPER_POLL_SECONDS)


if __name__ == "__main__":
    main()
