"""
Champion Research Agent — live Binance data, PAPER fills only.

Main loop (every PAPER_POLL_SECONDS):
  1. apply a finished background research (Champion / shadows)
  2. live watch: pending fill + SL/TP/LIQ on 1m high/low (1 light request)
  3. on a new closed 15m candle: full data fetch → Champion + shadow processing
  4. start background research when due (no Spec / flagged / stale)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import schedule

import config
from modules import champion_backup, champion_store, family, research_memory, shadow_book
from modules import strategy_decay, tick_history
from modules.archive_state import archive_learn_state
from modules.learner_crawler import run_learner_crawler
from modules.llm_researcher import post_mortem_paper_trade
from modules.memory_manager import MemoryManager
from modules.news_crawler import run_news_crawler
from modules.notifier import Notifier
from modules.paper_champion import PaperChampion, interval_td
from modules.quant_data import (
    enrich_derivatives_data,
    enrich_multi_timeframe,
    fetch_klines,
    fetch_recent_1m,
)
from modules.research_pipeline import apply_research, compute_research
from modules.strategy_spec import format_spec_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _load_runtime() -> dict[str, Any]:
    path = Path(config.PATHS["STATE"])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"last_research_ts": 0.0, "last_compact_week": None}


def _save_runtime(state: dict[str, Any]) -> None:
    path = Path(config.PATHS["STATE"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _backup(reason: str) -> None:
    try:
        champion_backup.backup_remote(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("champion backup after %s failed: %s", reason, exc)


# --- research in background --------------------------------------------------------


class ResearchWorker:
    """Runs compute_research() in a thread; the main thread applies the result."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._result: dict | None = None
        self._error: str | None = None
        self.last_fail_ts = 0.0
        self.started_ts = 0.0

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, reason: str) -> None:
        if self.running():
            return
        self._result, self._error = None, None
        self.started_ts = time.time()
        logger.info("=== Research start (background) reason=%s ===", reason)

        def _run() -> None:
            try:
                self._result = compute_research(use_llm=config.LLM_RESEARCH_ENABLED)
            except Exception:  # noqa: BLE001
                self._error = traceback.format_exc()

        self._thread = threading.Thread(target=_run, name="research", daemon=True)
        self._thread.start()

    def poll(self, notifier: Notifier) -> None:
        if self._thread is None or self._thread.is_alive():
            return
        self._thread = None
        mins = (time.time() - self.started_ts) / 60
        if self._error:
            self.last_fail_ts = time.time()
            logger.error("research failed:\n%s", self._error)
            notifier.notify_error("research", self._error.strip().splitlines()[-1][:300])
            return
        try:
            report = apply_research(self._result or {})
        except Exception:  # noqa: BLE001
            self.last_fail_ts = time.time()
            err = traceback.format_exc()
            logger.error("apply_research failed:\n%s", err)
            notifier.notify_error("apply_research", err.strip().splitlines()[-1][:300])
            return
        rt = _load_runtime()
        rt["last_research_ts"] = time.time()
        _save_runtime(rt)
        _notify_research(notifier, report, mins)
        _backup("research")
        logger.info("=== Research applied (%.1f min) ===", mins)


def _notify_research(notifier: Notifier, report: dict, minutes: float) -> None:
    champ = report.get("champion_spec") or {}
    ev = report.get("champion_evaluation") or {}
    review = report.get("ai_review") or {}
    cfg = report.get("config") or {}
    top = report.get("top_candidates") or []
    fam = report.get("family") or {}
    decision = report.get("decision") or "—"
    lines = [
        f"RESEARCH family xong ({minutes:.1f} phút, PAPER only | PnL-first)",
        f"Family={fam.get('label') or 'NONE'} id={(ev or {}).get('spec_id') or report.get('champion_spec_id') or '—'}",
        f"mode={cfg.get('research_mode') or '—'} quyết định={decision} "
        f"nguồn={report.get('promote_source') or '—'} fail_streak={fam.get('fail_streak')}",
        f"LLM ứng viên={cfg.get('llm_candidates')} | qualified top={sum(1 for c in top if c.get('qualified'))} | "
        f"folds+={(ev or {}).get('positive_folds')}/{cfg.get('wf_folds')}",
    ]
    if decision in {"genesis", "promote"} and report.get("changes"):
        lines.append("THAY ĐỔI:")
        lines += [f"  {c}" for c in report["changes"][:8]]
    if report.get("shadow_events"):
        lines += report["shadow_events"]
    if report.get("shadows"):
        lines.append("SHADOW:")
        lines += report["shadows"]
    lines.append(f"AI review: {(review.get('summary') or '')[:400]}")
    notifier.send("\n".join(lines))
    if champ:
        notifier.send(
            format_spec_telegram(
                champ,
                sid=ev.get("spec_id"),
                promote_mode=str(report.get("promote_mode") or "—"),
                research_mode=str(cfg.get("research_mode") or "—"),
                source=str(report.get("promote_source") or "evolution"),
            )
        )


def research_reason(worker: ResearchWorker) -> str | None:
    if worker.running():
        return None
    if time.time() - worker.last_fail_ts < config.RESEARCH_RETRY_MINUTES * 60:
        return None
    st = champion_store.get_paper_state()
    if family.needs_genesis(st):
        return "genesis"
    if st.get("research_due"):
        return f"due:{st.get('research_mode') or 'patch'}"
    last_ts = max(
        float(_load_runtime().get("last_research_ts") or 0),
        float(st.get("last_spec_research_ts") or 0),
    )
    if time.time() - last_ts >= config.RESEARCH_EVERY_HOURS * 3600:
        return "stale"
    return None


# --- trades ------------------------------------------------------------------------------


def _handle_closed_trade(
    trade: dict,
    trader: PaperChampion,
    notifier: Notifier,
    memory: MemoryManager,
) -> None:
    st = trader.state
    pm = post_mortem_paper_trade(
        trade,
        spec=st.get("champion_spec"),
        recent_trades=research_memory.recent_trades_for_prompt(10),
    )
    lesson = pm.get("lesson") or ""
    if lesson:
        try:
            memory.add_lesson(
                regime=str(((trade.get("entry_context") or {}).get("regime") or {}).get("tag") or "CHAMPION_PAPER"),
                mistake_or_insight=lesson,
                actionable_rule=pm.get("actionable_rule") or f"spec={trade.get('spec_id')} {pm.get('classification')}",
                classification=str(pm.get("classification") or ""),
                setup_tag=str(trade.get("family") or ""),
                close_reason=str(trade.get("exit_reason") or ""),
                pnl=float(trade.get("pnl") or 0),
                r_multiple=trade.get("r_multiple"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("add_lesson failed: %s", exc)
    if pm.get("research_hints"):
        st["research_hints"] = (list(st.get("research_hints") or []) + pm["research_hints"])[-9:]

    cooled = (
        time.time() - float(st.get("last_spec_research_ts") or 0)
        >= config.SPEC_PATCH_COOLDOWN_HOURS * 3600
    )
    decay = strategy_decay.check_decay(dict(st))
    trades_on = int(st.get("trades_on_spec") or 0)
    if cooled and (pm.get("decay_signal") or decay.get("decay")):
        st["research_due"], st["research_mode"] = True, "rework"
    elif cooled and trades_on >= config.SPEC_PATCH_MIN_TRADES:
        st["research_due"], st["research_mode"] = True, "patch"
    trader.save()

    side = "LONG" if trade.get("side") == 1 else "SHORT"
    r_mult = trade.get("r_multiple")
    notifier.send(
        f"PAPER CLOSE {side} {trade.get('exit_reason')} ({trade.get('exit_source')})\n"
        f"PnL={float(trade.get('pnl') or 0):+.4f}u"
        + (f" ({float(r_mult):+.2f}R)" if r_mult is not None else "")
        + f" | Equity={float(trade.get('equity_after') or st.get('equity') or 0):.2f}u\n"
        f"Spec={trade.get('family')} [{trade.get('spec_id')}] | {pm.get('classification')}\n"
        f"{lesson[:300]}\n"
        f"Decay={decay.get('decay')} {decay.get('reason') or ''}"
        + (f"\nresearch_due → {st.get('research_mode')}" if st.get("research_due") else "")
    )
    _backup("close")


def _on_open(notifier: Notifier):
    def cb(pos: dict) -> None:
        side = "LONG" if pos.get("side") == 1 else "SHORT"
        regime = ((pos.get("entry_context") or {}).get("regime") or {}).get("tag")
        notifier.send(
            f"PAPER OPEN {side}\n"
            f"entry={float(pos['entry']):.2f} lev={pos.get('leverage')}x "
            f"margin={float(pos.get('margin') or 0):.2f}\n"
            f"SL={float(pos['sl']):.2f} TP={float(pos['tp']):.2f}\n"
            f"spec={pos.get('family')} [{pos.get('spec_id')}] regime={regime}"
        )

    return cb


def job_live_watch(notifier: Notifier, memory: MemoryManager) -> None:
    """Between candle closes: fill pending at next-bar open, exit on SL/TP/LIQ touches."""
    try:
        trader = PaperChampion(on_open=_on_open(notifier))
        fill_ms = trader.pending_fill_ms()
        now_ms = int(time.time() * 1000)
        if fill_ms is not None and now_ms >= fill_ms:
            bars = fetch_recent_1m(start_ms=fill_ms, limit=2)
            if bars and bars[0]["open_time_ms"] == fill_ms:
                bars += [b for b in fetch_recent_1m(limit=2) if b["open_time_ms"] > bars[-1]["open_time_ms"]]
        elif trader.state.get("position") is not None:
            bars = fetch_recent_1m(limit=2)
        else:
            return
        for trade in trader.check_live(bars):
            _handle_closed_trade(trade, trader, notifier, memory)
    except Exception as exc:  # noqa: BLE001
        logger.warning("live watch error: %s", exc)


def candle_due() -> bool:
    last = champion_store.get_paper_state().get("last_candle")
    if not last:
        return True
    due = pd.Timestamp(last) + 2 * interval_td() + pd.Timedelta(seconds=config.CANDLE_CLOSE_GRACE_SECONDS)
    return pd.Timestamp.now(tz="UTC") >= due


def job_paper_tick(notifier: Notifier, memory: MemoryManager) -> None:
    """Process every closed candle not seen yet (Champion + shadows)."""
    try:
        days = max(14, min(config.HISTORY_DAYS, 45 if config.TIMEFRAME_MAIN.endswith("m") else 90))
        bars = fetch_klines(config.TIMEFRAME_MAIN, days, config.SYMBOL)
        bars = enrich_derivatives_data(bars, config.SYMBOL, days)
        bars = enrich_multi_timeframe(bars, config.SYMBOL, days)

        notify = {"on": False}

        def on_tick(report: dict) -> None:
            try:
                tick_history.append(report)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tick_history append failed: %s", exc)
            action = str(report.get("action") or "SKIP")
            if not notify["on"] or action == "OPEN" or action.startswith("CLOSE_"):
                return
            close = report.get("close")
            metrics = []
            if report.get("atr_pct") is not None:
                metrics.append(f"ATR%{float(report['atr_pct'])*100:.2f}")
            if report.get("rsi") is not None:
                metrics.append(f"RSI{float(report['rsi']):.0f}")
            if report.get("regime"):
                metrics.append(str(report["regime"]))
            notifier.send(
                "\n".join(
                    [
                        f"15m {str(report.get('candle') or '')[-25:]}",
                        f"close={float(close):.1f}" + (f" | {' · '.join(metrics)}" if metrics else "")
                        if close is not None
                        else "close=—",
                        f"{action} · {report.get('family') or '—'}",
                        str(report.get("why") or "")[:160],
                    ]
                )
            )

        trader = PaperChampion(on_open=_on_open(notifier), on_tick=on_tick)
        last_seen = trader.state.get("last_candle")
        times = bars.open_time.astype(str).tolist()
        idxs = [i for i, ts in enumerate(times) if ts > str(last_seen)] if last_seen else [len(bars) - 1]
        if len(idxs) > 1:
            logger.info("Replaying %d unseen candles", len(idxs))
        for idx in idxs:
            notify["on"] = idx == idxs[-1]
            window = bars.iloc[max(0, idx - config.PAPER_WARMUP_BARS) : idx + 1]
            closed = trader.process(window)
            shadow_book.run_on_slice(trader.state, window)
            trader.save()
            for trade in closed:
                _handle_closed_trade(trade, trader, notifier, memory)

        st = champion_store.get_paper_state()
        diag = tick_history.maybe_flag_research_due(st)
        if diag and st.get("research_due"):
            champion_store.set_paper_state(st)
            _backup("starve")
            if diag.get("should_reresearch"):
                notifier.send(
                    "Champion STARVING (chronic SKIP)\n"
                    f"{diag.get('summary')}\n"
                    + "\n".join((diag.get("focus") or [])[:3])
                    + "\n→ research sớm (đổi Spec, không nới filter live)"
                )
    except Exception as exc:  # noqa: BLE001
        logger.error("paper tick error: %s\n%s", exc, traceback.format_exc())
        notifier.notify_error("job_paper_tick", str(exc))


# --- context jobs -------------------------------------------------------------------------


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
    try:
        archived = archive_learn_state()
        if archived:
            logger.info("Archived legacy learn state → %s", archived)
    except Exception as exc:  # noqa: BLE001
        logger.warning("archive_learn_state skipped: %s", exc)

    champion_store.init_db()
    try:
        if champion_backup.ensure_restored():
            logger.info("Restored Champion snapshot from GitHub")
    except Exception as exc:  # noqa: BLE001
        logger.warning("champion restore skipped: %s", exc)

    memory = MemoryManager()
    notifier = Notifier()
    worker = ResearchWorker()

    s = PaperChampion().summary()
    shadows = shadow_book.summary_lines(champion_store.get_paper_state())
    notifier.send(
        f"Champion Spec Agent khởi động {datetime.now(timezone.utc).isoformat()} UTC\n"
        f"Mode=PAPER ONLY | Mission=PnL dương (accept risk in bounds)\n"
        f"Symbol={config.SYMBOL} interval={config.TIMEFRAME_MAIN} | poll {config.PAPER_POLL_SECONDS}s "
        f"(live SL/TP={'on' if config.LIVE_EXIT_ENABLED else 'off'})\n"
        f"Research nền: mỗi {config.RESEARCH_EVERY_HOURS:.0f}h hoặc khi cần | "
        f"patch≥{config.SPEC_PATCH_MIN_TRADES} lệnh | rework sau {config.FAMILY_REWORK_AFTER} lần yếu | "
        f"walk-forward {config.WF_FOLDS} fold\n"
        f"1 family {config.FAMILY_MIN_INDICATORS}-{config.FAMILY_MAX_INDICATORS} chỉ báo do LLM thiết kế"
        + (" — chưa có, sẽ tạo (genesis)" if family.needs_genesis(champion_store.get_paper_state()) else "")
        + "\n"
        f"Equity={s['equity']:.2f}u PnL={s['pnl_total']:+.2f}u "
        f"trades={s['trade_count']} on_version={s.get('trades_on_spec', 0)} "
        f"family={s['champion_family'] or 'NONE'} id={s['champion_id'] or '—'}"
        + ("\nSHADOW:\n" + "\n".join(shadows) if shadows else "")
    )

    if config.LEARNER_ENABLED:
        job_learn(notifier)
        schedule.every(24).hours.do(job_learn, notifier)
    if config.NEWS_ENABLED:
        job_news(notifier)
        schedule.every(24).hours.do(job_news, notifier)
    schedule.every().hour.do(job_compact, memory, notifier)
    schedule.every(1).hours.do(lambda: champion_backup.backup_remote(force=True))

    job_paper_tick(notifier, memory)
    logger.info(
        "Entering loop: poll %ss, live watch=%s, research in background",
        config.PAPER_POLL_SECONDS,
        config.LIVE_EXIT_ENABLED,
    )
    while True:
        try:
            schedule.run_pending()
            worker.poll(notifier)
            if config.LIVE_EXIT_ENABLED:
                job_live_watch(notifier, memory)
            if candle_due():
                job_paper_tick(notifier, memory)
            reason = research_reason(worker)
            if reason:
                worker.start(reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("main loop error: %s", exc)
            notifier.notify_error("main_loop", str(exc))
        time.sleep(config.PAPER_POLL_SECONDS)


if __name__ == "__main__":
    main()
