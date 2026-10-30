"""
1-month (configurable) historical replay — no lookahead.

Runs the same paper decision / path-exit / post-mortem / meta loop as live,
but clock and mark come from historical klines. Designed for Blitz one-shot
via REPLAY_ON_START (skip if done marker exists unless REPLAY_FORCE).
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

import config
from modules import adaptive_rules
from modules import hypothesis_store
from modules import decision_history
from modules import trade_analytics
from modules.binance_client import BinanceClient, BinanceClientError
from modules.llm_agent import LLMAgent
from modules.memory_manager import MemoryManager
from modules.notifier import Notifier
from modules.paper_portfolio import PaperPortfolio, set_paper_clock

logger = logging.getLogger(__name__)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _slice_asof(candles: list[dict[str, Any]], asof_ms: int) -> list[dict[str, Any]]:
    """Candles whose open_time <= asof (closed or current bar open)."""
    return [c for c in candles if int(c["open_time"]) <= asof_ms]


def _path_1m_since(
    candles_1m: list[dict[str, Any]],
    filled_at_iso: str,
    asof_ms: int,
) -> list[dict[str, Any]]:
    try:
        t0 = datetime.fromisoformat(str(filled_at_iso).replace("Z", "+00:00"))
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        start_ms = _ms(t0)
    except (TypeError, ValueError):
        start_ms = 0
    out = []
    for c in candles_1m:
        ot = int(c["open_time"])
        if start_ms <= ot <= asof_ms:
            out.append(c)
    return out


def _download_range(
    binance: BinanceClient,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    max_bars: int | None = None,
) -> list[dict[str, Any]]:
    # More pages for 1m
    pages = 80 if interval == "1m" else 20
    return binance.get_klines_range(
        interval,
        start_time=start_ms,
        end_time=end_ms,
        max_bars=max_bars,
        max_pages=pages,
        page_sleep_sec=0.08,
    )


def replay_already_done() -> bool:
    """True if local or remote done marker exists for same REPLAY_DAYS window."""
    if config.REPLAY_FORCE:
        return False
    local = Path(config.PATHS["REPLAY_DONE"])
    if local.exists():
        try:
            data = json.loads(local.read_text(encoding="utf-8"))
            if int(data.get("replay_days") or 0) == config.REPLAY_DAYS:
                return True
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    # Remote marker on history-data
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return False
    try:
        from modules.decision_history import _github_contents_url

        headers = {
            "Authorization": f"token {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "btc-dow-pa-agent-replay",
        }
        repo = config.HISTORY_GITHUB_REPO
        path = config.REPLAY_GITHUB_PATH
        branch = config.HISTORY_GITHUB_BRANCH or "history-data"
        resp = requests.get(
            _github_contents_url(repo, path, branch), headers=headers, timeout=30
        )
        if resp.status_code != 200:
            return False
        raw = base64.b64decode(resp.json().get("content") or "").decode("utf-8")
        data = json.loads(raw)
        return bool(data.get("done")) and int(data.get("replay_days") or 0) == config.REPLAY_DAYS
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay remote done check failed: %s", exc)
        return False


def _backup_replay_summary(summary: dict[str, Any]) -> bool:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return False
    try:
        from modules.decision_history import (
            _ensure_history_branch,
            _github_contents_url,
        )

        headers = {
            "Authorization": f"token {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "btc-dow-pa-agent-replay",
        }
        repo = config.HISTORY_GITHUB_REPO
        path = config.REPLAY_GITHUB_PATH
        branch = config.HISTORY_GITHUB_BRANCH or "history-data"
        if not _ensure_history_branch(headers, repo, branch):
            return False
        content_b64 = base64.b64encode(
            json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("ascii")
        meta = requests.get(
            _github_contents_url(repo, path, branch), headers=headers, timeout=30
        )
        sha = meta.json().get("sha") if meta.status_code == 200 else None
        body: dict[str, Any] = {
            "message": "chore: sync replay_summary (1m backtest seed)",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        put = requests.put(
            _github_contents_url(repo, path),
            headers=headers,
            json=body,
            timeout=60,
        )
        return put.status_code in {200, 201}
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay summary backup failed: %s", exc)
        return False


def _merge_lessons(src: MemoryManager, dst: MemoryManager) -> int:
    """Copy lessons from replay memory into live memory (additive)."""
    src_data = src.load()
    n = 0
    for regime, lessons in (src_data.get("lessons_by_regime") or {}).items():
        for item in lessons or []:
            dst.add_lesson(
                regime=regime,
                mistake_or_insight=str(item.get("mistake_or_insight") or ""),
                actionable_rule=str(item.get("actionable_rule") or ""),
                weight=int(item.get("weight") or 1),
                classification=str(item.get("classification") or ""),
                setup_tag=str(item.get("setup_tag") or ""),
                close_reason=str(item.get("close_reason") or ""),
                pnl=item.get("last_pnl"),
                r_multiple=item.get("last_r"),
            )
            n += 1
    return n


def _seed_rules(replay_rules_path: Path) -> int:
    if not replay_rules_path.exists():
        return 0
    try:
        data = json.loads(replay_rules_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    live = adaptive_rules.load_rules()
    incoming = adaptive_rules.validate_and_normalize(
        data,
        losing_tags=None,
        closes_now=int(live.get("closes_at_review") or 0),
    )
    merged = adaptive_rules.merge_rules(list(live.get("rules") or []), incoming)
    live["rules"] = merged
    live["updated_at"] = datetime.now(timezone.utc).isoformat()
    live["from_replay"] = True
    adaptive_rules.set_remote_backup(True)
    adaptive_rules.set_rules_path(None)
    adaptive_rules.save_rules(live)
    return len(merged)


def run_replay(
    binance: BinanceClient,
    llm: LLMAgent,
    notifier: Notifier | None = None,
    *,
    days: int | None = None,
) -> dict[str, Any]:
    """
    Replay paper bot from (now - days) → now.
    Returns summary dict; seeds memory/rules when configured.
    """
    from main import (
        effective_min_score,
        handle_paper_closed,
        maybe_run_meta_review,
        validate_decision_hard,
        validate_manage_hard,
    )

    days = int(days if days is not None else config.REPLAY_DAYS)
    step_h = int(config.REPLAY_STEP_HOURS)
    notifier = notifier or Notifier()

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    # Warmup lookback for structure
    warm_start = start - timedelta(days=45)
    start_ms, end_ms, warm_ms = _ms(start), _ms(end), _ms(warm_start)

    replay_dir = Path(config.REPLAY_DIR)
    if replay_dir.exists():
        shutil.rmtree(replay_dir, ignore_errors=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    paper_path = replay_dir / "paper_account.json"
    memory_path = replay_dir / "memory.json"
    rules_path = replay_dir / "adaptive_rules.json"
    hyp_path = replay_dir / "hypotheses.json"
    hist_path = replay_dir / "decision_history.jsonl"

    # Isolate artifacts from live GitHub paper/memory
    paper = PaperPortfolio(str(paper_path), enable_remote_backup=False)
    # Force fresh 1000u
    paper._write(
        {
            **paper._default(),
            "started_at": start.isoformat(),
            "balance": config.PAPER_BALANCE_USDT,
            "initial_balance": config.PAPER_BALANCE_USDT,
            "closed_trades": [],
            "pending": None,
            "position": None,
        },
        backup=False,
    )
    memory = MemoryManager(str(memory_path), enable_remote_backup=False)
    adaptive_rules.set_remote_backup(False)
    adaptive_rules.set_rules_path(rules_path)
    adaptive_rules.save_rules(dict(adaptive_rules.DEFAULT_RULES))
    hypothesis_store.set_remote_backup(False)
    hypothesis_store.set_path(hyp_path)
    hypothesis_store.save(dict(hypothesis_store.DEFAULT_STORE), backup=False)

    # Point decision_history local path for append during replay
    old_hist = config.PATHS.get("DECISION_HISTORY")
    config.PATHS["DECISION_HISTORY"] = str(hist_path)
    hist_path.write_text("", encoding="utf-8")

    notifier.send(
        f"REPLAY START — {days} ngày (step={step_h}h)\n"
        f"Từ {start.isoformat()} → {end.isoformat()} UTC\n"
        f"Vốn={config.PAPER_BALANCE_USDT:.0f}u | manage_llm="
        f"{'ON' if config.REPLAY_MANAGE_ENABLED else 'OFF (path SL/TP only)'}"
    )

    logger.info("Replay downloading klines warm=%s start=%s end=%s", warm_start, start, end)
    try:
        candles_1d = _download_range(binance, "1d", warm_ms, end_ms)
        candles_4h = _download_range(binance, "4h", warm_ms, end_ms)
        candles_1h = _download_range(binance, "1h", warm_ms, end_ms)
        candles_1m = _download_range(binance, "1m", start_ms, end_ms)
    except BinanceClientError as exc:
        notifier.notify_error("replay.download", str(exc))
        raise

    logger.info(
        "Replay bars 1D=%d 4H=%d 1H=%d 1m=%d",
        len(candles_1d),
        len(candles_4h),
        len(candles_1h),
        len(candles_1m),
    )

    # Decision times: each closed 1H bar open_time + 1h (= close) within [start, end]
    decision_times: list[int] = []
    for c in candles_1h:
        close_ms = int(c["open_time"]) + 3_600_000
        if start_ms <= close_ms <= end_ms:
            decision_times.append(close_ms)
    # Step subsample
    if step_h > 1:
        decision_times = decision_times[::step_h]

    n_steps = len(decision_times)
    logger.info("Replay steps=%d", n_steps)
    decisions = 0
    manages = 0
    errors = 0

    # Fake binance mark via closure — validate_* only needs get_mark_price + validate_stop_loss
    class _ReplayBinance:
        paper_mode = True

        def get_mark_price(self) -> float:
            return float(getattr(self, "_mark", 0) or 0)

        def validate_stop_loss(self, action, entry, sl):
            return binance.validate_stop_loss(action, entry, sl)

    rb = _ReplayBinance()
    # Shared mutable clock for paper timestamps (filled_at / closed_at)
    clock_holder: dict[str, str] = {"iso": start.isoformat()}
    set_paper_clock(lambda: clock_holder["iso"])

    for i, asof_ms in enumerate(decision_times):
        asof = _dt(asof_ms)
        clock_holder["iso"] = asof.isoformat()
        # Use last closed 1H close as mark (bar that just closed)
        bar_open = asof_ms - 3_600_000
        mark = None
        bar_high = None
        bar_low = None
        for c in candles_1h:
            if int(c["open_time"]) == bar_open:
                mark = float(c["close"])
                bar_high = float(c["high"])
                bar_low = float(c["low"])
                break
        if mark is None:
            # fallback nearest <= asof
            prior = [c for c in candles_1h if int(c["open_time"]) <= bar_open]
            if not prior:
                continue
            mark = float(prior[-1]["close"])
            bar_high = float(prior[-1]["high"])
            bar_low = float(prior[-1]["low"])
        rb._mark = mark

        c1d = _slice_asof(candles_1d, asof_ms)
        c4h = _slice_asof(candles_4h, asof_ms)
        c1h = _slice_asof(candles_1h, asof_ms)
        if len(c1h) < 24 or len(c4h) < 24:
            continue

        # Path exits first
        open_pos = paper.get_open_position()
        path_1m: list[dict] = []
        if open_pos and open_pos.get("filled_at"):
            path_1m = _path_1m_since(
                candles_1m, str(open_pos.get("filled_at")), asof_ms
            )
        try:
            events = paper.on_price(
                mark,
                high=bar_high or mark,
                low=bar_low or mark,
                path_candles=path_1m or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("replay on_price failed @%s: %s", asof, exc)
            errors += 1
            continue

        for ev in events:
            if ev.get("event") == "closed":
                try:
                    handle_paper_closed(
                        memory,
                        llm,
                        notifier,
                        ev,
                        rb,  # type: ignore[arg-type]
                        paper=paper,
                        post_trade_candles=path_1m[-48:] if path_1m else [],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("replay post-mortem failed: %s", exc)
                    errors += 1

        # Manage or new decision
        try:
            if paper.has_exposure() and config.REPLAY_MANAGE_ENABLED:
                pos = paper.get_open_position()
                pending = paper.summary().get("pending")
                closed = paper.closed_trades()
                manage = llm.get_manage_decision(
                    candles_4h=c4h,
                    candles_1h=c1h,
                    candles_1d=c1d,
                    current_price=mark,
                    position=pos,
                    pending=pending if not pos else None,
                    manage_min_score=config.MANAGE_MIN_SCORE,
                    trade_stats=trade_analytics.format_stats_for_prompt(closed),
                    adaptive_rules=adaptive_rules.format_for_prompt(),
                )
                manage, _rej = validate_manage_hard(
                    manage,
                    position=pos,
                    pending=pending if not pos else None,
                    mark=mark,
                )
                manages += 1
                action = str(manage.get("manage_action") or "HOLD").upper()
                if action == "CLOSE" and pos:
                    closed_ev = paper.close_now(mark, reason="MANAGE_CLOSE")
                    if closed_ev.get("ok"):
                        handle_paper_closed(
                            memory,
                            llm,
                            notifier,
                            closed_ev,
                            rb,  # type: ignore[arg-type]
                            paper=paper,
                            post_trade_candles=[],
                        )
                elif action == "CANCEL_PENDING" and pending and not pos:
                    paper.cancel_pending()
                elif action == "TRAIL" and pos and manage.get("new_stop_loss") is not None:
                    paper.update_stop_loss(
                        float(manage["new_stop_loss"]), mark_price=mark
                    )
                time.sleep(config.REPLAY_LLM_SLEEP_SEC)
            elif paper.has_exposure():
                # Path-only manage: skip LLM
                pass
            else:
                if paper.balance() < config.PAPER_MARGIN_PER_TRADE:
                    continue
                min_score = effective_min_score(memory)
                closed = paper.closed_trades()
                idle_h = None
                if closed and closed[-1].get("closed_at"):
                    try:
                        t_close = datetime.fromisoformat(
                            str(closed[-1]["closed_at"]).replace("Z", "+00:00")
                        )
                        if t_close.tzinfo is None:
                            t_close = t_close.replace(tzinfo=timezone.utc)
                        idle_h = (asof - t_close).total_seconds() / 3600.0
                    except (TypeError, ValueError):
                        idle_h = None
                allow_l2 = bool(
                    idle_h is not None and idle_h >= config.ZERO_TRADE_HOURS
                )
                decision = llm.get_decision(
                    candles_4h=c4h,
                    candles_1h=c1h,
                    candles_1d=c1d,
                    current_price=mark,
                    current_position=None,
                    memory_lessons=memory.get_all_lessons_summary(),
                    min_score_required=min_score,
                    decision_history=decision_history.format_for_prompt(
                        decision_history.load_recent(config.DECISION_HISTORY_MAX)
                    ),
                    trade_stats=trade_analytics.format_stats_for_prompt(closed),
                    adaptive_rules=adaptive_rules.format_for_prompt(),
                )
                decision, _reject = validate_decision_hard(
                    decision,
                    min_score,
                    rb,  # type: ignore[arg-type]
                    primary_bias=str(decision.get("primary_bias") or ""),
                    closed_trades=closed,
                    allow_l2_relief=allow_l2,
                )
                decisions += 1
                try:
                    decision_history.append(decision, mark, summarize_fn=None)
                except Exception:  # noqa: BLE001
                    pass
                action = str(decision.get("action") or "HOLD").upper()
                if action in {"BUY", "SELL"}:
                    placed = paper.place_signal(
                        action=action,
                        entry=float(decision["entry_price"]),
                        stop_loss=float(decision["stop_loss_price"]),
                        take_profit=float(decision["take_profit_price"]),
                        reasoning=str(decision.get("reasoning", "")),
                        regime=str(decision.get("market_regime", "")),
                        order_type="MARKET",
                        mark_price=mark,
                        signal_level=decision.get("signal_level"),
                        win_probability=decision.get("win_probability"),
                        trigger=str(decision.get("trigger") or ""),
                        setup_tag=str(
                            decision.get("setup_tag")
                            or trade_analytics.setup_tag_from_decision(decision)
                        ),
                        hyp_ids_at_entry=list(decision.get("hyp_ids_applied") or []),
                    )
                time.sleep(config.REPLAY_LLM_SLEEP_SEC)
        except Exception as exc:  # noqa: BLE001
            logger.exception("replay step failed @%s: %s", asof, exc)
            errors += 1
            time.sleep(max(2.0, config.REPLAY_LLM_SLEEP_SEC))

        if (i + 1) % max(1, config.REPLAY_PROGRESS_EVERY) == 0 or i == n_steps - 1:
            s = paper.summary()
            notifier.send(
                f"REPLAY progress {i+1}/{n_steps} @ {asof.strftime('%Y-%m-%d %H:%M')}Z\n"
                f"bal={s['balance']:.2f}u PnL={s['pnl_total']:+.2f} "
                f"trades={s['total_trades']} W{s['wins']}/L{s['losses']}\n"
                f"llm_decisions={decisions} manages={manages} errors={errors}"
            )

    # Final flatten open position at end mark
    final_mark = float(candles_1h[-1]["close"]) if candles_1h else 0.0
    clock_holder["iso"] = end.isoformat()
    if paper.get_open_position() and final_mark > 0:
        closed_ev = paper.close_now(final_mark, reason="REPLAY_END")
        if closed_ev.get("ok"):
            try:
                handle_paper_closed(
                    memory,
                    llm,
                    notifier,
                    closed_ev,
                    rb,  # type: ignore[arg-type]
                    paper=paper,
                    post_trade_candles=[],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("replay end close post-mortem: %s", exc)
    if paper.summary().get("pending"):
        paper.cancel_pending()
    set_paper_clock(None)

    # One last meta if enough trades
    try:
        maybe_run_meta_review(memory, llm, notifier, paper)
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay final meta failed: %s", exc)

    closed = paper.closed_trades()
    stats = trade_analytics.summarize_trades(closed)
    summary = {
        "done": True,
        "replay_days": days,
        "step_hours": step_h,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "steps": n_steps,
        "llm_decisions": decisions,
        "manages": manages,
        "errors": errors,
        "balance": paper.summary()["balance"],
        "pnl": paper.summary()["pnl_total"],
        "trades": stats.get("n"),
        "wins": stats.get("wins"),
        "losses": stats.get("losses"),
        "expectancy": stats.get("expectancy"),
        "by_setup": stats.get("by_setup"),
        "manage_llm": config.REPLAY_MANAGE_ENABLED,
    }
    Path(config.PATHS["REPLAY_SUMMARY"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config.PATHS["REPLAY_SUMMARY"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(config.PATHS["REPLAY_DONE"]).write_text(
        json.dumps(
            {
                "done": True,
                "replay_days": days,
                "finished_at": summary["finished_at"],
                "pnl": summary["pnl"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _backup_replay_summary(summary)

    # Seed live memory / rules / hypotheses
    seeded_lessons = 0
    seeded_rules = 0
    seeded_hyps = 0
    adaptive_rules.set_remote_backup(True)
    hypothesis_store.set_remote_backup(True)
    if config.REPLAY_SEED_MEMORY:
        live_mem = MemoryManager(enable_remote_backup=True)
        seeded_lessons = _merge_lessons(memory, live_mem)
        live_mem.resync_stats_from_trades(closed)
    if config.REPLAY_SEED_RULES:
        seeded_rules = _seed_rules(rules_path)
    # Merge replay hyps into live store
    try:
        replay_hyps = list(hypothesis_store.load().get("hypotheses") or [])
        hypothesis_store.set_path(None)
        live = hypothesis_store.load()
        by_key = {
            f"{h.get('when_setup')}|{h.get('effect')}": h
            for h in (live.get("hypotheses") or [])
            if str(h.get("status")) == "active"
        }
        for h in replay_hyps:
            if str(h.get("status")) != "active":
                continue
            key = f"{h.get('when_setup')}|{h.get('effect')}"
            if key in by_key:
                cur = by_key[key]
                cur["evidence_n"] = max(
                    int(cur.get("evidence_n") or 1), int(h.get("evidence_n") or 1)
                )
                cur["magnitude"] = max(
                    float(cur.get("magnitude") or 1), float(h.get("magnitude") or 1)
                )
            else:
                live.setdefault("hypotheses", []).append(h)
                seeded_hyps += 1
        hypothesis_store.save(live)
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay hyp seed failed: %s", exc)
        hypothesis_store.set_path(None)

    # Restore decision history path
    if old_hist:
        config.PATHS["DECISION_HISTORY"] = old_hist
    adaptive_rules.set_rules_path(None)
    hypothesis_store.set_path(None)

    notifier.send(
        "REPLAY DONE\n"
        f"PnL={summary['pnl']:+.2f}u bal={summary['balance']:.2f}u "
        f"trades={summary['trades']} exp={summary['expectancy']:+.3f}\n"
        f"seeded lessons={seeded_lessons} rules={seeded_rules} hyps={seeded_hyps}\n"
        f"errors={errors} — live bot continues with seeded memory"
    )
    logger.info("Replay done summary=%s", summary)
    return summary


def maybe_run_replay_on_start(
    binance: BinanceClient,
    llm: LLMAgent,
    notifier: Notifier,
) -> dict[str, Any] | None:
    if not config.REPLAY_ON_START:
        return None
    if replay_already_done():
        logger.info("Replay skipped — done marker present (set REPLAY_FORCE=true to rerun)")
        notifier.send(
            f"REPLAY skip — đã có marker {config.REPLAY_DAYS}d "
            "(REPLAY_FORCE=true để chạy lại)"
        )
        return None
    return run_replay(binance, llm, notifier)
