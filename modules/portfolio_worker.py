"""One continuous Futures paper service and one background research job."""

from __future__ import annotations

import logging
import math
import os
import signal
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy

import pandas as pd

import config
from modules import quant_data
from modules.notifier import Notifier
from modules.paper_champion import entry_context, interval_td
from modules.portfolio_data import MarketData, check_symbols, ms
from modules.portfolio_engine import Portfolio
from modules.portfolio_research import prepared, research
from modules.portfolio_store import PortfolioStore, writer_lock
from modules.strategy_spec import compile_signal, diagnose_spec_entry, spec_id

logger = logging.getLogger(__name__)
MINUTE = 60_000


def _due(market: dict, now: int, step: int) -> bool:
    last = market.get("last_candle")
    return last is None or now >= last + 2 * step + config.CANDLE_CLOSE_GRACE_SECONDS * 1000


def _market_events(symbol: str, state: dict, feed: MarketData, now: int, step: int) -> list[tuple]:
    market = state["markets"][symbol]
    spec = state["champion"]
    events = []
    last = market.get("last_candle")
    if _due(market, now, step):
        raw = feed.frame(symbol)
        frame = prepared(raw, spec)
        old = (market.get("position") or {}).get("spec")
        previous = prepared(raw, old) if old and spec_id(old) != spec_id(spec) else None
        if len(frame) < 203:
            raise RuntimeError(f"{symbol}: requires 203 candles for indicator warm-up")
        newest = ms(frame.index[-1])
        start = max(1, int(frame.index.searchsorted(pd.Timestamp(last, unit="ms", tz="UTC"), side="right"))) if last is not None else len(frame) - 1
        if last is not None and start < len(frame) and ms(frame.index[start]) != last + step:
            raise RuntimeError(f"{symbol}: missing main candle after {last}")
        fn = compile_signal(spec)
        old_fn = compile_signal(old) if previous is not None else fn
        for i in range(start, len(frame)):
            row = frame.iloc[i]
            stamp = ms(frame.index[i])
            current_signal = fn(row, frame.iloc[i - 1])
            opposite = old_fn(previous.iloc[i], previous.iloc[i - 1]) if previous is not None else current_signal
            context = entry_context(row, spec) if current_signal is not None else None
            atr = float(row["atr"])
            exit_atr = float(previous.iloc[i]["atr"]) if previous is not None else atr
            why = diagnose_spec_entry(row, frame.iloc[i - 1], spec)["why"].splitlines()[0] if stamp == newest else ""
            events.append((stamp + step, "main", symbol, (
                stamp, float(row.high), float(row.low), float(row.close), atr, exit_atr,
                opposite, current_signal, context, stamp == newest and now < stamp + step + MINUTE, why,
            )))

    active = market.get("position") or market.get("pending")
    needs_open = any(event[3][-2] and event[3][-4] is not None for event in events)
    if not active and not needs_open:
        return events

    cursor = market.get("last_minute")
    start = cursor + MINUTE if cursor is not None else (
        active.get("entry_time", active.get("fill_time")) if active else now // MINUTE * MINUTE
    )
    if (pos := market.get("position")) is not None:
        start = max(start, pos["entry_time"])
    if start + MINUTE <= now:
        minutes = quant_data.fetch_klines("1m", 1, symbol, start_ms=start, allow_empty=True)
        expected = start
        for row in minutes.itertuples(index=False):
            stamp = ms(row.open_time)
            if stamp != expected:
                raise RuntimeError(f"{symbol}: missing 1m candle at {expected}")
            expected += MINUTE
            events.append((stamp + MINUTE, "touch", symbol, (stamp, row.high, row.low, row.close)))
    bars = quant_data.fetch_recent_1m(symbol, limit=2)
    events.extend(
        (bar["open_time_ms"], "open", symbol, bar["open"])
        for bar in bars if bar["open_time_ms"] == now // MINUTE * MINUTE
    )
    for stamp, rate in feed.funding.get(symbol, {}).items():
        if start - MINUTE <= stamp <= now:
            events.append((stamp, "funding", symbol, rate))
    return events


def _paper_tick(state: dict, feed: MarketData, pool: ThreadPoolExecutor) -> Portfolio:
    engine = Portfolio(deepcopy(state))
    now = int(time.time() * 1000)
    step = int(interval_td().total_seconds() * 1000)
    symbols = state["symbols"]
    tasks = {
        symbol: pool.submit(_market_events, symbol, state, feed, now, step)
        for symbol in symbols
        if _due(state["markets"][symbol], now, step)
        or state["markets"][symbol].get("position") or state["markets"][symbol].get("pending")
    }
    by_time = defaultdict(lambda: defaultdict(dict))
    for symbol, task in tasks.items():
        try:
            events = task.result()
        except Exception as exc:
            logger.warning("Market %s delayed: %s", symbol, exc)
            continue
        for timestamp, kind, symbol, payload in events:
            by_time[timestamp][kind][symbol] = payload
    for timestamp, batch in sorted(by_time.items()):
        for symbol, bar in batch["touch"].items():
            stamp, high, low, close = bar
            engine.touch(symbol, timestamp, high, low, close)
            engine.state["markets"][symbol]["last_minute"] = stamp
        for symbol, bar in batch["main"].items():
            stamp, high, low, close, atr, exit_atr, opposite, current, context, allow_signal, why = bar
            engine.close_bar(symbol, timestamp, high, low, close, exit_atr, opposite)
            market = engine.state["markets"][symbol]
            market["last_candle"] = stamp
            if allow_signal:
                engine.signal(symbol, side=current, atr=atr, spec=state["champion"],
                              signal_time=stamp, fill_time=timestamp, context=context)
            if not market.get("position") and not market.get("pending"):
                if why:
                    ticks = market.setdefault("ticks", {})
                    ticks[why] = ticks.get(why, 0) + 1
                logger.info("TICK %s %s %s", symbol, stamp, "SKIP" if current is None else "NO_FILL")
        for symbol, price in batch["open"].items():
            engine.state["markets"][symbol]["mark"] = price
        for symbol in symbols:
            if symbol in batch["open"]:
                engine.fill(symbol, timestamp, batch["open"][symbol])
            elif (pending := engine.state["markets"][symbol].get("pending")) and pending["fill_time"] + MINUTE <= now:
                engine.state["markets"][symbol].pop("pending")
        for symbol, rate in batch["funding"].items():
            engine.funding(symbol, timestamp, rate)
    return engine


def _notify_trades(engine: Portfolio, notify) -> None:
    for entry in engine.opened:
        notify(f"PAPER OPEN {entry['symbol']} #{entry['entry_seq']} v{entry['version']} "
               f"entry={entry['entry']:.6g} | margin={entry['margin']:.2f} USDT")
    for trade in engine.closed:
        notify(f"PAPER CLOSE {trade['symbol']} #{trade['entry_seq']} v{trade['version']} "
               f"{trade['exit_reason']} | PnL={trade['pnl']:+.2f} USDT")


def run() -> None:
    if len(config.SYMBOLS) != len(set(config.SYMBOLS)):
        raise ValueError("SYMBOLS contains duplicates")
    if config.REVIEW_EVERY_N_ENTRIES <= 0 or not math.isfinite(config.INITIAL_CAPITAL) or config.INITIAL_CAPITAL <= 0:
        raise ValueError("Invalid portfolio review cadence or capital")
    if not (0 < config.MAX_MARGIN_FRACTION <= 1):
        raise ValueError("MAX_MARGIN_FRACTION must be in (0, 1]")
    if not config.LLM_API_KEY or not config.LLM_RESEARCH_ENABLED:
        raise ValueError("Portfolio research requires an enabled LLM and LLM_API_KEY")
    shutdown = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: shutdown.set())

    with writer_lock(config.PORTFOLIO_DATA_DIR):
        store = PortfolioStore(config.PORTFOLIO_DATA_DIR, config.SYMBOLS, create=config.PORTFOLIO_INIT)
        try:
            check_symbols(config.SYMBOLS)
            _run_loop(store, shutdown)
        finally:
            store.db.close()


def _run_loop(store: PortfolioStore, shutdown: threading.Event) -> None:
    step = interval_td()
    days = max(14, math.ceil(config.PAPER_WARMUP_BARS * step / pd.Timedelta(days=1)) + 5)
    feed = MarketData(days)
    notifier = Notifier()
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="candles")
    analyst = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")
    messenger = ThreadPoolExecutor(max_workers=1, thread_name_prefix="telegram")
    job: Future | None = None
    end_entry = 0
    retry_after = 0.0
    backup_at = time.monotonic()

    def notify(message):
        logger.info("%s", message)
        if notifier.token and notifier.chat_id:
            messenger.submit(notifier.send, message)

    notify(f"Portfolio PAPER {','.join(store.state['symbols'])} | cash={store.state['cash']:.2f} "
           f"| v{store.state['version']} | entries={store.state['entries']} "
           f"| commit={os.getenv('RELEASE_SHA', 'unknown')}")
    try:
        while not shutdown.is_set():
            if job and job.done():
                try:
                    selected, report = job.result()
                    state = deepcopy(store.state)
                    if selected:
                        state["version"] += 1
                        state["champion"] = selected
                        for market in state["markets"].values():
                            market.pop("pending", None)
                    report["champion_version"] = state["version"]
                    report["applied_spec_id"] = spec_id(state["champion"])
                    store.save(state, end_entry=end_entry, report=report)
                    notify(f"Portfolio review #{end_entry} {report['decision']} | v{state['version']} | {report['reason']}")
                except Exception:
                    logger.exception("Portfolio review #%s will retry", end_entry)
                    retry_after = time.monotonic() + config.RESEARCH_RETRY_MINUTES * 60
                finally:
                    job = None
            if store.state["champion"]:
                engine = _paper_tick(store.state, feed, pool)
                if engine.state != store.state:
                    store.save(engine.state, engine.opened, engine.closed)
                    _notify_trades(engine, notify)
            due = store.next_review()
            if job is None and due is not None and time.monotonic() >= retry_after:
                end_entry = due
                job = analyst.submit(research, deepcopy(store.state), store.review_context(due))
                logger.info("Portfolio research started at entry %s", end_entry)
            if time.monotonic() >= backup_at:
                try:
                    store.backup()
                except (OSError, sqlite3.Error):
                    logger.exception("Portfolio backup failed")
                backup_at = time.monotonic() + 3600
            shutdown.wait(config.PAPER_POLL_SECONDS)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        analyst.shutdown(wait=False, cancel_futures=True)
        messenger.shutdown(wait=False, cancel_futures=True)
