"""Local candle history cache (JSONL) + incremental sync / downtime catch-up."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import config
from modules.binance_client import BinanceClient, BinanceClientError

logger = logging.getLogger(__name__)

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


def history_dir() -> Path:
    d = Path(config.HISTORY_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def history_path(interval: str) -> Path:
    sym = config.SYMBOL.upper()
    safe = interval.replace("/", "")
    return history_dir() / f"{sym}_{safe}.jsonl"


def load(interval: str) -> list[dict[str, Any]]:
    path = history_path(interval)
    if not path.exists():
        return []
    by_ts: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                ot = int(row["open_time"])
            except (KeyError, TypeError, ValueError):
                continue
            by_ts[ot] = {
                "open_time": ot,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0) or 0),
                "close_time": int(row.get("close_time") or ot),
            }
    return [by_ts[k] for k in sorted(by_ts)]


def save(interval: str, candles: list[dict[str, Any]]) -> None:
    path = history_path(interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in candles:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(path)


def upsert(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ts: dict[int, dict[str, Any]] = {int(c["open_time"]): c for c in existing}
    for c in incoming:
        by_ts[int(c["open_time"])] = c
    return [by_ts[k] for k in sorted(by_ts)]


def trim(candles: list[dict[str, Any]], target_bars: int) -> list[dict[str, Any]]:
    if target_bars <= 0 or len(candles) <= target_bars:
        return candles
    # soft trim only when clearly over window
    soft = int(target_bars * 1.1)
    if len(candles) <= soft:
        return candles
    return candles[-target_bars:]


def _interval_ms(interval: str) -> int:
    return INTERVAL_MS.get(interval, 3_600_000)


def sync(
    binance: BinanceClient,
    interval: str,
    target_bars: int,
) -> list[dict[str, Any]]:
    """
    - Lần đầu: lấy 1 page nến gần nhất (bao nhiêu API cho thì lấy).
    - Mỗi chu kỳ: catch-up nến mới (từ last → now) + backfill thêm 1 page nến cũ
      cho đến khi đủ target_bars (bồi đắp dần, không ép full 1 năm ngay).
    """
    existing = load(interval)
    now_ms = int(time.time() * 1000)
    step = _interval_ms(interval)
    page = min(1000, max(config.HISTORY_BACKFILL_PER_SYNC, 100))

    try:
        if not existing:
            logger.info("bootstrap %s — 1 page gần nhất (target sau=%d)", interval, target_bars)
            fetched = binance.get_klines(interval, limit=page)
            merged = trim(upsert([], fetched), target_bars)
            save(interval, merged)
            logger.info(
                "bootstrap %s done bars=%d (sẽ bồi đắp dần tới %d)",
                interval,
                len(merged),
                target_bars,
            )
            return merged

        last_ot = int(existing[-1]["open_time"])
        # 1) Catch-up sau downtime / nến mới
        newer = binance.get_klines_range(
            interval,
            start_time=last_ot,
            end_time=now_ms,
            max_bars=page * 3,
            max_pages=5,
        )
        if not newer:
            newer = binance.get_klines(interval, limit=5)
        merged = upsert(existing, newer)
        added_new = len(merged) - len(existing)

        # 2) Bồi đắp dần về quá khứ (1 page / chu kỳ) nếu chưa đủ target
        added_old = 0
        if len(merged) < target_bars:
            oldest = int(merged[0]["open_time"])
            back_start = oldest - step * page
            older = binance.get_klines(
                interval,
                limit=page,
                start_time=back_start,
                end_time=oldest - 1,
            )
            before = len(merged)
            merged = upsert(older, merged)
            added_old = len(merged) - before
            logger.info(
                "backfill %s +%d older (total=%d/%d)",
                interval,
                added_old,
                len(merged),
                target_bars,
            )

        merged = trim(merged, target_bars)
        save(interval, merged)
        logger.info(
            "sync %s new≈%d old≈%d total=%d/%d last=%s",
            interval,
            max(added_new, 0),
            added_old,
            len(merged),
            target_bars,
            merged[-1]["open_time"] if merged else None,
        )
        return merged
    except BinanceClientError:
        logger.exception("sync failed interval=%s — return cached", interval)
        return existing


def sync_all(binance: BinanceClient) -> dict[str, list[dict[str, Any]]]:
    return {
        "1d": sync(binance, config.TIMEFRAME_HTF, config.HISTORY_BARS_1D),
        "4h": sync(binance, config.TIMEFRAME_TREND, config.HISTORY_BARS_4H),
        "1h": sync(binance, config.TIMEFRAME_MAIN, config.HISTORY_BARS_1H),
    }


def get_recent(candles: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    return candles[-n:]
