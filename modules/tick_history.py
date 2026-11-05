"""Short Champion tick history for research context (not AI entry authority)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def _path() -> Path:
    return Path(config.PATHS["TICK_HISTORY"])


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _one_line(text: str, max_len: int = 100) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def load_recent(n: int | None = None) -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    limit = n if n is not None else config.TICK_HISTORY_MAX
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("tick_history read failed: %s", exc)
        return []
    return rows[-limit:]


def format_for_prompt(entries: list[dict[str, Any]] | None = None) -> str:
    entries = entries if entries is not None else load_recent()
    if not entries:
        return "(chưa có tick history)"
    lines = [str(e.get("summary") or "").strip() for e in entries]
    lines = [x for x in lines if x]
    if not lines:
        return "(chưa có tick history)"
    budget = config.TICK_HISTORY_PROMPT_MAX_CHARS
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        add = len(line) + (1 if kept else 0)
        if total + add > budget:
            break
        kept.append(line)
        total += add
    kept.reverse()
    return "\n".join(kept)


def append(report: dict[str, Any]) -> dict[str, Any] | None:
    """Append one closed-candle tick report (SKIP/PENDING/HOLD/OPEN/CLOSE…)."""
    if not report:
        return None
    action = str(report.get("action") or "SKIP")
    candle = str(report.get("candle") or "")[-19:]
    close = report.get("close")
    close_s = f"{float(close):.1f}" if close is not None else "—"
    family = report.get("family") or "—"
    why = _one_line(str(report.get("why") or ""), 120)
    ts = _iso_now()
    summary = f"{ts} | {candle} | {action} | {family} | px={close_s} | {why}"
    record = {
        "ts": ts,
        "candle": report.get("candle"),
        "action": action,
        "family": family,
        "genome_id": report.get("genome_id"),
        "close": close,
        "why": why,
        "summary": summary,
    }
    existing = load_recent(n=10_000)
    existing.append(record)
    existing = existing[-config.TICK_HISTORY_MAX :]
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in existing:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        logger.info("tick_history n=%d | %s", len(existing), summary[:160])
    except OSError as exc:
        logger.warning("tick_history write failed: %s", exc)
        return None
    return record
