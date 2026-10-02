"""Short-term decision history for LLM context (separate from memory lessons)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[dict[str, Any], float], str]


def _history_path() -> Path:
    return Path(config.PATHS["DECISION_HISTORY"])


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _one_line(text: str, max_len: int = 80) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _extract_wait_or_level(decision: dict[str, Any]) -> str:
    loc = str(decision.get("market_location") or "").strip()
    reasoning = str(decision.get("reasoning") or "")
    pa = str(decision.get("price_action_signal") or "")
    blob = f"{loc} {reasoning} {pa}".upper()
    for token in (
        "BREAKOUT",
        "RETEST",
        "REJECTION",
        "PULLBACK",
        "ACCEPTANCE",
        "FAILED_BREAKOUT",
    ):
        if token in blob:
            return token
    if loc:
        return _one_line(loc, 40)
    return "n/a"


def rule_summary(
    decision: dict[str, Any],
    price: float,
    ts: str | None = None,
    *,
    truncate: bool = True,
) -> str:
    """Fixed one-line summary used in the next prompt."""
    ts = ts or _iso_now()
    regime = str(decision.get("market_regime") or "REGIME_UNKNOWN")
    action = str(decision.get("action") or "HOLD").upper()
    try:
        score = float(decision.get("setup_score") or 0)
        score_s = f"{score:.1f}"
    except (TypeError, ValueError):
        score_s = "?"
    wait = _extract_wait_or_level(decision)
    note = _one_line(
        str(decision.get("reasoning") or decision.get("price_action_signal") or ""),
        120 if not truncate else 70,
    )
    status = "thesis_ok"
    low = (str(decision.get("reasoning") or "") + " " + wait).lower()
    if "invalid" in low or "vô hiệu" in low or "fail" in low:
        status = "invalidated"
    elif action == "HOLD" and wait != "n/a":
        status = f"wait_{wait.lower()}"

    line = (
        f"{ts} | {regime} | {action} | {score_s} | px={round(float(price), 1)} | "
        f"{wait} | {status} | {note}"
    )
    if truncate:
        max_c = config.DECISION_SUMMARY_MAX_CHARS
        if len(line) > max_c:
            line = line[: max_c - 1] + "…"
    return line


def load_recent(n: int | None = None) -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    limit = n if n is not None else config.DECISION_HISTORY_MAX
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
        logger.warning("decision_history read failed: %s", exc)
        return []
    return rows[-limit:]


def format_for_prompt(entries: list[dict[str, Any]] | None = None) -> str:
    entries = entries if entries is not None else load_recent()
    if not entries:
        return "(chưa có lịch sử quyết định gần đây)"
    lines: list[str] = []
    for e in entries:
        s = str(e.get("summary") or "").strip()
        if s:
            lines.append(s)
    if not lines:
        return "(chưa có lịch sử quyết định gần đây)"

    budget = config.DECISION_HISTORY_PROMPT_MAX_CHARS
    # Prefer newest: keep from the end while under budget
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


def append(
    decision: dict[str, Any],
    price: float,
    *,
    ts: str | None = None,
    summarize_fn: SummarizeFn | None = None,
) -> dict[str, Any]:
    """
    Persist full decision + one-line summary.
    Uses rule_summary first; if longer than threshold and summarize_fn given, LLM compress.
    """
    ts = ts or _iso_now()
    draft = rule_summary(decision, price, ts=ts, truncate=False)
    reasoning_len = len(str(decision.get("reasoning") or "")) + len(
        str(decision.get("dow_structure_analysis") or "")
    )
    summary = rule_summary(decision, price, ts=ts, truncate=True)
    used_llm = False
    need_llm = (
        summarize_fn is not None
        and (
            len(draft) > config.DECISION_SUMMARY_LLM_THRESHOLD
            or reasoning_len > config.DECISION_SUMMARY_LLM_THRESHOLD
        )
    )
    if need_llm:
        try:
            llm_line = (summarize_fn(decision, price) or "").strip()
            llm_line = re.sub(r"\s+", " ", llm_line)
            if llm_line:
                if len(llm_line) > config.DECISION_SUMMARY_MAX_CHARS:
                    llm_line = llm_line[: config.DECISION_SUMMARY_MAX_CHARS - 1] + "…"
                if not llm_line.startswith(ts[:10]):
                    llm_line = f"{ts} | {llm_line}"
                    if len(llm_line) > config.DECISION_SUMMARY_MAX_CHARS:
                        llm_line = llm_line[: config.DECISION_SUMMARY_MAX_CHARS - 1] + "…"
                summary = llm_line
                used_llm = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM decision summarize failed, keep rule line: %s", exc)

    # Slim full for disk (drop bulky structure_meta lists if huge — keep meta keys)
    full = dict(decision)
    meta = full.get("structure_meta")
    if isinstance(meta, dict):
        full["structure_meta"] = {
            k: meta.get(k)
            for k in ("bars_1d", "bars_4h", "bars_1h")
            if k in meta
        }

    record = {
        "ts": ts,
        "price": float(price),
        "summary": summary,
        "full": full,
        "summary_via": "llm" if used_llm else "rule",
    }

    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_recent(n=10_000)
    existing.append(record)
    existing = existing[-config.DECISION_HISTORY_MAX :]
    try:
        with path.open("w", encoding="utf-8") as f:
            for row in existing:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "decision_history append via=%s chars=%d n=%d",
            record["summary_via"],
            len(summary),
            len(existing),
        )
    except OSError as exc:
        logger.exception("decision_history write failed: %s", exc)
    return record
