"""Short-term decision history for LLM context (separate from memory lessons).

Persists locally under generated/decision_history.jsonl and optionally mirrors
to GitHub so Blitz rebuilds can restore the last N summaries.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

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


def _rows_to_jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + (
        "\n" if rows else ""
    )


def _write_rows(rows: list[dict[str, Any]]) -> None:
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_rows_to_jsonl(rows), encoding="utf-8")


def _slim_for_remote(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public-repo safe: keep prompt-useful fields, drop bulky full blobs."""
    out: list[dict[str, Any]] = []
    for r in rows[-config.DECISION_HISTORY_MAX :]:
        out.append(
            {
                "ts": r.get("ts"),
                "price": r.get("price"),
                "summary": r.get("summary"),
                "summary_via": r.get("summary_via", "rule"),
                "full": {
                    "market_regime": (r.get("full") or {}).get("market_regime"),
                    "action": (r.get("full") or {}).get("action"),
                    "setup_score": (r.get("full") or {}).get("setup_score"),
                    "reasoning": _one_line(
                        str((r.get("full") or {}).get("reasoning") or ""), 120
                    ),
                },
            }
        )
    return out


def _github_headers() -> dict[str, str] | None:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return None
    return {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "btc-dow-pa-agent-history",
    }


def backup_remote(rows: list[dict[str, Any]] | None = None) -> bool:
    """Upsert slim history JSONL to GitHub (survives Blitz rebuild)."""
    headers = _github_headers()
    if not headers:
        return False
    rows = _slim_for_remote(rows if rows is not None else load_recent(n=10_000))
    repo = config.HISTORY_GITHUB_REPO
    path = config.HISTORY_GITHUB_PATH
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    content_b64 = base64.b64encode(_rows_to_jsonl(rows).encode("utf-8")).decode("ascii")
    sha = None
    try:
        meta = requests.get(url, headers=headers, timeout=30)
        if meta.status_code == 200:
            sha = meta.json().get("sha")
        elif meta.status_code != 404:
            logger.warning("history remote GET HTTP %s", meta.status_code)
            return False
        body: dict[str, Any] = {
            "message": "chore: sync decision_history for Blitz persistence",
            "content": content_b64,
            "branch": "main",
        }
        if sha:
            body["sha"] = sha
        put = requests.put(url, headers=headers, json=body, timeout=60)
        if put.status_code not in {200, 201}:
            logger.warning(
                "history remote PUT HTTP %s: %s", put.status_code, put.text[:200]
            )
            return False
        logger.info("decision_history backed up to GitHub n=%d", len(rows))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("history remote backup failed: %s", exc)
        return False


def restore_from_remote() -> int:
    """If local history empty, pull from GitHub. Returns rows restored."""
    local = load_recent(n=10_000)
    if local:
        return 0
    headers = _github_headers()
    if not headers:
        logger.info("No HISTORY_GITHUB_TOKEN — skip remote restore")
        return 0
    repo = config.HISTORY_GITHUB_REPO
    path = config.HISTORY_GITHUB_PATH
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            logger.info("No remote decision_history yet")
            return 0
        if resp.status_code != 200:
            logger.warning("history remote restore HTTP %s", resp.status_code)
            return 0
        data = resp.json()
        raw = base64.b64decode(data.get("content") or "").decode("utf-8")
        rows: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        rows = rows[-config.DECISION_HISTORY_MAX :]
        if not rows:
            return 0
        _write_rows(rows)
        logger.info("decision_history restored from GitHub n=%d", len(rows))
        return len(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("history remote restore failed: %s", exc)
        return 0


def ensure_restored() -> int:
    """Call once at process start — never deletes existing local history."""
    n_local = len(load_recent(n=10_000))
    if n_local > 0:
        logger.info("decision_history local n=%d — keep as-is", n_local)
        return 0
    return restore_from_remote()


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
    Never truncates history below DECISION_HISTORY_MAX except rolling window.
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

    existing = load_recent(n=10_000)
    existing.append(record)
    existing = existing[-config.DECISION_HISTORY_MAX :]
    try:
        _write_rows(existing)
        logger.info(
            "decision_history append via=%s chars=%d n=%d | %s",
            record["summary_via"],
            len(summary),
            len(existing),
            summary,
        )
    except OSError as exc:
        logger.exception("decision_history write failed: %s", exc)

    # Best-effort remote mirror (does not delete local on failure)
    try:
        backup_remote(existing)
    except Exception as exc:  # noqa: BLE001
        logger.warning("history backup skipped: %s", exc)

    return record
