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
    state = str(decision.get("state") or "").strip()
    try:
        level = int(decision.get("signal_level") or 0)
    except (TypeError, ValueError):
        level = 0
    level_s = f"L{level}" if level else ""
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

    state_part = f"{state}|{level_s}" if state or level_s else "n/a"
    line = (
        f"{ts} | {regime} | {state_part} | {action} | {score_s} | "
        f"px={round(float(price), 1)} | {wait} | {status} | {note}"
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
                    "state": (r.get("full") or {}).get("state"),
                    "signal_level": (r.get("full") or {}).get("signal_level"),
                    "action": (r.get("full") or {}).get("action"),
                    "setup_score": (r.get("full") or {}).get("setup_score"),
                    "win_probability": (r.get("full") or {}).get("win_probability"),
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


def _github_contents_url(repo: str, path: str, branch: str | None = None) -> str:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    if branch:
        return f"{url}?ref={branch}"
    return url


def _ensure_history_branch(headers: dict[str, str], repo: str, branch: str) -> bool:
    """Create history branch from main tip if missing."""
    if not branch or branch == "main":
        return True
    ref_url = f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}"
    try:
        chk = requests.get(ref_url, headers=headers, timeout=30)
        if chk.status_code == 200:
            return True
        if chk.status_code != 404:
            logger.warning("history branch check HTTP %s", chk.status_code)
            return False
        main_ref = requests.get(
            f"https://api.github.com/repos/{repo}/git/ref/heads/main",
            headers=headers,
            timeout=30,
        )
        if main_ref.status_code != 200:
            logger.warning("main ref HTTP %s", main_ref.status_code)
            return False
        sha = main_ref.json().get("object", {}).get("sha")
        if not sha:
            return False
        create = requests.post(
            f"https://api.github.com/repos/{repo}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch}", "sha": sha},
            timeout=30,
        )
        if create.status_code not in {201, 422}:
            logger.warning("create branch HTTP %s: %s", create.status_code, create.text[:120])
            return False
        logger.info("Created GitHub branch %s for decision_history", branch)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure history branch failed: %s", exc)
        return False


def _fetch_remote_rows(
    headers: dict[str, str],
    repo: str,
    path: str,
    branch: str,
) -> list[dict[str, Any]] | None:
    url = _github_contents_url(repo, path, branch)
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        logger.warning("history remote GET %s HTTP %s", branch, resp.status_code)
        return None
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
    return rows


def backup_remote(rows: list[dict[str, Any]] | None = None) -> bool:
    """Upsert slim history JSONL to GitHub (survives Blitz rebuild)."""
    headers = _github_headers()
    if not headers:
        return False
    rows = _slim_for_remote(rows if rows is not None else load_recent(n=10_000))
    repo = config.HISTORY_GITHUB_REPO
    path = config.HISTORY_GITHUB_PATH
    branch = config.HISTORY_GITHUB_BRANCH or "history-data"
    url = _github_contents_url(repo, path)
    content_b64 = base64.b64encode(_rows_to_jsonl(rows).encode("utf-8")).decode("ascii")
    sha = None
    try:
        if not _ensure_history_branch(headers, repo, branch):
            return False
        meta = requests.get(_github_contents_url(repo, path, branch), headers=headers, timeout=30)
        if meta.status_code == 200:
            sha = meta.json().get("sha")
        elif meta.status_code != 404:
            logger.warning("history remote GET HTTP %s", meta.status_code)
            return False
        body: dict[str, Any] = {
            "message": "chore: sync decision_history for Blitz persistence",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        put = requests.put(url, headers=headers, json=body, timeout=60)
        if put.status_code not in {200, 201}:
            logger.warning(
                "history remote PUT HTTP %s: %s", put.status_code, put.text[:200]
            )
            return False
        logger.info(
            "decision_history backed up to GitHub branch=%s n=%d", branch, len(rows)
        )
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
    branch = config.HISTORY_GITHUB_BRANCH or "history-data"
    try:
        rows = _fetch_remote_rows(headers, repo, path, branch)
        source = branch
        if rows is None:
            rows = _fetch_remote_rows(headers, repo, path, "main")
            source = "main"
        if not rows:
            logger.info("No remote decision_history yet (branch=%s)", branch)
            return 0
        rows = rows[-config.DECISION_HISTORY_MAX :]
        _write_rows(rows)
        logger.info("decision_history restored from GitHub branch=%s n=%d", source, len(rows))
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
