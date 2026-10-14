"""Adaptive rules from meta-review — injected into MEMORY, not core prompt rewrite."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

DEFAULT_RULES: dict[str, Any] = {
    "version": 1,
    "stable": False,
    "updated_at": "",
    "closes_at_review": 0,
    "freeze": False,
    "rules": [],
}


def _path() -> Path:
    return Path(config.PATHS["ADAPTIVE_RULES"])


def load_rules() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return dict(DEFAULT_RULES)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULT_RULES)
        data.setdefault("rules", [])
        data.setdefault("stable", False)
        data.setdefault("freeze", False)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("adaptive_rules read failed: %s", exc)
        return dict(DEFAULT_RULES)


def save_rules(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    try:
        backup_remote(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("adaptive_rules backup skipped: %s", exc)


def validate_and_normalize(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rules_in = raw.get("rules") if isinstance(raw.get("rules"), list) else raw
    if not isinstance(rules_in, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rules_in:
        if not isinstance(item, dict):
            continue
        when = str(item.get("when") or "").strip()
        action = str(item.get("action") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not when or not action:
            continue
        # Reject WR-chasing trail BE style
        blob = f"{when} {action} {reason}".lower()
        if "trail be" in blob or "breakeven để thắng" in blob or "wr ảo" in blob:
            continue
        rid = str(item.get("id") or f"r{len(out)+1}").strip()[:40]
        if rid in seen:
            continue
        seen.add(rid)
        try:
            evidence = int(item.get("evidence_n") or item.get("evidence") or 1)
        except (TypeError, ValueError):
            evidence = 1
        try:
            weight = int(item.get("weight") or 1)
        except (TypeError, ValueError):
            weight = 1
        out.append(
            {
                "id": rid,
                "when": when[:200],
                "action": action[:200],
                "reason": reason[:240],
                "evidence_n": max(1, evidence),
                "weight": max(1, min(3, weight)),
            }
        )
        if len(out) >= config.ADAPTIVE_RULES_MAX:
            break
    out.sort(key=lambda r: int(r.get("weight", 1)), reverse=True)
    return out


def merge_rules(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for r in existing + incoming:
        key = (str(r.get("when") or "") + "|" + str(r.get("action") or "")).lower()
        if not key.strip("|"):
            continue
        if key in by_key:
            prev = by_key[key]
            prev["evidence_n"] = int(prev.get("evidence_n", 1)) + int(
                r.get("evidence_n", 1) or 1
            )
            prev["weight"] = max(int(prev.get("weight", 1)), int(r.get("weight", 1)))
            if r.get("reason"):
                prev["reason"] = str(r.get("reason"))[:240]
        else:
            by_key[key] = dict(r)
    merged = list(by_key.values())
    merged.sort(
        key=lambda x: (int(x.get("weight", 1)), int(x.get("evidence_n", 1))),
        reverse=True,
    )
    return merged[: config.ADAPTIVE_RULES_MAX]


def format_for_prompt(data: dict[str, Any] | None = None, max_chars: int = 700) -> str:
    data = data if data is not None else load_rules()
    rules = list(data.get("rules") or [])
    if not rules:
        return "(chưa có adaptive_rules)"
    flag = []
    if data.get("stable"):
        flag.append("STABLE")
    if data.get("freeze"):
        flag.append("FREEZE")
    head = f"[{'|'.join(flag)}] " if flag else ""
    lines = [f"{head}Adaptive rules (ưu tiên expectancy/PnL):"]
    for r in rules[: config.ADAPTIVE_RULES_MAX]:
        lines.append(
            f"- [{r.get('id')}|w={r.get('weight')}|n={r.get('evidence_n')}] "
            f"WHEN {r.get('when')} → {r.get('action')} ({r.get('reason')})"
        )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _github_headers() -> dict[str, str] | None:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return None
    return {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "btc-dow-pa-agent-rules",
    }


def backup_remote(data: dict[str, Any] | None = None) -> bool:
    headers = _github_headers()
    if not headers:
        return False
    data = data if data is not None else load_rules()
    try:
        from modules.decision_history import (
            _ensure_history_branch,
            _github_contents_url,
        )

        repo = config.HISTORY_GITHUB_REPO
        path = config.ADAPTIVE_RULES_GITHUB_PATH
        branch = config.HISTORY_GITHUB_BRANCH or "history-data"
        if not _ensure_history_branch(headers, repo, branch):
            return False
        content_b64 = base64.b64encode(
            json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("ascii")
        meta = requests.get(
            _github_contents_url(repo, path, branch), headers=headers, timeout=30
        )
        sha = meta.json().get("sha") if meta.status_code == 200 else None
        body: dict[str, Any] = {
            "message": "chore: sync adaptive_rules for Blitz persistence",
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
        if put.status_code not in {200, 201}:
            logger.warning("adaptive_rules PUT HTTP %s", put.status_code)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("adaptive_rules backup failed: %s", exc)
        return False


def restore_from_remote() -> bool:
    path = _path()
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
            if cur.get("rules"):
                return False
        except (OSError, json.JSONDecodeError):
            pass
    headers = _github_headers()
    if not headers:
        return False
    try:
        from modules.decision_history import _github_contents_url

        repo = config.HISTORY_GITHUB_REPO
        remote = config.ADAPTIVE_RULES_GITHUB_PATH
        branch = config.HISTORY_GITHUB_BRANCH or "history-data"
        resp = requests.get(
            _github_contents_url(repo, remote, branch), headers=headers, timeout=30
        )
        if resp.status_code != 200:
            return False
        raw = base64.b64decode(resp.json().get("content") or "").decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("adaptive_rules restored from remote n=%d", len(data.get("rules") or []))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("adaptive_rules restore failed: %s", exc)
        return False
