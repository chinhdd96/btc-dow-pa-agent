"""Adaptive rules from meta-review — APPLY schema enforced by Python gates."""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

EFFECTS = frozenset({"RAISE_SCORE", "FORBID_SOFT", "PREFER_ALT", "CHECKLIST"})
_REMOTE_BACKUP = True
_PATH_OVERRIDE: Path | None = None


def set_remote_backup(enabled: bool) -> None:
    global _REMOTE_BACKUP
    _REMOTE_BACKUP = bool(enabled)


def set_rules_path(path: str | Path | None) -> None:
    global _PATH_OVERRIDE
    _PATH_OVERRIDE = Path(path) if path else None

DEFAULT_RULES: dict[str, Any] = {
    "version": 2,
    "stable": False,
    "updated_at": "",
    "closes_at_review": 0,
    "freeze": False,
    "rules": [],
}


def _path() -> Path:
    if _PATH_OVERRIDE is not None:
        return _PATH_OVERRIDE
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
        # Migrate legacy text rules into APPLY schema when possible
        migrated = []
        for r in data.get("rules") or []:
            nr = _normalize_one(r, losing_tags=None)
            if nr:
                migrated.append(nr)
        data["rules"] = migrated
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
    if _REMOTE_BACKUP:
        try:
            backup_remote(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("adaptive_rules backup skipped: %s", exc)


def _infer_effect(action: str, when: str) -> str | None:
    blob = f"{action} {when}".upper()
    if any(x in blob for x in ("FORBID", "BLOCK", "CẤM", "TRÁNH", "AVOID", "SKIP")):
        return "FORBID_SOFT"
    if any(x in blob for x in ("PREFER_ALT", "ĐỔI SETUP", "ALT SETUP", "KHÁC SETUP", "DIVERS")):
        return "PREFER_ALT"
    if any(x in blob for x in ("RAISE", "L3", "CHỈ L3", "SCORE+", "NÂNG ĐIỂM", "RR≥", "RR >=")):
        return "RAISE_SCORE"
    if any(x in blob for x in ("CHECK", "SL ", "STOP", "CHECKLIST", "KIỂM TRA")):
        return "CHECKLIST"
    # ENTER / trade more on same setup — reject (not an APPLY effect we want)
    if any(x in blob for x in ("ENTER", "VÀO LỆNH", "OPEN ")):
        return None
    return "CHECKLIST"


def _infer_setup(when: str, action: str, item: dict[str, Any]) -> str:
    explicit = str(item.get("when_setup") or "").strip().upper()
    if explicit:
        return explicit[:32]
    blob = f"{when} {action}".upper()
    for tag in (
        "FAILED_BREAKOUT",
        "SWEEP",
        "BREAKOUT",
        "BREAKDOWN",
        "RETEST",
        "PULLBACK",
        "RECLAIM",
        "COMPRESSION",
    ):
        if tag in blob:
            return tag
    m = re.search(r"SETUP_TAG\s*==\s*['\"]([A-Z_]+)['\"]", blob)
    if m:
        return m.group(1)[:32]
    return "*"


def _is_too_broad(when: str, when_setup: str) -> bool:
    w = (when or "").strip().lower()
    if not w:
        return True
    # Bare side filters without setup
    if re.fullmatch(r"action\s*==\s*['\"]?(buy|sell)['\"]?", w):
        return True
    if when_setup == "*" and "setup" not in w and not any(
        t.lower() in w
        for t in (
            "failed_breakout",
            "sweep",
            "breakout",
            "breakdown",
            "retest",
            "pullback",
        )
    ):
        # checklist on * is ok if mentions SL/process
        if any(x in w for x in ("stop", "sl", "checklist", "process")):
            return False
        return True
    return False


def _normalize_one(
    item: dict[str, Any],
    *,
    losing_tags: set[str] | None,
) -> dict[str, Any] | None:
    when = str(item.get("when") or "").strip()
    action = str(item.get("action") or item.get("then") or "").strip()
    reason = str(item.get("reason") or "").strip()
    effect = str(item.get("effect") or "").strip().upper()
    if effect not in EFFECTS:
        effect = _infer_effect(action, when) or ""
    if effect not in EFFECTS:
        return None
    when_setup = _infer_setup(when, action, item)
    if _is_too_broad(when, when_setup):
        return None
    blob = f"{when} {action} {reason}".lower()
    if "trail be" in blob or "breakeven để thắng" in blob or "wr ảo" in blob:
        return None
    # Reject reinforcing a losing setup via ENTER-style — only FORBID/RAISE/PREFER/CHECKLIST allowed
    if losing_tags and when_setup in losing_tags and effect not in {
        "FORBID_SOFT",
        "RAISE_SCORE",
        "PREFER_ALT",
        "CHECKLIST",
    }:
        return None
    # Explicit ENTER on losing tag → drop
    if losing_tags and when_setup in losing_tags and any(
        x in f"{action} {when}".upper() for x in ("ENTER", "VÀO LỆNH")
    ):
        return None

    try:
        evidence = int(item.get("evidence_n") or item.get("evidence") or 1)
    except (TypeError, ValueError):
        evidence = 1
    try:
        weight = int(item.get("weight") or 1)
    except (TypeError, ValueError):
        weight = 1
    try:
        magnitude = float(item.get("magnitude") or (0.5 if effect == "RAISE_SCORE" else 0.0))
    except (TypeError, ValueError):
        magnitude = 0.5 if effect == "RAISE_SCORE" else 0.0
    prefer_tags = item.get("prefer_tags") or []
    if isinstance(prefer_tags, str):
        prefer_tags = [prefer_tags]
    prefer_tags = [str(x).upper()[:32] for x in prefer_tags if str(x).strip()][:5]
    if effect == "PREFER_ALT" and not prefer_tags:
        prefer_tags = ["PULLBACK", "RETEST", "BREAKOUT", "SWEEP", "BREAKDOWN"]

    rid = str(item.get("id") or f"r{abs(hash(when_setup + effect + when)) % 10000}").strip()[:40]
    try:
        created_at = int(item.get("created_at_closes") or 0)
    except (TypeError, ValueError):
        created_at = 0
    try:
        pnl_since = float(item.get("pnl_since") or 0)
    except (TypeError, ValueError):
        pnl_since = 0.0

    # Human-readable when/action for prompt
    when_out = when or f"setup_tag=='{when_setup}'"
    action_out = action or f"{effect} mag={magnitude}"
    return {
        "id": rid,
        "when": when_out[:200],
        "action": action_out[:200],
        "reason": reason[:240],
        "when_setup": when_setup,
        "effect": effect,
        "magnitude": round(max(0.0, min(3.0, magnitude)), 2),
        "prefer_tags": prefer_tags,
        "evidence_n": max(1, evidence),
        "weight": max(1, min(3, weight)),
        "created_at_closes": created_at,
        "pnl_since": round(pnl_since, 4),
    }


def validate_and_normalize(
    raw: Any,
    *,
    losing_tags: set[str] | list[str] | None = None,
    closes_now: int = 0,
) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rules_in = raw.get("rules") if isinstance(raw.get("rules"), list) else raw
    if not isinstance(rules_in, list):
        return []
    lose = {str(x).upper() for x in (losing_tags or [])}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rules_in:
        if not isinstance(item, dict):
            continue
        nr = _normalize_one(item, losing_tags=lose or None)
        if not nr:
            continue
        if not nr.get("created_at_closes"):
            nr["created_at_closes"] = int(closes_now or 0)
        key = f"{nr['when_setup']}|{nr['effect']}|{nr['when']}".lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nr)
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
        nr = _normalize_one(r, losing_tags=None) if "effect" not in r else dict(r)
        if not nr:
            nr = _normalize_one(r, losing_tags=None)
        if not nr:
            continue
        key = (
            str(nr.get("when_setup") or "")
            + "|"
            + str(nr.get("effect") or "")
            + "|"
            + str(nr.get("when") or "")
        ).lower()
        if key in by_key:
            prev = by_key[key]
            prev["evidence_n"] = int(prev.get("evidence_n", 1)) + int(
                nr.get("evidence_n", 1) or 1
            )
            prev["weight"] = max(int(prev.get("weight", 1)), int(nr.get("weight", 1)))
            if nr.get("reason"):
                prev["reason"] = str(nr.get("reason"))[:240]
            if nr.get("prefer_tags"):
                prev["prefer_tags"] = nr["prefer_tags"]
            if nr.get("magnitude"):
                prev["magnitude"] = max(
                    float(prev.get("magnitude") or 0), float(nr.get("magnitude") or 0)
                )
        else:
            by_key[key] = dict(nr)
    merged = list(by_key.values())
    merged.sort(
        key=lambda x: (int(x.get("weight", 1)), int(x.get("evidence_n", 1))),
        reverse=True,
    )
    return merged[: config.ADAPTIVE_RULES_MAX]


def apply_rules_to_score(
    rules: list[dict[str, Any]],
    setup_tag: str,
) -> tuple[float, list[str], list[str]]:
    """
    Apply RAISE_SCORE / FORBID_SOFT / PREFER_ALT for current setup_tag.
    Returns (extra_score_penalty, notes, prefer_tags).
    """
    tag = (setup_tag or "").upper()
    extra = 0.0
    notes: list[str] = []
    prefer: list[str] = []
    for r in rules:
        when_setup = str(r.get("when_setup") or "*").upper()
        if when_setup not in {"*", tag}:
            continue
        effect = str(r.get("effect") or "").upper()
        mag = float(r.get("magnitude") or 0.5)
        rid = r.get("id")
        if effect == "RAISE_SCORE":
            extra += mag
            notes.append(f"Rule {rid}: RAISE_SCORE +{mag:.1f} ({when_setup})")
        elif effect == "FORBID_SOFT":
            extra += max(mag, 1.0)
            notes.append(f"Rule {rid}: FORBID_SOFT +{max(mag, 1.0):.1f} ({when_setup})")
        elif effect == "PREFER_ALT":
            extra += max(mag, 0.5)
            tags = list(r.get("prefer_tags") or [])
            prefer.extend(tags)
            notes.append(
                f"Rule {rid}: PREFER_ALT {tags or ['(any other)']} thay vì {when_setup}"
            )
        elif effect == "CHECKLIST":
            notes.append(f"Rule {rid}: CHECKLIST — {r.get('action') or r.get('reason')}")
    # dedupe prefer
    seen: set[str] = set()
    prefer_out = []
    for t in prefer:
        if t not in seen and t != tag:
            seen.add(t)
            prefer_out.append(t)
    return extra, notes, prefer_out


def kill_failed_rules(
    rules: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    *,
    closes_now: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop/decay rules whose setup still has negative expectancy after enough closes."""
    from modules import trade_analytics

    min_k = int(getattr(config, "RULE_KILL_MIN_TRADES", 6) or 6)
    kept: list[dict[str, Any]] = []
    killed: list[str] = []
    for r in rules:
        created = int(r.get("created_at_closes") or 0)
        age = max(0, closes_now - created)
        when_setup = str(r.get("when_setup") or "*").upper()
        effect = str(r.get("effect") or "").upper()
        if when_setup == "*" or age < min_k:
            kept.append(r)
            continue
        # FORBID/RAISE/PREFER on bleeding setups are working as intended — keep
        st = trade_analytics.setup_expectancy(trades, when_setup)
        if st["n"] < min_k:
            kept.append(r)
            continue
        if st["expectancy"] >= 0:
            kept.append(r)
            continue
        # Setup still bleeding: kill ENTER-like leftovers; decay CHECKLIST; keep FORBID/RAISE/PREFER
        if effect in {"FORBID_SOFT", "RAISE_SCORE", "PREFER_ALT"}:
            kept.append(r)
            continue
        killed.append(f"{r.get('id')}({when_setup}/{effect})")
        # else drop
    return kept, killed


def losing_setup_tags(trades: list[dict[str, Any]], min_n: int = 5) -> set[str]:
    from modules import trade_analytics

    s = trade_analytics.summarize_trades(trades)
    out: set[str] = set()
    for tag, st in (s.get("by_setup") or {}).items():
        if int(st.get("n") or 0) >= min_n and float(st.get("expectancy") or 0) < 0:
            out.add(str(tag).upper())
    return out


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
    lines = [f"{head}Adaptive rules (APPLY — Python gate):"]
    for r in rules[: config.ADAPTIVE_RULES_MAX]:
        lines.append(
            f"- [{r.get('id')}|{r.get('effect')}|setup={r.get('when_setup')}|"
            f"w={r.get('weight')}|n={r.get('evidence_n')}] "
            f"{r.get('action')} ({r.get('reason')})"
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
        logger.info(
            "adaptive_rules restored from remote n=%d", len(data.get("rules") or [])
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("adaptive_rules restore failed: %s", exc)
        return False
