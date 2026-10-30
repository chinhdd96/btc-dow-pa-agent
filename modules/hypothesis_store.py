"""AI-owned hypotheses: propose → apply → credit → keep/kill (pole-3 learning)."""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

EFFECTS = frozenset({"AVOID_SOFT", "PREFER_ALT", "RAISE_SCORE", "CHECKLIST"})
# Map legacy adaptive effects
_EFFECT_ALIAS = {
    "FORBID_SOFT": "AVOID_SOFT",
    "FORBID": "AVOID_SOFT",
    "AVOID": "AVOID_SOFT",
}

DEFAULT_STORE: dict[str, Any] = {
    "version": 1,
    "updated_at": "",
    "migrated_from_adaptive": False,
    "hypotheses": [],
}

_override_path: Path | None = None
_remote_backup: bool = True


def set_path(path: str | Path | None) -> None:
    global _override_path
    _override_path = Path(path) if path else None


def set_remote_backup(enabled: bool) -> None:
    global _remote_backup
    _remote_backup = bool(enabled)


def _path() -> Path:
    if _override_path is not None:
        return _override_path
    return Path(config.PATHS["HYPOTHESES"])


def load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        data = dict(DEFAULT_STORE)
        data["hypotheses"] = []
        return data
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULT_STORE)
        data.setdefault("hypotheses", [])
        data.setdefault("migrated_from_adaptive", False)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("hypotheses read failed: %s", exc)
        return dict(DEFAULT_STORE)


def save(data: dict[str, Any], *, backup: bool = True) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    # Cap active
    hyps = list(data.get("hypotheses") or [])
    active = [h for h in hyps if str(h.get("status") or "") == "active"]
    inactive = [h for h in hyps if str(h.get("status") or "") != "active"]
    active.sort(
        key=lambda h: (int(h.get("evidence_n") or 1), float(h.get("magnitude") or 0)),
        reverse=True,
    )
    max_a = int(getattr(config, "HYP_MAX_ACTIVE", 12) or 12)
    if len(active) > max_a:
        for h in active[max_a:]:
            h["status"] = "decayed"
            inactive.append(h)
        active = active[:max_a]
    data["hypotheses"] = active + inactive[-40:]
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
    if backup and _remote_backup:
        try:
            backup_remote(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hypotheses backup skipped: %s", exc)


def active_hypotheses(data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = data if data is not None else load()
    return [h for h in (data.get("hypotheses") or []) if str(h.get("status")) == "active"]


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (text or "").lower()).strip("_")
    return (s or "h")[:28]


def normalize_hypothesis(
    raw: Any,
    *,
    classification: str = "",
    closes_now: int = 0,
    origin: str = "post_mortem",
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    effect = str(raw.get("effect") or "").strip().upper()
    effect = _EFFECT_ALIAS.get(effect, effect)
    if effect not in EFFECTS:
        return None
    when_setup = str(raw.get("when_setup") or raw.get("setup_tag") or "").strip().upper()
    if not when_setup or when_setup == "UNKNOWN":
        # try infer from when text
        blob = str(raw.get("when") or raw.get("reason") or "").upper()
        for tag in (
            "FAILED_BREAKOUT",
            "SWEEP",
            "BREAKOUT",
            "BREAKDOWN",
            "RETEST",
            "PULLBACK",
            "RECLAIM",
        ):
            if tag in blob:
                when_setup = tag
                break
    if not when_setup:
        return None
    cls = str(classification or raw.get("origin_classification") or "").upper()
    if cls == "GOOD_EDGE" and effect == "AVOID_SOFT":
        return None
    if cls == "PROCESS_ERROR" and effect == "AVOID_SOFT" and not raw.get("force_avoid"):
        # prefer checklist for process errors unless explicitly forced
        effect = "CHECKLIST"
    try:
        mag = float(raw.get("magnitude") or 1.0)
    except (TypeError, ValueError):
        mag = 1.0
    mag = max(0.25, min(3.0, mag))
    prefer = raw.get("prefer_tags") or []
    if isinstance(prefer, str):
        prefer = [prefer]
    prefer_tags = [str(x).upper()[:32] for x in prefer if str(x).strip()][:5]
    if effect == "PREFER_ALT" and not prefer_tags:
        prefer_tags = ["PULLBACK", "RETEST", "BREAKOUT", "SWEEP", "BREAKDOWN"]
    rid = str(raw.get("id") or f"h_{_slug(when_setup)}_{_slug(effect)}").strip()[:40]
    try:
        evidence = int(raw.get("evidence_n") or 1)
    except (TypeError, ValueError):
        evidence = 1
    return {
        "id": rid,
        "when_setup": when_setup[:32],
        "when_extra": str(raw.get("when_extra") or raw.get("when") or "")[:200],
        "effect": effect,
        "magnitude": round(mag, 2),
        "prefer_tags": prefer_tags,
        "reason": str(raw.get("reason") or "")[:240],
        "origin": origin,
        "origin_classification": cls,
        "status": "active",
        "created_at_closes": int(raw.get("created_at_closes") or closes_now or 0),
        "evidence_n": max(1, evidence),
        "eval_trades": int(raw.get("eval_trades") or 0),
        "pnl_attributed": float(raw.get("pnl_attributed") or 0),
        "blocks_count": int(raw.get("blocks_count") or 0),
        "entries_while_active": int(raw.get("entries_while_active") or 0),
        "overrides_count": int(raw.get("overrides_count") or 0),
        "last_verdict_at_closes": raw.get("last_verdict_at_closes"),
    }


def hypothesis_required(classification: str) -> bool:
    return str(classification or "").upper() in {
        "BAD_SETUP",
        "VALID_LOSS",
        "PROCESS_ERROR",
    }


def upsert_hypothesis(
    raw: Any,
    *,
    classification: str = "",
    closes_now: int = 0,
    origin: str = "post_mortem",
) -> dict[str, Any] | None:
    hyp = normalize_hypothesis(
        raw, classification=classification, closes_now=closes_now, origin=origin
    )
    if not hyp:
        return None
    data = load()
    hyps = list(data.get("hypotheses") or [])
    key = f"{hyp['when_setup']}|{hyp['effect']}".upper()
    found = None
    for h in hyps:
        if str(h.get("status")) != "active":
            continue
        k2 = f"{h.get('when_setup')}|{h.get('effect')}".upper()
        if k2 == key:
            found = h
            break
    step = float(getattr(config, "HYP_MAG_STEP", 0.25) or 0.25)
    if found:
        found["evidence_n"] = int(found.get("evidence_n") or 1) + 1
        found["magnitude"] = round(
            min(3.0, float(found.get("magnitude") or 1.0) + step), 2
        )
        if hyp.get("reason"):
            found["reason"] = hyp["reason"]
        if hyp.get("prefer_tags"):
            found["prefer_tags"] = hyp["prefer_tags"]
        found["origin_classification"] = classification or found.get(
            "origin_classification"
        )
        out = found
    else:
        hyps.append(hyp)
        out = hyp
    data["hypotheses"] = hyps
    save(data)
    return out


def migrate_from_adaptive_rules() -> int:
    """One-shot: convert adaptive_rules into hypotheses. Returns n migrated."""
    data = load()
    if data.get("migrated_from_adaptive") and active_hypotheses(data):
        return 0
    try:
        from modules import adaptive_rules

        rules_data = adaptive_rules.load_rules()
        rules = list(rules_data.get("rules") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("migrate adaptive failed: %s", exc)
        return 0
    n = 0
    for r in rules:
        effect = str(r.get("effect") or "").upper()
        effect = _EFFECT_ALIAS.get(effect, effect)
        if effect == "FORBID_SOFT":
            effect = "AVOID_SOFT"
        raw = {
            "id": r.get("id"),
            "when_setup": r.get("when_setup"),
            "when_extra": r.get("when"),
            "effect": effect,
            "magnitude": r.get("magnitude") or 1.0,
            "prefer_tags": r.get("prefer_tags"),
            "reason": r.get("reason"),
            "evidence_n": r.get("evidence_n") or 1,
            "created_at_closes": r.get("created_at_closes") or 0,
        }
        if upsert_hypothesis(raw, origin="migrate", closes_now=0):
            n += 1
    data = load()
    data["migrated_from_adaptive"] = True
    save(data)
    logger.info("Migrated %d adaptive rules → hypotheses", n)
    return n


def apply_to_decision(
    setup_tag: str,
    *,
    signal_level: int = 1,
    risk_reward: float | None = None,
    allow_l2_relief: bool = False,
) -> dict[str, Any]:
    """
    Apply active hypotheses to a candidate setup.
    Returns dict with extra_score, notes, prefer_tags, hyp_ids_applied,
    blocked_by_hyp, block_allowed (for AVOID hold), hold_reason.
    """
    tag = (setup_tag or "").upper()
    hyps = active_hypotheses()
    extra = 0.0
    notes: list[str] = []
    prefer: list[str] = []
    applied: list[str] = []
    blocked_id: str | None = None
    hold_reason: str | None = None
    hold_mag = float(getattr(config, "HYP_AVOID_HOLD_MAG", 2.0) or 2.0)
    hold_ev = int(getattr(config, "HYP_AVOID_HOLD_EVIDENCE", 3) or 3)
    override_bonus = float(getattr(config, "HYP_OVERRIDE_RR_BONUS", 0.5) or 0.5)
    min_rr = float(getattr(config, "MIN_RR", 1.2) or 1.2)

    for h in hyps:
        when = str(h.get("when_setup") or "").upper()
        if when not in {tag, "*"}:
            continue
        effect = str(h.get("effect") or "").upper()
        mag = float(h.get("magnitude") or 1.0)
        ev = int(h.get("evidence_n") or 1)
        hid = str(h.get("id") or "")
        applied.append(hid)
        if effect == "RAISE_SCORE":
            extra += mag
            notes.append(f"Hyp {hid}: RAISE_SCORE +{mag:.2f} ({when})")
        elif effect == "PREFER_ALT":
            extra += max(mag, 0.5)
            prefer.extend(list(h.get("prefer_tags") or []))
            notes.append(
                f"Hyp {hid}: PREFER_ALT {h.get('prefer_tags')} thay vì {when}"
            )
        elif effect == "CHECKLIST":
            notes.append(f"Hyp {hid}: CHECKLIST — {h.get('reason') or h.get('when_extra')}")
        elif effect == "AVOID_SOFT":
            extra += mag
            notes.append(f"Hyp {hid}: AVOID_SOFT +{mag:.2f} ev={ev} ({when})")
            # HOLD only with enough evidence+mag; idle relief does NOT skip this
            if mag >= hold_mag and ev >= hold_ev and when == tag:
                # Override path: L3 + strong RR
                can_override = (
                    signal_level >= 3
                    and risk_reward is not None
                    and float(risk_reward) >= min_rr + override_bonus
                )
                if can_override:
                    notes.append(
                        f"Hyp {hid}: override AVOID (L3 RR={risk_reward:.2f})"
                    )
                    h["overrides_count"] = int(h.get("overrides_count") or 0) + 1
                else:
                    blocked_id = hid
                    hold_reason = (
                        f"AI hypothesis {hid} AVOID {when} "
                        f"(mag={mag:.2f} ev={ev}) — đổi setup hoặc L3+RR"
                    )

    # Persist override counters touched above
    data = load()
    by_id = {str(h.get("id")): h for h in hyps}
    merged = []
    for h in data.get("hypotheses") or []:
        hid = str(h.get("id"))
        merged.append(by_id[hid] if hid in by_id else h)
    data["hypotheses"] = merged
    save(data, backup=False)

    seen: set[str] = set()
    prefer_out = []
    for t in prefer:
        tu = str(t).upper()
        if tu not in seen and tu != tag:
            seen.add(tu)
            prefer_out.append(tu)

    return {
        "extra_score": extra,
        "notes": notes,
        "prefer_tags": prefer_out,
        "hyp_ids_applied": applied,
        "blocked_by_hyp": blocked_id,
        "hold_reason": hold_reason,
    }


def record_block(hyp_id: str) -> None:
    data = load()
    for h in data.get("hypotheses") or []:
        if str(h.get("id")) == str(hyp_id) and str(h.get("status")) == "active":
            h["blocks_count"] = int(h.get("blocks_count") or 0) + 1
            break
    save(data, backup=False)


def credit_close(
    setup_tag: str,
    pnl: float,
    *,
    hyp_ids_at_entry: list[str] | None = None,
    closes_now: int = 0,
) -> list[str]:
    """Attribute close to matching active hyps; bump eval; maybe auto-verdict."""
    tag = (setup_tag or "").upper()
    data = load()
    touched: list[str] = []
    ids_entry = {str(x) for x in (hyp_ids_at_entry or [])}
    for h in data.get("hypotheses") or []:
        if str(h.get("status")) != "active":
            continue
        when = str(h.get("when_setup") or "").upper()
        hid = str(h.get("id") or "")
        if when != tag and hid not in ids_entry:
            continue
        h["eval_trades"] = int(h.get("eval_trades") or 0) + 1
        if when == tag:
            h["entries_while_active"] = int(h.get("entries_while_active") or 0) + 1
        try:
            h["pnl_attributed"] = float(h.get("pnl_attributed") or 0) + float(pnl)
        except (TypeError, ValueError):
            pass
        touched.append(hid)
    save(data, backup=False)
    verdicts = auto_eval(closes_now=closes_now)
    return touched + [f"verdict:{v}" for v in verdicts]


def auto_eval(*, closes_now: int = 0) -> list[str]:
    """Deterministic keep/decay/kill when eval_trades >= MIN."""
    min_e = int(getattr(config, "HYP_MIN_EVAL", 6) or 6)
    step = float(getattr(config, "HYP_MAG_STEP", 0.25) or 0.25)
    data = load()
    actions: list[str] = []
    for h in data.get("hypotheses") or []:
        if str(h.get("status")) != "active":
            continue
        ev_tr = int(h.get("eval_trades") or 0)
        if ev_tr < min_e:
            continue
        effect = str(h.get("effect") or "").upper()
        pnl = float(h.get("pnl_attributed") or 0)
        entries = int(h.get("entries_while_active") or 0)
        blocks = int(h.get("blocks_count") or 0)
        hid = str(h.get("id"))
        if effect == "AVOID_SOFT":
            if blocks >= 2 and entries <= max(2, blocks):
                # working: keep / slight evidence bump
                h["evidence_n"] = int(h.get("evidence_n") or 1) + 1
                actions.append(f"keep:{hid}")
            elif entries >= min_e and pnl < 0:
                # still entering losing tag — tighten mag slowly
                h["magnitude"] = round(
                    min(3.0, float(h.get("magnitude") or 1.0) + step), 2
                )
                actions.append(f"tighten:{hid}")
            elif entries >= min_e and pnl >= 0:
                # avoid may be wrong if tag profitable — decay
                h["status"] = "killed"
                h["kill_reason"] = "AVOID but tag pnl_attributed>=0"
                actions.append(f"kill:{hid}")
        elif effect == "PREFER_ALT":
            if entries >= min_e and pnl < 0:
                h["status"] = "killed"
                h["kill_reason"] = "PREFER path still losing"
                actions.append(f"kill:{hid}")
            else:
                actions.append(f"keep:{hid}")
        elif effect == "RAISE_SCORE":
            if entries >= min_e and pnl < 0 and blocks == 0:
                # raise not stopping bleed enough — keep raise (tighten)
                h["magnitude"] = round(
                    min(3.0, float(h.get("magnitude") or 1.0) + step), 2
                )
                actions.append(f"tighten:{hid}")
            else:
                actions.append(f"keep:{hid}")
        elif effect == "CHECKLIST":
            # keep unless never evidenced again — soft keep
            actions.append(f"keep:{hid}")
        h["last_verdict_at_closes"] = closes_now
        # reset eval window partially to avoid permanent lock
        if str(h.get("status")) == "active" and ev_tr >= min_e:
            h["eval_trades"] = 0
    save(data)
    return actions


def apply_meta_actions(actions: list[dict[str, Any]], *, closes_now: int = 0) -> list[str]:
    data = load()
    by_id = {str(h.get("id")): h for h in (data.get("hypotheses") or [])}
    done: list[str] = []
    for act in actions or []:
        if not isinstance(act, dict):
            continue
        hid = str(act.get("id") or "")
        verdict = str(act.get("verdict") or "").lower()
        h = by_id.get(hid)
        if not h:
            continue
        if verdict == "kill":
            h["status"] = "killed"
            h["kill_reason"] = str(act.get("reason") or "meta kill")[:160]
            done.append(f"meta_kill:{hid}")
        elif verdict == "revoke":
            h["status"] = "revoked"
            h["kill_reason"] = str(act.get("reason") or "meta revoke")[:160]
            done.append(f"meta_revoke:{hid}")
        elif verdict == "revise":
            fields = act.get("new_fields") or {}
            if isinstance(fields, dict):
                if "magnitude" in fields:
                    try:
                        h["magnitude"] = max(
                            0.25, min(3.0, float(fields["magnitude"]))
                        )
                    except (TypeError, ValueError):
                        pass
                if "prefer_tags" in fields and isinstance(fields["prefer_tags"], list):
                    h["prefer_tags"] = [
                        str(x).upper()[:32] for x in fields["prefer_tags"]
                    ][:5]
                if "effect" in fields:
                    eff = str(fields["effect"]).upper()
                    eff = _EFFECT_ALIAS.get(eff, eff)
                    if eff in EFFECTS:
                        h["effect"] = eff
            h["status"] = "active"
            done.append(f"meta_revise:{hid}")
        elif verdict == "keep":
            done.append(f"meta_keep:{hid}")
        h["last_verdict_at_closes"] = closes_now
    data["hypotheses"] = list(by_id.values())
    # also keep inactive not in by_id from original — by_id has all
    save(data)
    return done


def format_for_prompt(max_chars: int = 900) -> str:
    hyps = active_hypotheses()
    if not hyps:
        return "(chưa có hypothesis active — sau lệnh đóng AI sẽ đề xuất)"
    lines = ["ACTIVE HYPOTHESES (do AI tạo — phải tôn trọng khi chọn setup):"]
    for h in hyps[: config.HYP_MAX_ACTIVE]:
        lines.append(
            f"- [{h.get('id')}|{h.get('effect')}|setup={h.get('when_setup')}|"
            f"mag={h.get('magnitude')}|ev={h.get('evidence_n')}|"
            f"blocks={h.get('blocks_count')}|entries={h.get('entries_while_active')}] "
            f"{h.get('reason')}"
        )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def telegram_line(hyp: dict[str, Any] | None) -> str:
    if not hyp:
        return "hyp: (none)"
    return (
        f"hyp: {hyp.get('id')}|{hyp.get('effect')}|{hyp.get('when_setup')}|"
        f"ev={hyp.get('evidence_n')}|mag={hyp.get('magnitude')}"
    )


def _github_headers() -> dict[str, str] | None:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return None
    return {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "btc-dow-pa-agent-hyp",
    }


def backup_remote(data: dict[str, Any] | None = None) -> bool:
    headers = _github_headers()
    if not headers:
        return False
    data = data if data is not None else load()
    try:
        from modules.decision_history import (
            _ensure_history_branch,
            _github_contents_url,
        )

        repo = config.HISTORY_GITHUB_REPO
        path = config.HYPOTHESES_GITHUB_PATH
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
            "message": "chore: sync hypotheses for Blitz persistence",
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
        logger.warning("hypotheses backup failed: %s", exc)
        return False


def restore_from_remote() -> bool:
    path = _path()
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
            if cur.get("hypotheses"):
                return False
        except (OSError, json.JSONDecodeError):
            pass
    headers = _github_headers()
    if not headers:
        return False
    try:
        from modules.decision_history import _github_contents_url

        repo = config.HISTORY_GITHUB_REPO
        remote = config.HYPOTHESES_GITHUB_PATH
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
            "hypotheses restored n=%d", len(data.get("hypotheses") or [])
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("hypotheses restore failed: %s", exc)
        return False
