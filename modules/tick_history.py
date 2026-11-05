"""Short Champion tick history for research context (not AI entry authority)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from modules.genome import Genome, genome_from_dict

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


def classify_why(why: str) -> str:
    w = (why or "").lower()
    if "chưa có champion" in w or "no champion" in w:
        return "no_champion"
    if "warm-up" in w or "thiếu indicator" in w:
        return "warmup"
    if "vol_ratio" in w or "volf" in w:
        return "volume_filter"
    if "atr%" in w or "max atr" in w or "maxatr" in w:
        return "max_atr"
    if "mtf" in w:
        return "mtf"
    if "vol mode" in w:
        return "volatility_mode"
    if "chưa đủ điều kiện setup" in w or "family=" in w:
        return "setup"
    if "pending" in w or "chờ fill" in w:
        return "pending"
    if "đang long" in w or "đang short" in w or "chờ sl" in w:
        return "in_position"
    return "other"


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
    why = _one_line(str(report.get("why") or ""), 120)
    reason = classify_why(why)
    candle = str(report.get("candle") or "")[-19:]
    close = report.get("close")
    close_s = f"{float(close):.1f}" if close is not None else "—"
    family = report.get("family") or "—"
    ts = _iso_now()
    summary = (
        f"{ts} | {candle} | {action} | {family} | {reason} | px={close_s} | {why}"
    )
    record = {
        "ts": ts,
        "candle": report.get("candle"),
        "action": action,
        "family": family,
        "genome_id": report.get("genome_id"),
        "close": close,
        "why": why,
        "reason": reason,
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


def starvation_report(n: int | None = None) -> dict[str, Any]:
    """Detect chronic SKIP from filters — signal for early research, not live override."""
    entries = load_recent(n or max(config.STARVE_MIN_TICKS, 24))
    empty = {
        "n": 0,
        "skip_ratio": 0.0,
        "reason_counts": {},
        "top_reasons": [],
        "starving": False,
        "should_reresearch": False,
        "focus": [],
        "summary": "(no ticks)",
    }
    if not entries:
        return empty

    active = [
        e
        for e in entries
        if classify_why(str(e.get("why") or e.get("reason") or ""))
        not in {"warmup", "in_position", "pending"}
    ]
    sample = active if active else entries
    reasons = Counter(
        str(e.get("reason") or classify_why(str(e.get("why") or ""))) for e in sample
    )
    skip_n = sum(1 for e in sample if str(e.get("action") or "").upper() == "SKIP")
    n_sample = len(sample)
    skip_ratio = skip_n / n_sample if n_sample else 0.0
    top = reasons.most_common(5)
    filter_blocks = {"volume_filter", "max_atr", "mtf", "volatility_mode", "setup"}
    filter_skip_n = sum(
        1
        for e in sample
        if str(e.get("action") or "").upper() == "SKIP"
        and str(e.get("reason") or classify_why(str(e.get("why") or "")))
        in filter_blocks
    )
    filter_ratio = filter_skip_n / n_sample if n_sample else 0.0

    focus: list[str] = []
    if reasons.get("volume_filter", 0) >= max(3, n_sample // 5):
        focus.append("Champion bị chặn volume_filter — thử volume_filter thấp hơn (0–1.2)")
    if reasons.get("max_atr", 0) >= max(3, n_sample // 5):
        focus.append("Champion bị chặn max_atr_pct — thử max_atr_pct cao hơn (0.04–0.08)")
    if reasons.get("mtf", 0) >= max(3, n_sample // 5):
        focus.append("Champion bị chặn MTF — thử mtf_filter=false hoặc family khác")
    if reasons.get("setup", 0) >= max(3, n_sample // 4):
        focus.append(
            "Family hiện tại ít setup trên TF live — thử family đa dạng hơn "
            "(breakout/momentum/trend_cross)"
        )
    if reasons.get("no_champion", 0) >= 1:
        focus.append("Chưa có Champion — ưu tiên soft-qualify candidate có trade count đủ")

    starving = (
        n_sample >= config.STARVE_MIN_TICKS
        and skip_ratio >= config.STARVE_SKIP_RATIO
        and filter_ratio >= 0.5
    )
    summary = (
        f"ticks={n_sample} skip_ratio={skip_ratio:.2f} filter_ratio={filter_ratio:.2f} "
        f"top={top[:3]} starving={starving}"
    )
    return {
        "n": n_sample,
        "skip_ratio": round(skip_ratio, 3),
        "filter_ratio": round(filter_ratio, 3),
        "reason_counts": dict(reasons),
        "top_reasons": top,
        "starving": starving,
        "should_reresearch": bool(starving and config.STARVE_TRIGGER_RESEARCH),
        "focus": focus,
        "summary": summary,
    }


def challenger_genomes_from_starvation(
    champion: dict | None,
    *,
    report: dict[str, Any] | None = None,
) -> list[Genome]:
    """Python-seeded looser variants of current Champion — research tests them, no live override."""
    if not champion:
        return []
    g = genome_from_dict(champion)
    if not g:
        try:
            g = Genome(**{k: champion[k] for k in Genome.__dataclass_fields__ if k in champion})
        except Exception:
            return []
    diag = report or starvation_report()
    reasons = set(diag.get("reason_counts") or {})
    out: list[Genome] = []

    def _add(patch: dict) -> None:
        d = asdict(g)
        d.update(patch)
        cand = genome_from_dict(d)
        if cand and cand.genome_id != g.genome_id:
            out.append(cand)

    if "volume_filter" in reasons or diag.get("starving"):
        for vf in (1.2, 1.0, 0.0):
            if vf < float(g.volume_filter or 0):
                _add({"volume_filter": vf})
    if "max_atr" in reasons or diag.get("starving"):
        for mx in (0.04, 0.06, 0.08):
            if mx > float(g.max_atr_pct or 0):
                _add({"max_atr_pct": mx})
    if "mtf" in reasons and g.mtf_filter:
        _add({"mtf_filter": False})
    if "setup" in reasons or diag.get("starving"):
        for fam in ("breakout", "momentum", "trend_cross", "volatility_breakout"):
            if fam != g.family:
                _add({"family": fam, "volume_filter": min(float(g.volume_filter or 0), 1.0)})
    if "volatility_mode" in reasons and g.volatility_mode != "any":
        _add({"volatility_mode": "any"})

    # Deduplicate by genome_id
    seen: set[str] = set()
    uniq: list[Genome] = []
    for cand in out:
        if cand.genome_id in seen:
            continue
        seen.add(cand.genome_id)
        uniq.append(cand)
    return uniq[:12]


def maybe_flag_research_due(state: dict[str, Any]) -> dict[str, Any] | None:
    """If starving, set research_due (with cooldown). Returns diagnosis or None."""
    if not config.STARVE_TRIGGER_RESEARCH:
        return None
    diag = starvation_report()
    if not diag.get("should_reresearch"):
        return None
    import time

    last = float(state.get("last_starve_research_ts") or 0)
    if time.time() - last < config.STARVE_COOLDOWN_HOURS * 3600:
        logger.info(
            "Starvation detected but cooldown active (%.1fh): %s",
            config.STARVE_COOLDOWN_HOURS,
            diag.get("summary"),
        )
        return diag
    state["research_due"] = True
    state["starve_focus"] = diag.get("focus") or []
    state["last_starve_research_ts"] = time.time()
    logger.warning("Champion starving → research_due | %s", diag.get("summary"))
    return diag
