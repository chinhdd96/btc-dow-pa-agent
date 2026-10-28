"""Paper trade analytics: expectancy, quality-win, setup breakdown for prompts/gates."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)


def setup_tag_from_decision(decision: dict[str, Any] | None) -> str:
    """Normalize setup tag for stats/gates (FAILED_BREAKOUT, SWEEP, ...)."""
    if not decision:
        return "UNKNOWN"
    blob = " ".join(
        str(decision.get(k) or "")
        for k in ("state", "market_regime", "trigger", "structure", "reasoning")
    ).upper()
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
    regime = str(decision.get("market_regime") or "").upper()
    m = re.search(r"REGIME_[A-Z]+_([A-Z_]+)", regime)
    if m:
        return m.group(1)[:32]
    structure = str(decision.get("structure") or "UNKNOWN").upper()
    return structure[:32] or "UNKNOWN"


def setup_tag_from_trade(trade: dict[str, Any]) -> str:
    tag = str(trade.get("setup_tag") or "").strip().upper()
    # Strip REGIME_*_ prefix leftovers
    if tag.startswith("REGIME_"):
        m = re.search(r"REGIME_[A-Z]+_([A-Z_]+)$", tag)
        if m:
            tag = m.group(1)
        else:
            tag = tag.replace("REGIME_", "", 1)
    if tag and tag not in {"UNKNOWN", "RANGE", "UPTREND", "DOWNTREND", "SIDEWAY", "CHOP"}:
        return tag[:32]
    inferred = setup_tag_from_decision(
        {
            "market_regime": trade.get("regime"),
            "state": trade.get("state"),
            "trigger": trade.get("trigger"),
            "reasoning": trade.get("reasoning"),
            "structure": trade.get("structure"),
        }
    )
    return inferred if inferred else (tag[:32] if tag else "UNKNOWN")


def ensure_trade_setup_tag(trade: dict[str, Any]) -> str:
    """Normalize and write setup_tag onto trade dict; return tag."""
    tag = setup_tag_from_trade(trade)
    trade["setup_tag"] = tag
    return tag


def _r_multiple(trade: dict[str, Any]) -> float | None:
    """Signed R multiple from pnl and initial risk distance."""
    try:
        pnl = float(trade.get("pnl") or 0)
    except (TypeError, ValueError):
        return None
    initial_r = trade.get("initial_r")
    try:
        risk_u = float(initial_r) if initial_r is not None else 0.0
    except (TypeError, ValueError):
        risk_u = 0.0
    if risk_u <= 0:
        # Fallback: estimate R from entry/sl * qty
        try:
            entry = float(trade.get("entry") or 0)
            sl = float(trade.get("sl") or 0)
            qty = float(trade.get("quantity") or 0)
            risk_u = abs(entry - sl) * qty if entry and sl and qty else 0.0
        except (TypeError, ValueError):
            risk_u = 0.0
    if risk_u <= 0:
        return None
    return pnl / risk_u


def r_multiple_of_trade(trade: dict[str, Any]) -> float | None:
    return _r_multiple(trade)


def is_quality_win(trade: dict[str, Any]) -> bool:
    """WIN with meaningful profit (not BE/tiny trail scrape)."""
    if str(trade.get("result") or "").upper() != "WIN":
        return False
    reason = str(trade.get("reason") or "").upper()
    try:
        pnl = float(trade.get("pnl") or 0)
    except (TypeError, ValueError):
        pnl = 0.0
    if reason == "TP" and pnl > 0:
        return True
    if reason == "MANAGE_CLOSE" and pnl > 0:
        r = _r_multiple(trade)
        if r is not None:
            return r >= config.QUALITY_WIN_MIN_R
        return pnl > 0
    r = _r_multiple(trade)
    if r is not None:
        return r >= config.QUALITY_WIN_MIN_R
    return pnl > 0


def summarize_trades(
    trades: list[dict[str, Any]],
    *,
    rolling_n: int | None = None,
    with_setups: bool = True,
) -> dict[str, Any]:
    n_roll = rolling_n if rolling_n is not None else config.META_REVIEW_ROLLING_N
    window = list(trades[-n_roll:]) if n_roll > 0 else list(trades)
    n = len(window)
    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "quality_wins": 0,
            "quality_wr": 0.0,
            "pnl": 0.0,
            "expectancy": 0.0,
            "tp_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "by_setup": {},
            "stable": False,
        }

    wins = [t for t in window if str(t.get("result") or "").upper() == "WIN"]
    losses = [t for t in window if str(t.get("result") or "").upper() != "WIN"]
    q_wins = [t for t in window if is_quality_win(t)]
    tp_like = 0
    for t in window:
        reason = str(t.get("reason") or "").upper()
        if reason == "TP" or (reason == "MANAGE_CLOSE" and is_quality_win(t)):
            tp_like += 1

    pnls = []
    for t in window:
        try:
            pnls.append(float(t.get("pnl") or 0))
        except (TypeError, ValueError):
            pnls.append(0.0)
    total_pnl = sum(pnls)
    win_pnls = [float(t.get("pnl") or 0) for t in wins]
    loss_pnls = [abs(float(t.get("pnl") or 0)) for t in losses]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    wr = len(wins) / n
    expectancy = (avg_win * wr) - (avg_loss * (1 - wr))

    by_setup: dict[str, dict[str, Any]] = {}
    if with_setups:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in window:
            buckets[setup_tag_from_trade(t)].append(t)
        for tag, items in buckets.items():
            s = summarize_trades(items, rolling_n=len(items), with_setups=False)
            gross_win = sum(max(0.0, float(x.get("pnl") or 0)) for x in items)
            gross_loss = sum(abs(min(0.0, float(x.get("pnl") or 0))) for x in items)
            pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
            by_setup[tag] = {
                "n": s["n"],
                "wr": s["wr"],
                "quality_wr": s["quality_wr"],
                "pnl": s["pnl"],
                "expectancy": s["expectancy"],
                "tp_ratio": s["tp_ratio"],
                "profit_factor": round(pf, 3),
            }

    quality_wr = len(q_wins) / n
    tp_ratio = tp_like / n
    streak_tag, streak_n = consecutive_setup_streak(window)
    stable = (
        n >= min(20, max(n_roll, 1))
        and expectancy >= 0
        and total_pnl >= 0
        and quality_wr >= config.META_REVIEW_TARGET_QUALITY_WR
        and tp_ratio >= config.META_REVIEW_MIN_TP_RATIO
    )
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "wr": round(wr, 4),
        "quality_wins": len(q_wins),
        "quality_wr": round(quality_wr, 4),
        "pnl": round(total_pnl, 4),
        "expectancy": round(expectancy, 4),
        "tp_ratio": round(tp_ratio, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "by_setup": by_setup,
        "stable": stable,
        "streak_tag": streak_tag,
        "streak_n": streak_n,
    }


def consecutive_setup_streak(
    trades: list[dict[str, Any]],
) -> tuple[str, int]:
    """Return (tag, count) for consecutive same setup_tag at end of list."""
    if not trades:
        return "", 0
    last = setup_tag_from_trade(trades[-1])
    n = 0
    for t in reversed(trades):
        if setup_tag_from_trade(t) == last:
            n += 1
        else:
            break
    return last, n


def diversity_score_penalty(
    trades: list[dict[str, Any]],
    setup_tag: str,
) -> tuple[float, str]:
    """
    Extra score when repeating a bleeding setup streak.
    Returns (extra_penalty, note).
    """
    tag = (setup_tag or "").upper()
    if not tag or tag == "UNKNOWN":
        return 0.0, ""
    streak_tag, streak_n = consecutive_setup_streak(trades)
    need = int(getattr(config, "DIVERSITY_AFTER_STREAK", 3) or 3)
    if streak_n < need or streak_tag != tag:
        return 0.0, ""
    stats = setup_expectancy(trades, tag)
    # Bleed if overall tag exp < 0, or recent streak mostly losses
    recent = trades[-streak_n:]
    recent_pnl = sum(float(t.get("pnl") or 0) for t in recent)
    bleeding = stats.get("expectancy", 0) < 0 or recent_pnl < 0
    if not bleeding:
        return 0.0, ""
    pen = float(getattr(config, "SETUP_STREAK_PENALTY", 1.0) or 1.0)
    pen += float(getattr(config, "DIVERSITY_SCORE_BONUS", 1.0) or 1.0)
    note = (
        f"Diversity: streak {streak_tag}×{streak_n} đang bleed "
        f"(exp={stats.get('expectancy', 0):+.3f} streakPnL={recent_pnl:+.2f}) "
        f"→ +{pen:.1f} điểm — ưu tiên setup khác"
    )
    return pen, note


def close_line_for_telegram(
    trade: dict[str, Any],
    classification: str,
    trades: list[dict[str, Any]] | None = None,
) -> str:
    tag = ensure_trade_setup_tag(dict(trade))
    r = _r_multiple(trade)
    r_txt = f"{r:+.2f}R" if r is not None else "?R"
    exp_txt = ""
    if trades is not None:
        st = setup_expectancy(trades, tag)
        if st["n"]:
            exp_txt = f" | setup_exp={st['expectancy']:+.3f} n={st['n']}"
    return f"{tag} | {r_txt} | {classification or '?'}{exp_txt}"


def setup_expectancy(trades: list[dict[str, Any]], setup_tag: str) -> dict[str, Any]:
    tagged = [t for t in trades if setup_tag_from_trade(t) == setup_tag.upper()]
    return summarize_trades(tagged, rolling_n=len(tagged))


def should_block_setup(trades: list[dict[str, Any]], setup_tag: str) -> tuple[bool, str]:
    """Hard-block only when SETUP_BLOCK_HARD=true and expectancy < 0."""
    tag = setup_tag.upper()
    if tag in {"UNKNOWN", ""}:
        return False, ""
    if not getattr(config, "SETUP_BLOCK_HARD", False):
        return False, ""
    stats = setup_expectancy(trades, tag)
    if stats["n"] < config.SETUP_BLOCK_MIN_TRADES:
        return False, ""
    if stats["expectancy"] < 0:
        return (
            True,
            (
                f"Chặn setup {tag}: expectancy={stats['expectancy']:.3f} "
                f"pnl={stats['pnl']:+.2f} trên {stats['n']} lệnh"
            ),
        )
    return False, ""


def setup_score_penalty(trades: list[dict[str, Any]], setup_tag: str) -> tuple[float, str]:
    """
    Soft gate: expectancy âm + đủ mẫu → nâng ngưỡng điểm, KHÔNG cấm vào lệnh.
    Trả (penalty_score, note).
    """
    tag = setup_tag.upper()
    if tag in {"UNKNOWN", ""}:
        return 0.0, ""
    stats = setup_expectancy(trades, tag)
    if stats["n"] < config.SETUP_BLOCK_MIN_TRADES:
        return 0.0, ""
    if stats["expectancy"] < 0:
        pen = float(getattr(config, "SETUP_SOFT_PENALTY", 0.5) or 0.5)
        note = (
            f"Soft: {tag} exp={stats['expectancy']:.3f} "
            f"pnl={stats['pnl']:+.2f} n={stats['n']} → +{pen:.1f} điểm"
        )
        return pen, note
    return 0.0, ""


def is_chop_failed_breakout(decision: dict[str, Any], primary_bias: str = "") -> bool:
    bias = str(primary_bias or decision.get("primary_bias") or "").upper()
    regime = str(decision.get("market_regime") or "").upper()
    structure = str(decision.get("structure") or "").upper()
    state = str(decision.get("state") or "").upper()
    tag = setup_tag_from_decision(decision)
    chopish = any(
        x in bias or x in regime or x in structure
        for x in ("SIDEWAY", "CHOP", "RANGE")
    )
    fb = tag in {"FAILED_BREAKOUT", "SWEEP"} or "FAILED_BREAKOUT" in state or "SWEEP" in state
    return chopish and fb


def effective_min_rr(decision: dict[str, Any], primary_bias: str = "") -> float:
    if is_chop_failed_breakout(decision, primary_bias):
        return max(config.MIN_RR, config.CHOP_MIN_RR)
    return config.MIN_RR


def format_stats_for_prompt(
    trades: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
) -> str:
    budget = max_chars if max_chars is not None else config.TRADE_STATS_MAX_CHARS
    s = summarize_trades(trades)
    if s["n"] == 0:
        return "(chưa có lệnh đóng — chưa có TRADE_STATS)"
    streak_tag = s.get("streak_tag") or ""
    streak_n = int(s.get("streak_n") or 0)
    lines = [
        (
            f"Rolling {s['n']}: PnL={s['pnl']:+.2f}u | expectancy={s['expectancy']:+.3f}/lệnh | "
            f"WR={s['wr']*100:.0f}% | qualityWR={s['quality_wr']*100:.0f}% "
            f"(≥{config.QUALITY_WIN_MIN_R}R/TP) | TP-like={s['tp_ratio']*100:.0f}% | "
            f"avgW={s['avg_win']:+.2f} avgL=-{s['avg_loss']:.2f}"
        ),
        (
            f"streak={streak_tag or '-'}×{streak_n} | "
            f"stable_mode={'YES' if s['stable'] else 'NO'} — "
            "ưu tiên expectancy/PnL; setup âm/streak bleed → đổi setup/TF, "
            "vẫn được trade L2+ (không đứng ngoài mãi)."
        ),
    ]
    setups = sorted(
        s["by_setup"].items(),
        key=lambda kv: kv[1].get("n", 0),
        reverse=True,
    )[:5]
    for tag, st in setups:
        lines.append(
            f"  {tag}: n={st['n']} exp={st['expectancy']:+.3f} "
            f"pnl={st['pnl']:+.2f} PF={st.get('profit_factor', 0):.2f} "
            f"qWR={st['quality_wr']*100:.0f}% TP={st.get('tp_ratio', 0)*100:.0f}%"
        )
    text = "\n".join(lines)
    if len(text) > budget:
        text = text[: budget - 1] + "…"
    return text


def hours_since_iso(value: Any) -> float | None:
    if not value:
        return None
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except (TypeError, ValueError, OSError):
        return None
