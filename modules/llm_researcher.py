"""LLM as researcher / AI review / paper post-mortem — never entry authority."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)


def _llm_chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.5,
    timeout: int = 90,
) -> dict | list | None:
    if not config.LLM_API_KEY:
        logger.info("LLM disabled: missing LLM_API_KEY")
        return None
    try:
        r = requests.post(
            config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": config.LLM_MODEL,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": config.LLM_MAX_COMPLETION_TOKENS,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


def _load_research_context(max_chars: int = 2500) -> dict[str, str]:
    daily = ""
    news = ""
    try:
        p = Path(config.PATHS["DAILY_PLAYBOOK"])
        if p.exists():
            daily = p.read_text(encoding="utf-8")[: config.DAILY_PLAYBOOK_MAX_CHARS]
    except OSError:
        pass
    try:
        p = Path(config.PATHS["CONTEXT_NEWS"])
        if p.exists():
            news = p.read_text(encoding="utf-8")[: config.NEWS_SUMMARY_MAX_CHARS]
    except OSError:
        pass
    blob = (daily + "\n" + news).strip()
    return {
        "daily_playbook": daily,
        "news_context": news,
        "combined": blob[:max_chars],
    }


def _family_rules() -> list[str]:
    return [
        "only catalog indicators and allowed params",
        f"the family uses {config.FAMILY_MIN_INDICATORS}-{config.FAMILY_MAX_INDICATORS} "
        "DISTINCT indicator types in total (ema12 and ema50 count as one 'ema'; "
        "donchian_high/low count as one; raw close is free; mtf_filter=true counts as one)",
        "give each indicator a role: 1 entry trigger, 1 trend/regime filter, "
        "1 volatility or volume filter, optionally 1-2 confirmations",
        "entry_long and entry_short should mirror each other",
        "never propose empty entries or no-trade philosophies",
    ]


_EXIT_OPTIONS = (
    "exit keys: atr_stop, reward_r, leverage; optional (0/false = off): breakeven_r = move SL "
    "to entry after +xR; trail_atr = trailing stop at close - k*ATR; max_bars = close after "
    "N 15m bars; exit_on_opposite = close when the opposite entry fires"
)

_FEEDBACK_RULE = (
    "previous_attempts (if any) are candidates from THIS research run that already FAILED the "
    "out-of-sample walk-forward test, with their metrics and verdict. Do not resubmit them or "
    "near-copies; fix the failure: too_few_trades → looser thresholds or a trigger that fires "
    "more often; losing / low profit factor → different trigger or regime filter; "
    "high_drawdown → tighter stop, lower leverage, trend filter."
)

_SPEC_EXAMPLE = {
    "name": "trend_pullback",
    "entry_long": [
        {"ind": "rsi", "params": {"period": 14}, "op": "cross_up", "value": 40},
        {"ind": "dist_to_ema", "params": {"period": 200}, "op": "gt", "value": 0},
    ],
    "entry_short": [
        {"ind": "rsi", "params": {"period": 14}, "op": "cross_down", "value": 60},
        {"ind": "dist_to_ema", "params": {"period": 200}, "op": "lt", "value": 0},
    ],
    "filters": [
        {"ind": "adx", "params": {"period": 14}, "op": "gte", "value": 18},
        {"ind": "atr_pct", "params": {"period": 14}, "op": "lte", "value": 0.03},
    ],
    "exit": {"atr_period": 14, "atr_stop": 1.5, "reward_r": 2.0, "leverage": 2, "breakeven_r": 1.0},
    "mtf_filter": False,
}


def propose_family_drafts(
    market_summary: dict,
    count: int,
    *,
    research_focus: list[str] | None = None,
    tick_history_text: str | None = None,
    learning: dict | None = None,
    feedback: list[dict] | None = None,
) -> list[dict]:
    """Genesis: drafts of the bot's single indicator family; walk-forward picks v1."""
    if not config.LLM_RESEARCH_ENABLED:
        return []
    from modules.indicator_catalog import catalog_for_prompt
    from modules.strategy_spec import family_shape_ok, normalize_spec

    learning = learning or {}
    prompt = {
        "task": (
            "MISSION: positive expectancy / PnL after fees on BTCUSDT perpetual paper trading, 15m. "
            "Design the bot's ONE indicator family. It will be the only strategy: later research "
            "rounds will edit it (add/remove/replace/tune indicators) version by version. "
            f"Give {count} alternative drafts with DIFFERENT core ideas (trend, pullback, breakout, "
            "mean-revert, order flow...) so out-of-sample testing can pick the best as v1. "
            "LEARN FROM HISTORY: research_memory lists Specs already evaluated out-of-sample and "
            "real paper results — build on indicators that qualified, avoid the worst ones. "
            "Name each draft with a short snake_case family name. "
            + _FEEDBACK_RULE
            + " JSON only."
        ),
        "mission": "pnl_positive_priority_accept_controlled_risk",
        "previous_attempts": feedback or [],
        "market_summary": market_summary,
        "research_focus": research_focus or [],
        "research_memory": learning.get("research_memory") or "",
        "lessons": learning.get("lessons") or "",
        "recent_trades": learning.get("recent_trades") or [],
        "market_notes": _load_research_context()["combined"],
        "tick_history": (tick_history_text or "")[: config.TICK_HISTORY_PROMPT_MAX_CHARS],
        "catalog": catalog_for_prompt(),
        "leverage_candidates": list(config.LEVERAGE_CANDIDATES),
        "exit_options": _EXIT_OPTIONS,
        "requirements": _family_rules(),
        "output_schema": {"specs": [_SPEC_EXAMPLE]},
    }
    obj = _llm_chat(
        "You design indicator-based futures strategies for paper research. "
        "Goal is positive PnL expectancy; accept risk within leverage bounds. "
        "Never advise perpetual flat.",
        json.dumps(prompt, default=str),
        temperature=0.75,
    )
    if not isinstance(obj, dict):
        return []
    out: list[dict] = []
    for item in (obj.get("specs") or [])[:count]:
        if not isinstance(item, dict):
            continue
        s = normalize_spec(item)
        ok, msg = family_shape_ok(s)
        if not ok:
            logger.info("LLM draft %s rejected: %s", s.get("name"), msg)
            continue
        s["variant"] = {"source": "llm_genesis", "reason": str(item.get("reason") or "")[:300]}
        out.append(s)
    logger.info("LLM family drafts validated=%d", len(out))
    return out


def propose_family_changes(
    current_spec: dict,
    *,
    mode: str,
    count: int,
    family_history: list[dict] | None = None,
    paper_stats: dict | None = None,
    research_focus: list[str] | None = None,
    tick_history_text: str | None = None,
    learning: dict | None = None,
    feedback: list[dict] | None = None,
) -> list[dict]:
    """Edits to the family. Returns [{"ops": [...], "reason": str}] — Python tests each one."""
    if not config.LLM_RESEARCH_ENABLED:
        return []
    from modules.indicator_catalog import catalog_for_prompt
    from modules.strategy_spec import indicator_types

    learning = learning or {}
    if mode == "no_trade":
        scope = (
            "NO_TRADE: the current version has not entered a single trade for "
            f"{(paper_stats or {}).get('hours_without_entry')}h on live candles — it is too strict "
            "for the current market. Each variant may use up to 3 ops: loosen the thresholds that "
            "block entries (see research_focus / tick_history skip reasons), remove or replace the "
            "strictest filter, or replace the trigger with one that fires more often. A variant is "
            "only accepted if it trades clearly more often out-of-sample while staying profitable."
        )
    elif mode == "rework":
        scope = (
            "REWORK: the current version keeps failing. Each variant may use up to 4 ops and may "
            "REPLACE the core entry trigger indicator, change the trend/volatility filters and exits."
        )
    else:
        scope = (
            "PATCH: keep the core entry idea. Each variant uses 1-2 ops: add, remove, replace or "
            "tune ONE indicator condition, or adjust exits."
        )
    prompt = {
        "task": (
            "MISSION: improve paper PnL / expectancy of the bot's single indicator family. "
            f"Propose {count} alternative variants of the current version. {scope} "
            "Base every variant on evidence: family_history shows what each past edit changed and "
            "how that version did (backtest + real paper) — repeat edits that helped, do not "
            "repeat edits that hurt. Compare indicator values and regime at entry for winners vs "
            "losers in recent_trades. Explain the evidence in reason. "
            + _FEEDBACK_RULE
            + " Do NOT suggest stopping trading. JSON only."
        ),
        "mode": mode,
        "previous_attempts": feedback or [],
        "current_spec": current_spec,
        "current_indicator_types": indicator_types(current_spec),
        "family_history": family_history or [],
        "paper_stats": paper_stats or {},
        "research_focus": research_focus or [],
        "recent_trades": learning.get("recent_trades") or [],
        "lessons": learning.get("lessons") or "",
        "research_memory": learning.get("research_memory") or "",
        "tick_history": (tick_history_text or "")[:1200],
        "catalog": catalog_for_prompt(),
        "leverage_candidates": list(config.LEVERAGE_CANDIDATES),
        "exit_options": _EXIT_OPTIONS,
        "requirements": _family_rules(),
        "ops_reference": [
            {"action": "add", "target": "entry_long|entry_short|filters", "condition": {"ind": "adx", "params": {"period": 14}, "op": "gte", "value": 20}},
            {"action": "remove", "target": "filters", "index": 0},
            {"action": "replace", "target": "entry_long", "index": 0, "condition": {"ind": "stoch_k", "params": {"period": 14}, "op": "cross_up", "value": 20}},
            {"action": "tune", "target": "filters", "index": 1, "params": {"period": 20}, "value": 1.2},
            {"action": "exit", "exit": {"atr_stop": 1.8, "reward_r": 2.5, "trail_atr": 2.0}},
            {"action": "mtf", "value": True},
        ],
        "output_schema": {
            "variants": [
                {
                    "reason": "losers cluster in range regime (adx<18) → add adx trend filter",
                    "ops": [
                        {"action": "add", "target": "filters", "condition": {"ind": "adx", "params": {"period": 14}, "op": "gte", "value": 18}}
                    ],
                }
            ]
        },
    }
    obj = _llm_chat(
        "You improve one indicator strategy family for better expectancy. "
        "Never recommend flat forever.",
        json.dumps(prompt, default=str),
        temperature=0.5 if mode == "patch" else 0.6,
    )
    if not isinstance(obj, dict):
        return []
    out = []
    for v in (obj.get("variants") or [])[:count]:
        if isinstance(v, dict) and isinstance(v.get("ops"), list) and v["ops"]:
            out.append({"ops": v["ops"][:6], "reason": str(v.get("reason") or "")[:300]})
    logger.info("LLM family %s variants n=%d", mode, len(out))
    return out


def review_research_report(report_lite: dict) -> dict[str, Any]:
    """AI Review of research metrics — does NOT promote."""
    if not config.LLM_RESEARCH_ENABLED:
        return {
            "summary": "LLM review skipped",
            "failure_modes": [],
            "next_research_focus": [],
        }
    prompt = {
        "task": (
            "Review these BTCUSDT strategy research results. "
            "Explain failures, overfitting risks, and suggest next research focus. "
            "Use tick_history (if present) only to explain paper flatness / filter blocks — "
            "not to pick live entries. Do NOT claim ready for live trading. JSON only."
        ),
        "report": report_lite,
        "output_schema": {
            "summary": "string",
            "failure_modes": ["string"],
            "next_research_focus": ["string"],
        },
    }
    obj = _llm_chat(
        "You are a skeptical quant reviewer. Be concrete and falsifiable.",
        json.dumps(prompt, default=str),
        temperature=0.3,
    )
    if not isinstance(obj, dict):
        return {
            "summary": "LLM review unavailable",
            "failure_modes": [],
            "next_research_focus": [],
        }
    return {
        "summary": str(obj.get("summary") or "")[:2000],
        "failure_modes": [
            str(x)[:300] for x in (obj.get("failure_modes") or [])[:8]
        ],
        "next_research_focus": [
            str(x)[:300] for x in (obj.get("next_research_focus") or [])[:8]
        ],
    }


def post_mortem_paper_trade(
    trade: dict,
    *,
    spec: dict | None = None,
    recent_trades: list[dict] | None = None,
) -> dict[str, Any]:
    """Analyze one closed paper trade; may set decay_signal + research_hints."""
    default = {
        "classification": "INSUFFICIENT_EVIDENCE",
        "lesson": "",
        "actionable_rule": "",
        "decay_signal": False,
        "research_hints": [],
    }
    if not config.LLM_RESEARCH_ENABLED or not config.LLM_API_KEY:
        pnl = float(trade.get("pnl") or 0)
        default["classification"] = "VALID_LOSS" if pnl < 0 else "GOOD_EDGE"
        default["lesson"] = f"Paper close {trade.get('exit_reason')} pnl={pnl:+.4f}"
        return default

    from modules import tick_history

    slim = {
        k: trade.get(k)
        for k in (
            "side", "entry", "exit", "entry_time", "exit_time", "exit_reason", "exit_source",
            "pnl", "r_multiple", "bars_held", "leverage", "sl", "initial_sl", "tp",
            "entry_context", "family",
        )
    }
    prompt = {
        "task": (
            "Post-mortem a PAPER BTCUSDT trade taken by the Champion Strategy Spec. "
            "entry_context has the indicator values and market regime when the signal fired. "
            "Compare with recent_trades (same structure) to find a pattern. "
            "Classify outcome, write a short lesson, an actionable_rule phrased as an "
            "indicator condition (e.g. 'add adx(14) >= 22 filter for LONG'), say if the "
            "strategy may be decaying, and give up to 3 short research_hints for the next "
            "Spec search. tick_history is context only. JSON only."
        ),
        "trade": slim,
        "spec": spec,
        "recent_trades": recent_trades or [],
        "tick_history": tick_history.format_for_prompt(),
        "output_schema": {
            "classification": "GOOD_EDGE|VALID_LOSS|BAD_SETUP|PROCESS_ERROR|INSUFFICIENT_EVIDENCE",
            "lesson": "string Vietnamese or English",
            "actionable_rule": "string",
            "decay_signal": False,
            "research_hints": ["string"],
        },
    }
    obj = _llm_chat(
        "You analyze paper trades for research. Never recommend live orders.",
        json.dumps(prompt, default=str),
        temperature=0.4,
    )
    if not isinstance(obj, dict):
        return default
    hints = [str(h)[:200] for h in (obj.get("research_hints") or [])[:3] if h]
    cls = str(obj.get("classification") or "INSUFFICIENT_EVIDENCE").upper()
    return {
        "classification": cls,
        "lesson": str(obj.get("lesson") or "")[:800],
        "actionable_rule": str(obj.get("actionable_rule") or "")[:300],
        "decay_signal": bool(obj.get("decay_signal")),
        "research_hints": hints,
    }
