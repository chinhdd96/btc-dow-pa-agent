"""LLM as researcher / AI review / paper post-mortem — never entry authority."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

import config
from modules.genome import FAMILIES, Genome, genome_from_dict

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


def propose_genomes(
    market_summary: dict,
    count: int = 12,
    *,
    research_focus: list[str] | None = None,
    tick_history_text: str | None = None,
    starvation: dict | None = None,
) -> list[Genome]:
    if not config.LLM_RESEARCH_ENABLED:
        logger.info("LLM researcher disabled via LLM_RESEARCH_ENABLED")
        return []
    allowed = {
        "family": list(FAMILIES),
        "fast": [5, 8, 10, 12, 16, 20, 24, 30],
        "slow": [30, 40, 50, 75, 100, 150, 200],
        "rsi_low": [20, 25, 30, 35, 40],
        "rsi_high": [60, 65, 70, 75, 80],
        "breakout_n": [10, 15, 20, 30, 40, 55],
        "atr_stop": [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "reward_r": [1.2, 1.5, 1.8, 2, 2.5, 3, 4],
        "volume_filter": [0, 1, 1.2, 1.5],
        "trend_filter": [True, False],
        "max_atr_pct": [0.025, 0.04, 0.06, 0.08],
        "leverage": list(config.LEVERAGE_CANDIDATES),
        "funding_abs_max": [0.0002, 0.0005, 0.001, 0.002],
        "oi_change_min": [0, 0.005, 0.01, 0.02],
        "volatility_mode": ["any", "high", "low"],
        "mtf_filter": [True, False],
    }
    ctx = _load_research_context()
    starve = starvation or {}
    prompt = {
        "task": (
            "Propose diverse, testable BTCUSDT perpetual futures strategy hypotheses. "
            "Do not claim profitability. Return JSON only. "
            "tick_history / starvation diagnose WHY the current Champion stayed flat "
            "(volume_filter, max_atr, mtf, setup). Respond by proposing NEW genomes with "
            "looser or different parameters — do NOT suggest live filter overrides, "
            "and do NOT invent per-candle BUY/SELL instructions."
        ),
        "market_summary": market_summary,
        "research_focus": research_focus or [],
        "starvation": {
            "starving": bool(starve.get("starving")),
            "skip_ratio": starve.get("skip_ratio"),
            "top_reasons": starve.get("top_reasons"),
            "focus": starve.get("focus"),
            "summary": starve.get("summary"),
        },
        "market_notes": ctx["combined"],
        "tick_history": (tick_history_text or "")[: config.TICK_HISTORY_PROMPT_MAX_CHARS],
        "allowed_values": allowed,
        "requirements": [
            "diverse families and parameter combinations",
            "avoid lookahead",
            "prefer interpretable hypotheses",
            "do not optimize only for recent noise",
            "use only allowed_values",
            "if starving / chronic SKIP: majority of proposals must address top_reasons "
            "(lower volume_filter, higher max_atr_pct, mtf_filter=false, or other families)",
        ],
        "output_schema": {
            "proposals": [
                {
                    "family": "one allowed family",
                    "fast": 12,
                    "slow": 50,
                    "rsi_low": 30,
                    "rsi_high": 70,
                    "breakout_n": 20,
                    "atr_stop": 2.0,
                    "reward_r": 2.0,
                    "volume_filter": 0,
                    "trend_filter": True,
                    "max_atr_pct": 0.06,
                    "leverage": 2,
                    "funding_abs_max": 0.001,
                    "oi_change_min": 0.005,
                    "volatility_mode": "any",
                    "mtf_filter": False,
                }
            ]
        },
    }
    obj = _llm_chat(
        "You are a quantitative research scientist. "
        "Never assert a strategy is profitable without out-of-sample evidence.",
        json.dumps(prompt),
        temperature=0.8,
    )
    if not isinstance(obj, dict):
        return []
    proposals: list[Genome] = []
    for item in obj.get("proposals", [])[:count]:
        if not isinstance(item, dict):
            continue
        g = genome_from_dict(item)
        if g:
            proposals.append(g)
    logger.info("LLM researcher returned %d validated strategy proposals", len(proposals))
    return proposals


def propose_specs(
    market_summary: dict,
    count: int = 8,
    *,
    research_focus: list[str] | None = None,
    tick_history_text: str | None = None,
    starvation: dict | None = None,
    learning: dict | None = None,
) -> list[dict]:
    """LLM proposes Strategy Specs from indicator catalog (PnL-first, no forever-flat)."""
    if not config.LLM_RESEARCH_ENABLED:
        return []
    from modules.indicator_catalog import catalog_for_prompt
    from modules.strategy_spec import normalize_spec, validate_spec

    ctx = _load_research_context()
    starve = starvation or {}
    learning = learning or {}
    prompt = {
        "task": (
            "MISSION: positive expectancy / PnL after fees on BTCUSDT paper. "
            "Propose Strategy Specs by COMBINING catalog indicators (classic methods OK). "
            "LEARN FROM HISTORY: research_memory lists Specs already evaluated out-of-sample "
            "(best and worst) and real paper results; do NOT re-propose the worst ones or "
            "near-copies, build on indicators that qualified. recent_trades show the "
            "indicator values and regime at entry for real paper trades — use the "
            "win/loss patterns (e.g. losers cluster in range regime → add a trend filter). "
            "Do NOT recommend standing aside forever or HOLD-only strategies. "
            "If starving (chronic SKIP), propose LOOSER filters and alternate indicators "
            "so the system can take learnable trades. JSON only."
        ),
        "mission": "pnl_positive_priority_accept_controlled_risk",
        "market_summary": market_summary,
        "research_focus": research_focus or [],
        "starvation": starve,
        "research_memory": learning.get("research_memory") or "",
        "lessons": learning.get("lessons") or "",
        "recent_trades": learning.get("recent_trades") or [],
        "market_notes": ctx["combined"],
        "tick_history": (tick_history_text or "")[: config.TICK_HISTORY_PROMPT_MAX_CHARS],
        "catalog": catalog_for_prompt(),
        "leverage_candidates": list(config.LEVERAGE_CANDIDATES),
        "output_schema": {
            "specs": [
                {
                    "name": "rsi_ema_example",
                    "entry_long": [
                        {
                            "ind": "rsi",
                            "params": {"period": 14},
                            "op": "lt",
                            "value": 35,
                        }
                    ],
                    "entry_short": [
                        {
                            "ind": "rsi",
                            "params": {"period": 14},
                            "op": "gt",
                            "value": 65,
                        }
                    ],
                    "filters": [
                        {
                            "ind": "vol_ratio",
                            "params": {"period": 20},
                            "op": "gte",
                            "value": 1.0,
                        }
                    ],
                    "exit": {
                        "atr_period": 14,
                        "atr_stop": 1.5,
                        "reward_r": 2.0,
                        "leverage": 2,
                        "breakeven_r": 1.0,
                        "trail_atr": 0,
                        "max_bars": 0,
                        "exit_on_opposite": False,
                    },
                    "mtf_filter": False,
                }
            ]
        },
        "exit_options": (
            "optional exit keys (0/false = off): breakeven_r = move SL to entry after +xR; "
            "trail_atr = trailing stop at close - k*ATR; max_bars = close after N 15m bars; "
            "exit_on_opposite = close when the opposite entry fires"
        ),
        "requirements": [
            "only catalog indicators and allowed params",
            "each spec needs entry_long and/or entry_short",
            "prefer interpretable classic setups (trend, pullback, breakout, mean-revert)",
            "never propose empty entries or no-trade philosophies",
            "keep each side to <= 4 conditions including filters (complexity is penalised)",
        ],
    }
    obj = _llm_chat(
        "You design indicator-based futures specs for paper research. "
        "Goal is positive PnL expectancy; accept risk within leverage bounds. "
        "Never advise perpetual flat.",
        json.dumps(prompt, default=str),
        temperature=0.75,
    )
    if not isinstance(obj, dict):
        return []
    out: list[dict] = []
    for item in (obj.get("specs") or obj.get("proposals") or [])[:count]:
        if not isinstance(item, dict):
            continue
        s = normalize_spec(item)
        ok, msg = validate_spec(s)
        if ok:
            out.append(s)
        else:
            logger.info("LLM spec rejected: %s", msg)
    logger.info("LLM propose_specs validated=%d", len(out))
    return out


def propose_spec_patch(
    current_spec: dict,
    *,
    paper_stats: dict | None = None,
    tick_history_text: str | None = None,
    learning: dict | None = None,
) -> list[dict]:
    """Propose at most small patches (tune/add/remove one indicator condition)."""
    if not config.LLM_RESEARCH_ENABLED:
        return []
    from modules.indicator_catalog import catalog_for_prompt

    learning = learning or {}
    prompt = {
        "task": (
            "MISSION: improve paper PnL/expectancy. "
            "Given current Strategy Spec and paper stats, propose up to 3 patches. "
            "Each patch changes ONE condition: action=tune|add|remove on "
            "target=entry_long|entry_short|filters. "
            "Base every patch on evidence: compare indicator values and regime at entry "
            "for winners vs losers in recent_trades, and explain the pattern in reason. "
            "Do NOT suggest stopping trading. JSON only."
        ),
        "current_spec": current_spec,
        "paper_stats": paper_stats or {},
        "recent_trades": learning.get("recent_trades") or [],
        "lessons": learning.get("lessons") or "",
        "research_memory": learning.get("research_memory") or "",
        "tick_history": (tick_history_text or "")[:1200],
        "catalog": catalog_for_prompt(),
        "output_schema": {
            "patches": [
                {
                    "action": "tune",
                    "target": "filters",
                    "index": 0,
                    "params": {"period": 14},
                    "value": 1.0,
                    "reason": "loosen volume to get more samples with edge",
                }
            ]
        },
    }
    obj = _llm_chat(
        "You patch indicator specs for better expectancy. Never recommend flat forever.",
        json.dumps(prompt, default=str),
        temperature=0.5,
    )
    if not isinstance(obj, dict):
        return []
    patches = [p for p in (obj.get("patches") or [])[:5] if isinstance(p, dict)]
    logger.info("LLM propose_spec_patch n=%d", len(patches))
    return patches


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
