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
    prompt = {
        "task": (
            "Propose diverse, testable BTCUSDT perpetual futures strategy hypotheses. "
            "Do not claim profitability. Return JSON only. "
            "Use recent tick_history only as research context (why Champion stayed flat / fired); "
            "do NOT invent per-candle BUY/SELL instructions."
        ),
        "market_summary": market_summary,
        "research_focus": research_focus or [],
        "market_notes": ctx["combined"],
        "tick_history": (tick_history_text or "")[: config.TICK_HISTORY_PROMPT_MAX_CHARS],
        "allowed_values": allowed,
        "requirements": [
            "diverse families and parameter combinations",
            "avoid lookahead",
            "prefer interpretable hypotheses",
            "do not optimize only for recent noise",
            "use only allowed_values",
            "if tick_history shows chronic SKIP from filters, prefer looser volume/ATR/mtf variants",
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
    genome: dict | None = None,
    recent_bars_summary: dict | None = None,
) -> dict[str, Any]:
    """Analyze one closed paper trade; may set decay_signal + challenger_hints."""
    default = {
        "classification": "INSUFFICIENT_EVIDENCE",
        "lesson": "",
        "decay_signal": False,
        "challenger_hints": [],
    }
    if not config.LLM_RESEARCH_ENABLED or not config.LLM_API_KEY:
        pnl = float(trade.get("pnl") or 0)
        default["classification"] = "VALID_LOSS" if pnl < 0 else "GOOD_EDGE"
        default["lesson"] = f"Paper close {trade.get('exit_reason')} pnl={pnl:+.4f}"
        return default

    from modules import tick_history

    prompt = {
        "task": (
            "Post-mortem a PAPER BTCUSDT trade from a Champion genome. "
            "Classify outcome, write a short lesson, say if strategy may be decaying, "
            "and optionally propose challenger genome parameter hints "
            "(must use allowed genome fields only). "
            "tick_history is research context only — not entry authority. JSON only."
        ),
        "trade": trade,
        "genome": genome,
        "market": recent_bars_summary or {},
        "tick_history": tick_history.format_for_prompt(),
        "allowed_families": list(FAMILIES),
        "output_schema": {
            "classification": "GOOD_EDGE|VALID_LOSS|BAD_SETUP|PROCESS_ERROR|INSUFFICIENT_EVIDENCE",
            "lesson": "string Vietnamese or English",
            "decay_signal": False,
            "challenger_hints": [
                {
                    "family": "trend_cross",
                    "fast": 12,
                    "slow": 50,
                    "atr_stop": 2.0,
                    "reward_r": 2.0,
                    "leverage": 2,
                }
            ],
        },
    }
    obj = _llm_chat(
        "You analyze paper trades for research. Never recommend live orders.",
        json.dumps(prompt, default=str),
        temperature=0.4,
    )
    if not isinstance(obj, dict):
        return default
    hints_raw = obj.get("challenger_hints") or []
    hints: list[dict] = []
    for h in hints_raw[:5]:
        if isinstance(h, dict):
            g = genome_from_dict(h)
            if g:
                from dataclasses import asdict

                hints.append(asdict(g))
    cls = str(obj.get("classification") or "INSUFFICIENT_EVIDENCE").upper()
    return {
        "classification": cls,
        "lesson": str(obj.get("lesson") or "")[:800],
        "decay_signal": bool(obj.get("decay_signal")),
        "challenger_hints": hints,
    }
