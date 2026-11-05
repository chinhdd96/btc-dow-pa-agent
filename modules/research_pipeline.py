"""End-to-end research: data → evolution → AI review → promote."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import config
from modules import champion_store
from modules.evolution import StrategyEvolution, maybe_promote
from modules.genome import Genome
from modules.llm_researcher import propose_genomes, review_research_report
from modules.paper_champion import PaperChampion
from modules.quant_data import fetch_market_snapshot, load_research_frame
from modules import tick_history

logger = logging.getLogger(__name__)


def run_research(*, use_llm: bool | None = None) -> dict:
    use_llm = config.LLM_RESEARCH_ENABLED if use_llm is None else use_llm
    champion_store.init_db()
    Path(config.RESEARCH_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fetching %s %s candles for %d days",
        config.SYMBOL,
        config.TIMEFRAME_MAIN,
        config.HISTORY_DAYS,
    )
    data = load_research_frame(config.HISTORY_DAYS, config.TIMEFRAME_MAIN)
    # Need enough bars for indicator warm-up + train/val/test (spot fallback may be thinner)
    min_bars = 800
    if len(data) < min_bars:
        raise RuntimeError(f"Insufficient history: {len(data)} candles (need >= {min_bars})")

    market_summary = {
        "bars": len(data),
        "last_close": float(data.close.iloc[-1]),
        "atr_pct": (
            float(data.atr_pct.iloc[-1]) if data.atr_pct.notna().iloc[-1] else None
        ),
        "funding_rate": (
            float(data.funding_rate.iloc[-1])
            if "funding_rate" in data and data.funding_rate.notna().iloc[-1]
            else None
        ),
    }
    state = champion_store.get_paper_state()
    focus = list(state.get("next_research_focus") or [])

    tick_ctx = tick_history.format_for_prompt()
    proposals: list[Genome] = []
    if use_llm:
        proposals = propose_genomes(
            market_summary,
            count=max(8, config.POPULATION_SIZE // 3),
            research_focus=focus,
            tick_history_text=tick_ctx,
        )

    evolution = StrategyEvolution(llm_proposals=proposals)
    ranked = evolution.research(data)

    trader = PaperChampion()
    current = trader.state.get("champion")
    selected = maybe_promote(ranked, current)

    baseline_exp = None
    if selected:
        eval_row = next(
            (r for r in ranked if r["genome_id"] == selected.genome_id), None
        )
        if eval_row:
            # Prefer test expectancy as promote baseline for decay detection
            baseline_exp = float(eval_row["test"].get("expectancy") or 0)
        trader.set_champion(selected, baseline_expectancy=baseline_exp)
        # Clear spent challenger hints after a research cycle
        st = champion_store.get_paper_state()
        st["challenger_hints"] = []
        st["research_due"] = False
        champion_store.set_paper_state(st)

    soft_used = bool(
        selected
        and next(
            (
                r
                for r in ranked
                if r["genome_id"] == selected.genome_id and r.get("soft_qualified")
            ),
            None,
        )
    )
    report_lite = {
        "bars": len(data),
        "champion": asdict(selected) if selected else None,
        "promote_mode": (
            "soft"
            if soft_used
            else ("hard" if selected else "none")
        ),
        "tick_history": tick_ctx,
        "top_candidates": [
            {
                "genome_id": r["genome_id"],
                "qualified": r["qualified"],
                "soft_qualified": bool(r.get("soft_qualified")),
                "robust_score": r["robust_score"],
                "family": r["genome"].get("family"),
                "validation": r["validation"],
                "test": r["test"],
            }
            for r in ranked[:10]
        ],
    }
    ai_review = (
        review_research_report(report_lite)
        if use_llm
        else {
            "summary": "LLM review skipped",
            "failure_modes": [],
            "next_research_focus": [],
        }
    )
    if ai_review.get("next_research_focus"):
        st = champion_store.get_paper_state()
        st["next_research_focus"] = ai_review["next_research_focus"]
        champion_store.set_paper_state(st)

    report = {
        "version": "champion-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbol": config.SYMBOL,
            "interval": config.TIMEFRAME_MAIN,
            "history_days": config.HISTORY_DAYS,
            "initial_capital": config.INITIAL_CAPITAL,
            "max_margin_fraction": config.MAX_MARGIN_FRACTION,
            "population_size": config.POPULATION_SIZE,
            "generations": config.GENERATIONS,
            "leverage_candidates": list(config.LEVERAGE_CANDIDATES),
            "llm_enabled": use_llm,
        },
        "market_snapshot": fetch_market_snapshot(config.SYMBOL),
        "bars": len(data),
        "champion": asdict(selected) if selected else None,
        "champion_evaluation": next(
            (r for r in ranked if selected and r["genome_id"] == selected.genome_id),
            None,
        ),
        "top_candidates": ranked[:20],
        "ai_review": ai_review,
        "qualification_note": (
            "Hard: val+test expectancy>0, PF>1.05, min trades. "
            "Soft (PAPER_SOFT_PROMOTE): lower trade mins, PF>=floor, either-split expectancy>0."
        ),
        "promote_mode": report_lite["promote_mode"],
        "execution_mode": "PAPER_ONLY; live trading is hard-disabled",
    }
    with open(config.RESEARCH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    champion_store.save_research(report)
    logger.info(
        "Research complete. Champion=%s report=%s",
        selected.genome_id if selected else None,
        config.RESEARCH_REPORT_PATH,
    )
    return report
