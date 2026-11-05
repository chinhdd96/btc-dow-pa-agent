"""End-to-end research: data → Spec evolution → AI review → promote (PnL-first)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from modules import champion_store
from modules import tick_history
from modules.llm_researcher import (
    propose_spec_patch,
    propose_specs,
    review_research_report,
)
from modules.paper_champion import PaperChampion
from modules.quant_data import fetch_market_snapshot, load_research_frame
from modules.spec_evolution import (
    SpecEvolution,
    maybe_promote_spec,
    patched_specs_from_llm,
)
from modules.strategy_spec import (
    genome_dict_to_spec,
    loosen_spec_for_starve,
    normalize_spec,
    spec_id,
)

logger = logging.getLogger(__name__)


def run_research(*, use_llm: bool | None = None) -> dict:
    use_llm = config.LLM_RESEARCH_ENABLED if use_llm is None else use_llm
    champion_store.init_db()
    Path(config.RESEARCH_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fetching %s %s candles for %d days (Spec research)",
        config.SYMBOL,
        config.TIMEFRAME_MAIN,
        config.HISTORY_DAYS,
    )
    data = load_research_frame(config.HISTORY_DAYS, config.TIMEFRAME_MAIN)
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
    starve = tick_history.starvation_report()
    focus = list(
        dict.fromkeys(
            [*focus, *(state.get("starve_focus") or []), *(starve.get("focus") or [])]
        )
    )
    tick_ctx = tick_history.format_for_prompt()
    research_mode = str(state.get("research_mode") or "replace")
    # Auto mode from trade counts (PnL-first learning cadence)
    trades_on = int(state.get("trades_on_spec") or 0)
    if trades_on >= config.SPEC_REPLACE_MIN_TRADES:
        research_mode = "replace"
    elif trades_on >= config.SPEC_PATCH_MIN_TRADES:
        research_mode = research_mode if research_mode in {"patch", "replace"} else "patch"

    current_spec = state.get("champion_spec")
    if not current_spec and state.get("champion"):
        try:
            current_spec = genome_dict_to_spec(state["champion"])
        except Exception:
            current_spec = None

    llm_specs: list[dict] = []
    patch_cands: list[dict] = []
    seed_specs: list[dict] = []

    if current_spec and starve.get("starving"):
        seed_specs.append(loosen_spec_for_starve(current_spec))

    if use_llm:
        if research_mode == "patch" and current_spec:
            patches = propose_spec_patch(
                current_spec,
                paper_stats={
                    "trades_on_spec": trades_on,
                    "trade_count": state.get("trade_count"),
                    "equity": state.get("equity"),
                    "baseline_expectancy": state.get("champion_baseline_expectancy"),
                },
                tick_history_text=tick_ctx,
            )
            patch_cands = patched_specs_from_llm(current_spec, patches)
            # Also allow a few fresh specs so search does not stall
            llm_specs = propose_specs(
                market_summary,
                count=max(4, config.POPULATION_SIZE // 6),
                research_focus=focus,
                tick_history_text=tick_ctx,
                starvation=starve,
            )
        else:
            llm_specs = propose_specs(
                market_summary,
                count=max(8, config.POPULATION_SIZE // 3),
                research_focus=focus,
                tick_history_text=tick_ctx,
                starvation=starve,
            )

    evolution = SpecEvolution(
        llm_specs=llm_specs,
        seed_specs=seed_specs,
        patch_candidates=patch_cands,
    )
    ranked = evolution.research(data)

    trader = PaperChampion()
    selected = maybe_promote_spec(
        ranked,
        current_spec,
        starving=bool(starve.get("starving")),
        paper_trade_count=int(state.get("trade_count") or 0),
    )

    baseline_exp = None
    if selected:
        eval_row = next(
            (r for r in ranked if r["spec_id"] == spec_id(selected)),
            None,
        )
        if eval_row:
            baseline_exp = float(eval_row["test"].get("expectancy") or 0)
        trader.set_champion_spec(selected, baseline_expectancy=baseline_exp)
        st = champion_store.get_paper_state()
        st["challenger_hints"] = []
        st["research_due"] = False
        st["research_mode"] = None
        st["starve_focus"] = []
        st["last_spec_research_ts"] = time.time()
        st["explore_mode"] = bool(starve.get("starving")) and not (
            eval_row and eval_row.get("qualified")
        )
        champion_store.set_paper_state(st)

    soft_used = bool(
        selected
        and next(
            (
                r
                for r in ranked
                if r["spec_id"] == spec_id(selected) and r.get("soft_qualified")
            ),
            None,
        )
    )
    report_lite = {
        "bars": len(data),
        "champion_spec": normalize_spec(selected) if selected else None,
        "champion_spec_id": spec_id(selected) if selected else None,
        "promote_mode": (
            "soft" if soft_used else ("hard" if selected else "none")
        ),
        "research_mode": research_mode,
        "tick_history": tick_ctx,
        "starvation": starve,
        "research_focus": focus,
        "top_candidates": [
            {
                "spec_id": r["spec_id"],
                "name": r.get("name"),
                "qualified": r["qualified"],
                "soft_qualified": bool(r.get("soft_qualified")),
                "robust_score": r["robust_score"],
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
        "version": "champion-spec-1.0",
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
            "research_mode": research_mode,
            "mission": "pnl_positive_priority",
        },
        "market_snapshot": fetch_market_snapshot(config.SYMBOL),
        "bars": len(data),
        "champion": None,  # legacy field unused
        "champion_spec": normalize_spec(selected) if selected else None,
        "champion_evaluation": next(
            (r for r in ranked if selected and r["spec_id"] == spec_id(selected)),
            None,
        ),
        "top_candidates": ranked[:20],
        "ai_review": ai_review,
        "qualification_note": (
            "Spec path: hard val+test expectancy>0 PF>1.05; "
            "soft promote keeps trading for PnL learning. "
            "No perpetual flat — starve loosens/researches specs."
        ),
        "promote_mode": report_lite["promote_mode"],
        "execution_mode": "PAPER_ONLY; live trading is hard-disabled",
    }
    with open(config.RESEARCH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    champion_store.save_research(report)
    logger.info(
        "Research complete. Spec=%s report=%s",
        spec_id(selected) if selected else None,
        config.RESEARCH_REPORT_PATH,
    )
    return report
