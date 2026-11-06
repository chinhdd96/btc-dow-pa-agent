"""Research: data → LLM (with memory) → Spec evolution → walk-forward → promote.

compute_research() is safe to run in a background thread: it only reads paper
state and writes the research report. apply_research() runs on the main thread
and is the only place that changes the Champion / shadows.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from modules import champion_store, research_memory, shadow_book, tick_history
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


def _learning_context() -> dict[str, Any]:
    from modules.memory_manager import MemoryManager

    try:
        lessons = MemoryManager().get_all_lessons_summary()
    except Exception as exc:  # noqa: BLE001
        logger.warning("lessons unavailable: %s", exc)
        lessons = ""
    return {
        "research_memory": research_memory.summarize(),
        "lessons": lessons[:1500],
        "recent_trades": research_memory.recent_trades_for_prompt(),
    }


def compute_research(*, use_llm: bool | None = None) -> dict[str, Any]:
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
    if len(data) < 800:
        raise RuntimeError(f"Insufficient history: {len(data)} candles (need >= 800)")

    market_summary = {
        "bars": len(data),
        "last_close": float(data.close.iloc[-1]),
        "atr_pct": float(data.atr_pct.iloc[-1]) if data.atr_pct.notna().iloc[-1] else None,
        "funding_rate": (
            float(data.funding_rate.iloc[-1])
            if "funding_rate" in data and data.funding_rate.notna().iloc[-1]
            else None
        ),
    }
    state = champion_store.get_paper_state()
    starve = tick_history.starvation_report()
    focus = list(
        dict.fromkeys(
            [
                *(state.get("next_research_focus") or []),
                *(state.get("starve_focus") or []),
                *(state.get("research_hints") or []),
                *(starve.get("focus") or []),
            ]
        )
    )
    tick_ctx = tick_history.format_for_prompt()
    research_mode = str(state.get("research_mode") or "replace")
    trades_on = int(state.get("trades_on_spec") or 0)
    if trades_on >= config.SPEC_REPLACE_MIN_TRADES:
        research_mode = "replace"
    elif trades_on >= config.SPEC_PATCH_MIN_TRADES:
        research_mode = research_mode if research_mode in {"patch", "replace"} else "patch"

    current_spec = state.get("champion_spec")
    if not current_spec and state.get("champion"):
        try:
            current_spec = genome_dict_to_spec(state["champion"])
        except Exception:  # noqa: BLE001
            current_spec = None

    learning = _learning_context() if use_llm else {}
    llm_specs: list[dict] = []
    patch_cands: list[dict] = []
    seed_specs: list[dict] = [
        sh["champion_spec"] for sh in shadow_book.shadows(dict(state)) if sh.get("champion_spec")
    ]
    if current_spec and starve.get("starving"):
        seed_specs.append(loosen_spec_for_starve(current_spec))

    if use_llm:
        fresh_count = max(8, config.POPULATION_SIZE // 3)
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
                learning=learning,
            )
            patch_cands = patched_specs_from_llm(current_spec, patches)
            fresh_count = max(4, config.POPULATION_SIZE // 6)
        llm_specs = propose_specs(
            market_summary,
            count=fresh_count,
            research_focus=focus,
            tick_history_text=tick_ctx,
            starvation=starve,
            learning=learning,
        )

    ranked = SpecEvolution(
        llm_specs=llm_specs,
        seed_specs=seed_specs,
        patch_candidates=patch_cands,
    ).research(data)

    selected = maybe_promote_spec(
        ranked,
        current_spec,
        starving=bool(starve.get("starving")),
        paper_trade_count=int(state.get("trade_count") or 0),
    )
    report = _build_report(
        data_len=len(data),
        ranked=ranked,
        selected=selected,
        research_mode=research_mode,
        starve=starve,
        focus=focus,
        tick_ctx=tick_ctx,
        use_llm=use_llm,
        llm_counts={"llm_specs": len(llm_specs), "patches": len(patch_cands)},
    )
    with open(config.RESEARCH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    champion_store.save_research(report)
    logger.info("Research computed. backtest pick=%s", spec_id(selected) if selected else None)
    return {"report": report, "ranked": ranked, "selected": selected, "starve": starve}


def _eval_for(ranked: list[dict], spec: dict | None) -> dict | None:
    if not spec:
        return None
    sid = spec_id(spec)
    return next((r for r in ranked if r["spec_id"] == sid), None)


def _build_report(
    *,
    data_len: int,
    ranked: list[dict],
    selected: dict | None,
    research_mode: str,
    starve: dict,
    focus: list[str],
    tick_ctx: str,
    use_llm: bool,
    llm_counts: dict,
) -> dict[str, Any]:
    eval_row = _eval_for(ranked, selected)
    soft_used = bool(eval_row and eval_row.get("soft_qualified"))
    top = [
        {
            "spec_id": r["spec_id"],
            "name": r.get("name"),
            "qualified": r["qualified"],
            "soft_qualified": bool(r.get("soft_qualified")),
            "robust_score": r["robust_score"],
            "positive_folds": r.get("positive_folds"),
            "folds": r.get("folds"),
            "validation": r["validation"],
            "test": r["test"],
        }
        for r in ranked[:10]
    ]
    report_lite = {
        "bars": data_len,
        "champion_spec": normalize_spec(selected) if selected else None,
        "champion_spec_id": spec_id(selected) if selected else None,
        "promote_mode": "soft" if soft_used else ("hard" if selected else "none"),
        "research_mode": research_mode,
        "tick_history": tick_ctx,
        "starvation": starve,
        "research_focus": focus,
        "top_candidates": top,
    }
    ai_review = (
        review_research_report(report_lite)
        if use_llm
        else {"summary": "LLM review skipped", "failure_modes": [], "next_research_focus": []}
    )
    return {
        "version": "champion-spec-2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbol": config.SYMBOL,
            "interval": config.TIMEFRAME_MAIN,
            "history_days": config.HISTORY_DAYS,
            "population_size": config.POPULATION_SIZE,
            "wf_folds": config.WF_FOLDS,
            "leverage_candidates": list(config.LEVERAGE_CANDIDATES),
            "llm_enabled": use_llm,
            "research_mode": research_mode,
            **llm_counts,
        },
        "market_snapshot": fetch_market_snapshot(config.SYMBOL),
        "bars": data_len,
        "champion_spec": report_lite["champion_spec"],
        "champion_evaluation": eval_row,
        "top_candidates": ranked[:20],
        "evaluated_digest": [research_memory.digest_row(r) for r in ranked],
        "ai_review": ai_review,
        "promote_mode": report_lite["promote_mode"],
        "execution_mode": "PAPER_ONLY; live trading is hard-disabled",
    }


def apply_research(result: dict[str, Any]) -> dict[str, Any]:
    """Main thread only: choose Champion (shadow forward results first), manage shadows."""
    report = result["report"]
    ranked = result["ranked"]
    starve = result.get("starve") or {}
    trader = PaperChampion()
    state = trader.state
    old_spec = state.get("champion_spec")
    old_id = spec_id(old_spec) if old_spec else None

    selected = result.get("selected")
    promote_source = "backtest"
    challenger = shadow_book.best_challenger(state)
    if challenger is not None:
        selected = challenger["champion_spec"]
        promote_source = "shadow"
        logger.info("Shadow %s beats Champion on forward paper", challenger.get("name"))

    events: list[str] = []
    if selected:
        eval_row = _eval_for(ranked, selected)
        baseline = None
        if eval_row:
            baseline = float((eval_row.get("oos") or eval_row["test"]).get("expectancy") or 0)
        elif challenger is not None:
            baseline = shadow_book.stats(challenger.get("trades") or [])["expectancy"]
        trader.set_champion_spec(selected, baseline_expectancy=baseline)
        new_id = trader.state["champion_spec_id"]
        if new_id != old_id:
            shadow_book.remove(state, new_id)
            if old_spec and shadow_book.add_spec(state, old_spec, _eval_for(ranked, old_spec)):
                events.append(f"Champion cũ → shadow: {old_spec.get('name')}")
        report["champion_spec"] = trader.state["champion_spec"]
        report["champion_evaluation"] = eval_row
        if promote_source == "shadow":
            report["promote_mode"] = "shadow"

    retired = shadow_book.retire(state)
    added = shadow_book.refill(state, ranked, state.get("champion_spec_id"))
    events += [f"loại shadow: {r}" for r in retired]
    events += [f"thêm shadow: {a}" for a in added]

    review = report.get("ai_review") or {}
    state["challenger_hints"] = []
    state["research_hints"] = []
    state["research_due"] = False
    state["research_mode"] = None
    state["starve_focus"] = []
    state["last_spec_research_ts"] = time.time()
    eval_sel = report.get("champion_evaluation") or {}
    state["explore_mode"] = bool(starve.get("starving")) and not eval_sel.get("qualified")
    if review.get("next_research_focus"):
        state["next_research_focus"] = review["next_research_focus"]
    trader.save()
    try:
        state["research_memory_cache"] = research_memory.summarize()
        trader.save()
    except Exception as exc:  # noqa: BLE001
        logger.warning("research memory cache failed: %s", exc)

    report["promote_source"] = promote_source
    report["shadow_events"] = events
    report["shadows"] = shadow_book.summary_lines(state)
    return report


def run_research(*, use_llm: bool | None = None) -> dict:
    """Synchronous compute + apply (CLI script)."""
    return apply_research(compute_research(use_llm=use_llm))
