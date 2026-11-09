"""Research for the single indicator family.

genesis : no family yet → LLM drafts → walk-forward picks v1.
patch   : LLM proposes small edits (1-2 ops) of the current version.
rework  : after repeated failure / decay the LLM may replace the core indicators.

compute_research() is safe to run in a background thread: it only reads paper
state and writes the research report. apply_research() runs on the main thread
and is the only place that changes the Champion / family / shadows.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from modules import champion_store, family, research_memory, shadow_book, tick_history
from modules.llm_researcher import (
    propose_family_changes,
    propose_family_drafts,
    review_research_report,
)
from modules.paper_champion import PaperChampion
from modules.quant_data import fetch_market_snapshot, load_research_frame
from modules.spec_evolution import SpecEvolution, cap_leverage, choose_version, oos_trades
from modules.strategy_spec import (
    apply_ops,
    base_name,
    indicator_types,
    normalize_spec,
    short_conditions,
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


def _paper_of(sid: str | None) -> dict[str, Any]:
    return next(
        (p for p in research_memory.paper_results_by_spec() if p["id"] == sid),
        {"n": 0, "pnl": 0.0, "expectancy": None},
    )


def _healthy_on_paper(paper: dict) -> bool:
    return paper["n"] < 5 or (paper.get("expectancy") or 0) >= 0


def _improve_mode(state: dict, paper: dict) -> str:
    fam = family.get(state) or {}
    requested = str(state.get("research_mode") or "")
    if requested == "no_trade":
        return "no_trade"
    if requested in {"rework", "replace"} or state.get("explore_mode"):
        return "rework"
    if int(fam.get("fail_streak") or 0) >= config.FAMILY_REWORK_AFTER:
        return "rework"
    if paper["n"] >= config.SPEC_REPLACE_MIN_TRADES and (paper.get("expectancy") or 0) < 0:
        return "rework"
    return "patch"


def _variants(current: dict, proposals: list[dict], mode: str, name: str) -> list[dict]:
    out = []
    for v in proposals:
        s = apply_ops(current, v["ops"])
        if s is None:
            continue
        s["name"] = name
        s["variant"] = {"source": f"llm_{mode}", "reason": v.get("reason") or "", "ops": v["ops"]}
        out.append(s)
    logger.info("family variants usable=%d/%d", len(out), len(proposals))
    return out


def _attempt_feedback(ranked: list[dict], rnd: int, *, min_trades: int = 0) -> list[dict]:
    """Best tested form of each LLM candidate this round, with a verdict the LLM can act on."""
    best: dict[tuple, dict] = {}
    for r in ranked:
        meta = r["spec"].get("variant") or {}
        if not str(meta.get("source") or "").startswith("llm"):
            continue
        key = (base_name(r.get("name")), str(meta.get("reason") or ""))
        if key not in best or r["robust_score"] > best[key]["robust_score"]:
            best[key] = r
    out = []
    for (name, reason), r in best.items():
        oos = r.get("oos") or {}
        n = int(oos.get("trade_count") or 0)
        verdict = []
        if n < max(config.MIN_TEST_TRADES * 2, min_trades):
            verdict.append("too_few_trades")
        if float(oos.get("expectancy") or 0) <= 0 or float(oos.get("profit_factor") or 0) < 1.05:
            verdict.append("losing")
        if float(oos.get("max_drawdown_pct") or 0) > 30:
            verdict.append("high_drawdown")
        if int(r.get("positive_folds") or 0) < config.WF_MIN_POSITIVE_FOLDS:
            verdict.append("unstable_across_folds")
        out.append(
            {
                "round": rnd,
                "name": name,
                "reason": reason[:200],
                "indicators": indicator_types(r["spec"]),
                "conditions": short_conditions(r["spec"], max_len=260),
                "oos": {
                    "trades": n,
                    "expectancy": round(float(oos.get("expectancy") or 0), 4),
                    "profit_factor": round(float(oos.get("profit_factor") or 0), 2),
                    "win_rate_pct": round(float(oos.get("win_rate_pct") or 0), 1),
                    "max_drawdown_pct": round(float(oos.get("max_drawdown_pct") or 0), 1),
                },
                "positive_folds": f"{r.get('positive_folds')}/{config.WF_FOLDS}",
                "verdict": verdict or ["below_promotion_bar"],
            }
        )
    return out[:8]


def compute_research(*, use_llm: bool | None = None) -> dict[str, Any]:
    use_llm = config.LLM_RESEARCH_ENABLED if use_llm is None else use_llm
    champion_store.init_db()
    Path(config.RESEARCH_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fetching %s %s candles for %d days (family research)",
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
    learning = _learning_context() if use_llm else {}

    genesis = family.needs_genesis(state)
    current = None if genesis else cap_leverage(state["champion_spec"])
    paper = _paper_of(state.get("champion_spec_id")) if current else {"n": 0, "expectancy": None}
    mode = "genesis" if genesis else _improve_mode(state, paper)
    idle_h = family.hours_without_entry(state)
    logger.info("Family research mode=%s family=%s", mode, family.label(state))

    name = None if genesis else family.get(state)["name"]
    paper_stats = {
        "version": family.label(state),
        "trades_on_version": paper["n"],
        "pnl_on_version": paper.get("pnl"),
        "expectancy_on_version": paper.get("expectancy"),
        "hours_without_entry": round(idle_h, 1) if idle_h is not None else None,
        "equity": state.get("equity"),
        "baseline_expectancy": state.get("champion_baseline_expectancy"),
    }

    def llm_candidates(feedback: list[dict]) -> list[dict]:
        if not use_llm:
            return []
        if genesis:
            return propose_family_drafts(
                market_summary,
                config.FAMILY_GENESIS_DRAFTS,
                research_focus=focus,
                tick_history_text=tick_ctx,
                learning=learning,
                feedback=feedback,
            )
        proposals = propose_family_changes(
            current,
            mode=mode,
            count=config.FAMILY_VARIANTS,
            family_history=family.history_for_prompt(state),
            paper_stats=paper_stats,
            research_focus=focus,
            tick_history_text=tick_ctx,
            learning=learning,
            feedback=feedback,
        )
        return _variants(current, proposals, mode, name)

    extra: list[dict] = []
    if not genesis:
        for sh in shadow_book.shadows(dict(state)):
            if sh.get("champion_spec"):
                s = normalize_spec(sh["champion_spec"])
                s["name"] = name
                extra.append(s)

    starving = mode == "no_trade" or (
        bool(starve.get("starving")) and int(state.get("trades_on_spec") or 0) <= 0
    )
    deadline = time.monotonic() + config.RESEARCH_MAX_MINUTES * 60
    rounds = max(1, config.FAMILY_FEEDBACK_ROUNDS) if use_llm else 1
    merged: dict[str, dict] = {}
    feedback: list[dict] = []
    llm_count = 0
    selected, decision, min_trades = None, "none", 0
    for rnd in range(1, rounds + 1):
        cands = llm_candidates(feedback)
        llm_count += len(cands)
        if rnd == 1:
            cands += extra
        if not cands and rnd > 1:
            break
        left = deadline - time.monotonic()
        if left <= 30 and merged:
            break
        ranked_r = SpecEvolution(
            candidates=cands,
            current=current,
            deadline=time.monotonic() + left / (rounds - rnd + 1),
            seed_offset=rnd,
        ).research(data)
        for r in ranked_r:
            merged.setdefault(r["spec_id"], r)
        ranked = sorted(merged.values(), key=lambda z: z["robust_score"], reverse=True)
        if mode == "no_trade":
            cur_n = oos_trades(_eval_for(ranked, current))
            min_trades = max(int(cur_n * config.FAMILY_NO_TRADE_MIN_MORE + 0.999), cur_n + 5)
        selected, decision = choose_version(
            ranked,
            current,
            healthy_on_paper=_healthy_on_paper(paper),
            starving=starving,
            min_trades=min_trades,
        )
        good = (
            bool((_eval_for(ranked, selected) or {}).get("qualified"))
            if genesis
            else decision in {"promote", "forward"}
        )
        logger.info("Feedback round %d/%d: candidates=%d decision=%s ok=%s", rnd, rounds, len(cands), decision, good)
        if good or rnd == rounds:
            break
        feedback += _attempt_feedback(ranked_r, rnd, min_trades=min_trades)

    ranked = sorted(merged.values(), key=lambda z: z["robust_score"], reverse=True)
    if genesis and not ranked:
        raise RuntimeError("LLM chưa tạo được family hợp lệ (3-5 chỉ báo) — sẽ thử lại")
    if (
        decision == "keep"
        and selected
        and state.get("champion_spec_id")
        and spec_id(selected) != state["champion_spec_id"]
    ):
        # Same logic, only the leverage cap changed → still a new version on record
        selected = {**selected, "variant": {"source": "leverage_cap", "reason": f"đòn bẩy ≤ {config.FAMILY_MAX_LEVERAGE}x"}}
        decision = "promote"
    report = _build_report(
        data_len=len(data),
        ranked=ranked,
        selected=selected,
        current=current,
        mode=mode,
        decision=decision,
        starve=starve,
        focus=focus,
        tick_ctx=tick_ctx,
        use_llm=use_llm,
        llm_count=llm_count,
    )
    report["config"]["feedback_rounds"] = rnd
    report["config"]["min_oos_trades"] = min_trades
    with open(config.RESEARCH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    champion_store.save_research(report)
    logger.info("Research computed. mode=%s decision=%s pick=%s", mode, decision, spec_id(selected) if selected else None)
    return {
        "report": report,
        "ranked": ranked,
        "selected": selected,
        "mode": mode,
        "decision": decision,
        "paper": paper,
        "starve": starve,
    }


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
    current: dict | None,
    mode: str,
    decision: str,
    starve: dict,
    focus: list[str],
    tick_ctx: str,
    use_llm: bool,
    llm_count: int,
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
    changes = family.diff_specs(current, selected) if selected and decision in {"genesis", "promote"} else []
    report_lite = {
        "bars": data_len,
        "family_mode": mode,
        "decision": decision,
        "changes": changes,
        "champion_spec": family.pop_meta(selected)[0] if selected else None,
        "promote_mode": "soft" if soft_used else ("hard" if selected else "none"),
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
        "version": "champion-family-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbol": config.SYMBOL,
            "interval": config.TIMEFRAME_MAIN,
            "history_days": config.HISTORY_DAYS,
            "population_size": config.POPULATION_SIZE,
            "wf_folds": config.WF_FOLDS,
            "leverage_candidates": list(config.LEVERAGE_CANDIDATES),
            "llm_enabled": use_llm,
            "research_mode": mode,
            "llm_candidates": llm_count,
        },
        "market_snapshot": fetch_market_snapshot(config.SYMBOL),
        "bars": data_len,
        "family_mode": mode,
        "decision": decision,
        "changes": changes,
        "champion_spec": report_lite["champion_spec"],
        "champion_evaluation": eval_row,
        "top_candidates": ranked[:20],
        "evaluated_digest": [research_memory.digest_row(r) for r in ranked],
        "ai_review": ai_review,
        "promote_mode": report_lite["promote_mode"],
        "execution_mode": "PAPER_ONLY; live trading is hard-disabled",
    }


def apply_research(result: dict[str, Any]) -> dict[str, Any]:
    """Main thread only: set the family version, record the changelog, manage shadows."""
    report = result["report"]
    ranked = result["ranked"]
    mode = result.get("mode") or "patch"
    decision = result.get("decision") or "keep"
    starve = result.get("starve") or {}
    trader = PaperChampion()
    state = trader.state
    old_spec = state.get("champion_spec")
    old_label = family.label(state)

    selected = result.get("selected")
    promote_source = "backtest"
    events: list[str] = []
    challenger = None

    if mode != "genesis":
        challenger = shadow_book.best_challenger(state)
        if challenger is not None:
            selected = cap_leverage(challenger["champion_spec"])
            decision = "promote"
            promote_source = "shadow"
            logger.info("Shadow %s beats Champion on forward paper", challenger.get("name"))

    eval_row = _eval_for(ranked, selected)
    if selected and decision in {"genesis", "promote"}:
        spec, meta = family.pop_meta(selected)
        if mode == "genesis":
            state["shadows"] = []
        else:
            spec["name"] = family.get(state)["name"]
        if eval_row:
            baseline = float((eval_row.get("oos") or eval_row["test"]).get("expectancy") or 0)
        else:
            baseline = shadow_book.stats((challenger or {}).get("trades") or [])["expectancy"]
        new_id = spec_id(spec)
        if mode == "genesis":
            trader.set_champion_spec(spec, baseline_expectancy=baseline)
            family.start(state, spec, meta=meta, row=eval_row)
            events.append(f"Family mới: {family.label(state)}")
        elif new_id != state.get("champion_spec_id"):
            trader.set_champion_spec(spec, baseline_expectancy=baseline)
            family.record_version(state, old_spec, spec, meta=meta, source=promote_source, row=eval_row)
            shadow_book.remove(state, new_id)
            old_capped = cap_leverage(old_spec) if old_spec else None
            if (
                old_capped
                and spec_id(old_capped) != new_id
                and shadow_book.add_spec(
                    state, old_capped, _eval_for(ranked, old_capped), label=f"{old_label} (cũ)", champion=spec
                )
            ):
                events.append(f"{old_label} → shadow")
            events.append(f"Lên {family.label(state)}")
        report["champion_spec"] = trader.state["champion_spec"]
        report["champion_evaluation"] = eval_row
        report["changes"] = family.diff_specs(None if mode == "genesis" else old_spec, spec)
    elif decision == "forward" and ranked:
        best = next((r for r in ranked if r["qualified"]), None)
        if best and shadow_book.ensure(state, best, champion=state.get("champion_spec")):
            events.append("ứng viên tốt hơn → chạy shadow kiểm chứng trước khi lên version")

    fam = family.get(state)
    if fam is not None:
        # Consecutive rounds where the running version is weak → next round reworks the core
        running = _eval_for(ranked, state.get("champion_spec"))
        promoted = decision in {"genesis", "promote"}
        ok_now = promote_source == "shadow" or bool(
            running and (running["qualified"] or running.get("soft_qualified"))
        )
        healthy = promoted or _healthy_on_paper(result.get("paper") or {"n": 0})
        fam["fail_streak"] = 0 if ok_now and healthy else int(fam.get("fail_streak") or 0) + 1

    retired = shadow_book.retire(state)
    added = [] if mode == "genesis" else shadow_book.refill(state, ranked, state.get("champion_spec"))
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
    state["explore_mode"] = not eval_sel.get("qualified")
    if review.get("next_research_focus"):
        state["next_research_focus"] = review["next_research_focus"]
    trader.save()
    try:
        state["research_memory_cache"] = research_memory.summarize()
        trader.save()
    except Exception as exc:  # noqa: BLE001
        logger.warning("research memory cache failed: %s", exc)

    report["decision"] = decision
    report["explore_mode"] = bool(state.get("explore_mode"))
    report["promote_source"] = promote_source
    report["family"] = {
        "label": family.label(state),
        "fail_streak": (family.get(state) or {}).get("fail_streak"),
    }
    report["shadow_events"] = events
    report["shadows"] = shadow_book.summary_lines(state)
    return report


def run_research(*, use_llm: bool | None = None) -> dict:
    """Synchronous compute + apply (CLI script)."""
    return apply_research(compute_research(use_llm=use_llm))
