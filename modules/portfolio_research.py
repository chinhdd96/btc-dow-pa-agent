"""One portfolio research job, with a chronological holdout for promotion."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

import config
from modules.features_catalog import ensure_features_for_spec
from modules.indicator_catalog import catalog_for_prompt
from modules.llm_researcher import _llm_chat
from modules.paper_champion import interval_td
from modules.portfolio_data import MarketData, ms
from modules.portfolio_engine import Portfolio, new_state
from modules.spec_backtest import _score, metrics_from_trades
from modules.spec_evolution import cap_leverage
from modules.strategy_spec import apply_ops, compile_signal, family_shape_ok, normalize_spec, spec_id

logger = logging.getLogger(__name__)
PROMPT_VERSION = "portfolio-v1"


def _qualified(metrics: dict, minimum_trades: int) -> bool:
    return (
        metrics["trade_count"] >= minimum_trades and metrics["net_return_pct"] > 0
        and metrics["profit_factor"] > 1.05
    )


def prepared(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return ensure_features_for_spec(frame, spec).set_index("open_time", drop=False)


def simulate(frames: dict[str, pd.DataFrame], spec: dict, times: pd.DatetimeIndex, deadline: float) -> dict:
    symbols = tuple(frames)
    fn = compile_signal(spec)
    engine = Portfolio(new_state(symbols, config.INITIAL_CAPITAL))
    engine.state["version"] = 1
    start = times[0]
    duration = interval_td()
    end = times[-1]
    duration_ms = int(duration.total_seconds() * 1000)
    data = {}
    prior = {}
    for symbol, frame in frames.items():
        relevant = frame.loc[start - duration : end]
        expected = pd.DatetimeIndex([start - duration]).append(times)
        if not relevant.index.equals(expected):
            raise ValueError(f"Unaligned or incomplete research window for {symbol}")
        records = iter(relevant.to_dict("records"))
        prior[symbol] = next(records)
        data[symbol] = records
    peak, drawdown = config.INITIAL_CAPITAL, 0.0
    for index, stamp in enumerate(times):
        if index % 256 == 0 and time.monotonic() >= deadline:
            raise TimeoutError("Portfolio backtest reached its time budget")
        now = ms(stamp)
        rows = {symbol: next(data[symbol]) for symbol in symbols}
        for symbol, row in rows.items():
            engine.state["markets"][symbol]["mark"] = row["open"]
        for symbol, row in rows.items():
            engine.fill(symbol, now, row["open"])
        for symbol, row in rows.items():
            rate = row.get("funding_event_rate")
            if rate is not None:
                engine.funding(symbol, now, rate)
            engine.touch(symbol, now, row["high"], row["low"], row["close"])
            signal = fn(row, prior[symbol])
            engine.close_bar(symbol, now + duration_ms, row["high"], row["low"], row["close"], row["atr"], signal)
            engine.signal(symbol, side=signal, atr=row["atr"], spec=spec,
                          signal_time=now, fill_time=now + duration_ms)
        prior = rows
        equity = engine.equity()
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    for symbol, market in engine.state["markets"].items():
        if market.get("position"):
            engine.close(symbol, ms(end) + duration_ms, "END", market["mark"])
    totals = metrics_from_trades(engine.closed)
    result = {**totals, "max_drawdown_pct": 100 * drawdown}
    if engine.closed:
        result["score"] = _score(result["net_return_pct"], drawdown, result["profit_factor"], len(engine.closed))
    by_symbol = {symbol: {"trades": 0, "pnl": 0.0} for symbol in symbols}
    for trade in engine.closed:
        item = by_symbol[trade["symbol"]]
        item["trades"] += 1
        item["pnl"] += trade["pnl"]
    result["symbols"] = by_symbol
    return result


def _propose(symbols: tuple[str, ...], context: dict, current: dict | None, summary: dict) -> list[dict]:
    count = config.FAMILY_VARIANTS if current else config.FAMILY_GENESIS_DRAFTS
    prompt = {
        "task": (
            "Propose one shared indicator strategy for all portfolio symbols, with one shared wallet. "
            "Evaluate mechanisms across symbols and regimes and avoid relying on one coin. "
            "Reference supplied evidence IDs. Open entries are censored outcomes. "
            "Genesis supplies complete Specs. Review supplies alternative patches of one or two ops. "
            "If review evidence is insufficient, return insufficient_evidence and no variants."
        ),
        "max_candidates": count,
        "symbols": symbols, "timeframe": config.TIMEFRAME_MAIN, "market": summary,
        "portfolio": context, "current_spec": current, "catalog": catalog_for_prompt(),
        "constraints": {
            "distinct_indicators": [config.FAMILY_MIN_INDICATORS, config.FAMILY_MAX_INDICATORS],
            "patch_actions": ["add", "remove", "replace", "tune", "exit", "mtf"],
            "condition_blocks": ["entry_long", "entry_short", "filters"],
            "leverage": [x for x in config.LEVERAGE_CANDIDATES if x <= config.FAMILY_MAX_LEVERAGE],
        },
        "output": {
            "insufficient_evidence": False,
            "variants": [{
                "hypothesis": "string", "evidence_refs": ["entry:1 or market:SYMBOL"],
                "expected_effect": "measurable outcome",
                "spec": {"name": "name", "entry_long": [], "entry_short": [], "filters": [],
                         "mtf_filter": False, "exit": {"atr_stop": 2, "reward_r": 2, "leverage": 2}},
                "ops": [{"action": "tune", "target": "filters", "index": 0, "value": 20}],
            }],
        },
    }
    response = _llm_chat("You propose paper research candidates. Return one JSON object.", json.dumps(prompt),
                         temperature=0.35, timeout=45)
    if (not isinstance(response, dict) or not isinstance(response.get("variants"), list)
            or not isinstance(response.get("insufficient_evidence"), bool)):
        raise RuntimeError("Portfolio candidate prompt failed or returned invalid JSON")
    if response["insufficient_evidence"]:
        return []
    found = {}
    evidence = {f"market:{symbol}" for symbol in symbols}
    evidence.update(f"entry:{entry['entry_seq']}" for entry in context["entries"])
    for item in response["variants"][:count]:
        if not isinstance(item, dict):
            continue
        refs = item.get("evidence_refs")
        if (not isinstance(refs, list) or not refs or
                any(not isinstance(ref, str) or ref not in evidence for ref in refs)):
            continue
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("hypothesis", "expected_effect")):
            continue
        try:
            ops = item.get("ops")
            candidate = (apply_ops(current, ops) if isinstance(ops, list) and 1 <= len(ops) <= 2 else None
                         ) if current else normalize_spec(item["spec"])
            if candidate and family_shape_ok(candidate)[0]:
                candidate = cap_leverage(candidate)
                found[spec_id(candidate)] = {
                    "spec": candidate, "hypothesis": item["hypothesis"][:400],
                    "evidence_refs": refs, "expected_effect": item["expected_effect"][:200],
                }
        except (KeyError, TypeError, AttributeError, ValueError):
            logger.info("Discarded malformed portfolio candidate")
    return list(found.values())


def _review(report: dict) -> dict:
    payload = {
        "instructions": (
            "Audit the portfolio decision using the supplied metrics and evidence IDs. "
            "Identify thin samples, PnL concentration, and disagreement between historical and paper results. "
            "Do not infer outcomes from open entries or pick orders. Suggest at most three testable hypotheses. "
            "Return JSON with summary, insufficient_evidence, and findings and next_hypotheses arrays. "
            "Each array item has text and evidence_refs. Use only IDs in evidence."
        ),
        "evidence": report["evidence"],
        "decision": report["decision"],
    }
    response = _llm_chat("You are a portfolio research reviewer. Return one JSON object.",
                         json.dumps(payload), temperature=0.2, timeout=45)
    if (not isinstance(response, dict) or not isinstance(response.get("summary"), str)
            or not isinstance(response.get("insufficient_evidence"), bool)):
        raise RuntimeError("Portfolio review prompt failed or returned invalid JSON")
    checked = {}
    for key in ("findings", "next_hypotheses"):
        checked[key] = []
        items = response.get(key)
        for item in (items if isinstance(items, list) else [])[:3]:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            refs = item.get("evidence_refs")
            if isinstance(refs, list) and refs and all(isinstance(x, str) and x in report["evidence"] for x in refs):
                checked[key].append({"text": item["text"][:500], "evidence_refs": refs})
    return {
        "summary": str(response.get("summary") or "")[:800],
        "insufficient_evidence": response.get("insufficient_evidence") is True,
        **checked,
    }


def research(state: dict, context: dict) -> tuple[dict | None, dict]:
    symbols = tuple(state["symbols"])
    current = state["champion"]
    feed = MarketData(config.HISTORY_DAYS)
    frames = {symbol: feed.frame(symbol) for symbol in symbols}
    start = max(frame.open_time.iloc[0] for frame in frames.values())
    latest = min(frame.open_time.iloc[-1] for frame in frames.values())
    proposal_context = context
    if context["entries"]:
        first_entry = min(item["entry_time"] for item in context["entries"])
        before_paper = pd.Timestamp(first_entry, unit="ms", tz="UTC") - 2 * interval_td()
        if before_paper - start >= 800 * interval_td():
            latest = min(latest, before_paper)
        else:
            # A sparse batch can span the whole historical sample. Keep its outcomes
            # out of proposal generation when there is no disjoint evaluation window.
            proposal_context = {**context, "entries": [], "closed_count": 0, "previous_hypotheses": []}
    times = pd.date_range(start, latest, freq=interval_td())
    if len(times) < 800:
        raise ValueError("Portfolio research requires at least 800 aligned candles")
    middle = int(len(times) * config.TRAIN_FRACTION)
    split = (middle + len(times)) // 2
    validation = times[middle:split]
    holdout = times[split:]
    if len(validation) < 2 or holdout.empty:
        raise ValueError("Portfolio research windows are empty")
    summary = {
        symbol: {
            "bars": len(frame.loc[frame.open_time < holdout[0]]),
            "last_validation_close": float(frame.loc[frame.open_time < holdout[0], "close"].iloc[-1]),
        }
        for symbol, frame in frames.items()
    }
    candidates = _propose(symbols, proposal_context, current, summary)
    if not candidates and current is None:
        raise RuntimeError("Portfolio genesis produced no valid Spec")
    if current:
        current_id = spec_id(current)
        candidates = [{"spec": current}] + [row for row in candidates if spec_id(row["spec"]) != current_id]
    deadline = time.monotonic() + config.RESEARCH_MAX_MINUTES * 60
    validation_deadline = time.monotonic() + max(1, (deadline - time.monotonic()) * 0.7)
    ranked = []
    for candidate in candidates:
        if ranked and time.monotonic() >= validation_deadline:
            break
        spec = candidate["spec"]
        try:
            data = {symbol: prepared(frame, spec) for symbol, frame in frames.items()}
            windows = np.array_split(np.arange(len(validation)), min(config.WF_FOLDS, len(validation)))
            folds = [simulate(data, spec, validation[indices], validation_deadline) for indices in windows]
        except TimeoutError:
            if not ranked:
                raise
            break
        rank = sum(fold["score"] for fold in folds) / len(folds)
        ranked.append({**candidate, "id": spec_id(spec), "folds": folds, "score": rank,
                       "positive_folds": sum(fold["net_return_pct"] > 0 for fold in folds),
                       "trades": sum(fold["trade_count"] for fold in folds)})
    current_row = ranked[0] if current else None
    qualified = [row for row in ranked if row["positive_folds"] >= min(config.WF_MIN_POSITIVE_FOLDS, len(row["folds"]))
                 and row["trades"] >= config.MIN_VALIDATION_TRADES]
    finalist = max(qualified or (ranked if current is None else []), key=lambda row: row["score"], default=None)
    selected = None
    reason = "no_validation_improvement"
    evidence = {
        "batch": context,
        "validation": [{key: value for key, value in row.items() if key != "spec"} for row in ranked],
    }
    evidence.update({f"entry:{item['entry_seq']}": item for item in context["entries"]})
    for symbol in symbols:
        trades = [item for item in context["entries"] if item["symbol"] == symbol]
        closed = [item for item in trades if item["status"] == "closed"]
        evidence[f"market:{symbol}"] = {
            **summary[symbol], "entries": len(trades), "closed": len(closed),
            "realized_pnl": sum(item["pnl"] for item in closed),
            "ticks": context["markets"].get(symbol, {}),
        }
    if finalist and (not current_row or finalist["score"] > current_row["score"]):
        spec = finalist["spec"]
        challenger = simulate({s: prepared(f, spec) for s, f in frames.items()}, spec, holdout, deadline)
        evidence["holdout:challenger"] = challenger
        incumbent = None
        if current:
            incumbent = simulate({s: prepared(f, current) for s, f in frames.items()}, current, holdout, deadline)
            evidence["holdout:champion"] = incumbent
        if _qualified(challenger, config.MIN_TEST_TRADES) and (not incumbent or challenger["score"] > incumbent["score"]):
            selected = spec
            reason = "qualified_holdout"
        else:
            reason = "holdout_gate_failed"
    report = {
        "end_entry": context["end_entry"], "prompt_version": PROMPT_VERSION, "model": config.LLM_MODEL,
        "decision": "PROMOTE" if selected else "KEEP", "reason": reason,
        "candidate_id": spec_id(selected) if selected else None, "evidence": evidence,
        "validation_window": [str(validation[0]), str(validation[-1])],
        "holdout_window": [str(holdout[0]), str(holdout[-1])],
    }
    report["review"] = _review(report)
    if not selected and not current:
        raise RuntimeError(f"Portfolio genesis could not qualify a shared Spec: {reason}")
    return selected, report
