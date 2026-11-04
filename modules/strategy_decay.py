"""Detect Champion decay from recent paper trades → early research."""

from __future__ import annotations

import logging

import config
from modules import champion_store

logger = logging.getLogger(__name__)


def recent_expectancy(limit: int | None = None) -> tuple[float | None, int]:
    limit = limit or max(config.DECAY_MIN_TRADES, 12)
    trades = champion_store.list_paper_trades(limit=limit)
    if not trades:
        return None, 0
    # list is newest-first; keep chronological for clarity
    pnls = [float(t.get("pnl") or 0.0) for t in reversed(trades)]
    n = len(pnls)
    if n == 0:
        return None, 0
    return sum(pnls) / n, n


def check_decay(state: dict | None = None) -> dict:
    """
    Return {decay: bool, reason: str, expectancy: float|None, n: int}.
    Marks research_due on paper_state when decay detected.
    """
    state = state if state is not None else champion_store.get_paper_state()
    exp, n = recent_expectancy()
    baseline = state.get("champion_baseline_expectancy")
    result = {
        "decay": False,
        "reason": "",
        "expectancy": exp,
        "n": n,
        "baseline": baseline,
    }
    if exp is None or n < config.DECAY_MIN_TRADES:
        return result

    # Absolute: rolling paper expectancy <= threshold (default 0)
    if exp <= config.DECAY_EXPECTANCY_RATIO:
        result["decay"] = True
        result["reason"] = (
            f"rolling_expectancy={exp:.4f} <= {config.DECAY_EXPECTANCY_RATIO} "
            f"over {n} trades"
        )
    # Relative: far below promote-time baseline when baseline was positive
    elif (
        baseline is not None
        and float(baseline) > 0
        and exp < float(baseline) * 0.25
    ):
        result["decay"] = True
        result["reason"] = (
            f"expectancy={exp:.4f} << baseline={float(baseline):.4f} over {n} trades"
        )

    if result["decay"]:
        logger.warning("Champion decay detected: %s", result["reason"])
        state["research_due"] = True
        champion_store.set_paper_state(state)
    return result
