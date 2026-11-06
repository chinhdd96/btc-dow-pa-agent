"""Evaluate versions of the single indicator family.

Only the LLM adds / removes / replaces indicators. Python mutation here just
tunes numbers (thresholds, exits) of the candidates it is given, so every
Spec in the population keeps the LLM-chosen indicator set (3–5 types).
"""

from __future__ import annotations

import logging
import random
import time
from copy import deepcopy

import pandas as pd

import config
from modules.features import add_features as add_base_features
from modules.spec_backtest import evaluate_spec
from modules.strategy_spec import (
    base_name,
    family_shape_ok,
    loosen_spec_for_starve,
    normalize_spec,
    spec_id,
)

logger = logging.getLogger(__name__)


def _row(ranked: list[dict], spec: dict | None) -> dict | None:
    if not spec:
        return None
    sid = spec_id(spec)
    return next((r for r in ranked if r["spec_id"] == sid), None)


def choose_version(
    ranked: list[dict],
    current: dict | None,
    *,
    healthy_on_paper: bool,
    starving: bool = False,
) -> tuple[dict | None, str]:
    """Pick the family version to run. Returns (spec, decision).

    decision: genesis | promote | keep | forward (better candidate must prove itself as a shadow).
    """
    hard = [r for r in ranked if r["qualified"]]
    soft = [r for r in ranked if r.get("soft_qualified") and not r["qualified"]]
    pool = hard or (soft if config.PAPER_SOFT_PROMOTE else [])

    if current is None:
        # PnL-first: genesis always yields a family; unqualified drafts run in explore mode
        best = (pool or ranked or [None])[0]
        return (normalize_spec(best["spec"]), "genesis") if best else (None, "none")

    if not pool:
        return normalize_spec(current), "keep"
    best = pool[0]
    cur_row = _row(ranked, current)
    if cur_row and best["spec_id"] == cur_row["spec_id"]:
        return normalize_spec(current), "keep"
    cur_score = float(cur_row["robust_score"]) if cur_row else float("-inf")

    if starving:
        margin = 0.0
    elif cur_row and cur_row["qualified"] and healthy_on_paper:
        if best["robust_score"] < cur_score + config.FAMILY_DIRECT_MARGIN:
            return normalize_spec(current), "forward"
        margin = config.FAMILY_DIRECT_MARGIN
    else:
        margin = 1.0
    if best["robust_score"] <= cur_score + margin:
        return normalize_spec(current), "keep"
    logger.info(
        "Promote family version id=%s robust=%.2f (current %.2f)",
        best["spec_id"],
        best["robust_score"],
        cur_score,
    )
    return normalize_spec(best["spec"]), "promote"


def _mutate_spec(spec: dict, rng: random.Random) -> dict:
    """Numeric tuning only — never changes which indicators are used."""
    s = normalize_spec(spec)
    if rng.random() < 0.4:
        s["exit"]["atr_stop"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    if rng.random() < 0.4:
        s["exit"]["reward_r"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0])
    if rng.random() < 0.3:
        s["exit"]["leverage"] = rng.choice(list(config.LEVERAGE_CANDIDATES))
    if rng.random() < 0.2:
        s["exit"]["breakeven_r"] = rng.choice([0, 0, 0.8, 1.0, 1.5])
    if rng.random() < 0.2:
        s["exit"]["trail_atr"] = rng.choice([0, 0, 1.5, 2.0, 3.0])
    if rng.random() < 0.15:
        s["exit"]["max_bars"] = rng.choice([0, 0, 16, 32, 64])
    if rng.random() < 0.1:
        s["exit"]["exit_on_opposite"] = not s["exit"].get("exit_on_opposite")
    for key in ("filters", "entry_long", "entry_short"):
        block = s.get(key) or []
        if block and rng.random() < 0.35:
            idx = rng.randrange(len(block))
            cond = deepcopy(block[idx])
            if cond.get("value") is not None and not isinstance(cond["value"], list):
                try:
                    cond["value"] = round(float(cond["value"]) * rng.choice([0.85, 0.9, 1.1, 1.15]), 6)
                    block[idx] = cond
                    s[key] = block
                except (TypeError, ValueError):
                    pass
            break
    meta = dict(s.get("variant") or {})
    meta["source"] = f"{meta.get('source') or 'current'}+tune"
    s["variant"] = meta
    s["name"] = base_name(s.get("name"))
    return normalize_spec(s) if family_shape_ok(s)[0] else normalize_spec(spec)


class SpecEvolution:
    def __init__(
        self,
        *,
        candidates: list[dict],
        current: dict | None = None,
    ) -> None:
        self.rng = random.Random(config.RESEARCH_RANDOM_SEED)
        self.candidates = candidates
        self.current = current

    def research(self, df: pd.DataFrame) -> list[dict]:
        population = list(self.candidates)
        if self.current:
            population += [normalize_spec(self.current), loosen_spec_for_starve(self.current)]

        uniq: dict[str, dict] = {}
        for raw in population:
            try:
                s = normalize_spec(raw)
            except Exception:  # noqa: BLE001
                continue
            ok, msg = family_shape_ok(s)
            if not ok:
                logger.info("candidate %s skipped: %s", s.get("name"), msg)
                continue
            uniq.setdefault(spec_id(s), s)
        if not uniq:
            return []

        pop_size = max(8, config.POPULATION_SIZE // 2)
        specs = list(uniq.values())
        attempts = 0
        while len(uniq) < pop_size and attempts < pop_size * 5:
            attempts += 1
            child = _mutate_spec(self.rng.choice(specs), self.rng)
            uniq.setdefault(spec_id(child), child)
        specs = list(uniq.values())

        base = add_base_features(df)
        all_results: dict[str, dict] = {}
        n_gens = max(1, min(config.GENERATIONS, config.SPEC_MAX_GENERATIONS))
        deadline = time.monotonic() + config.RESEARCH_MAX_MINUTES * 60
        for gen in range(n_gens):
            results = []
            for s in specs:
                sid = spec_id(s)
                if sid not in all_results:
                    all_results[sid] = evaluate_spec(df, s, base=base)
                results.append(all_results[sid])
            results.sort(key=lambda z: z["robust_score"], reverse=True)
            logger.info(
                "Spec gen %d/%d pop=%d evaluated=%d best=%.2f qualified=%s",
                gen + 1,
                n_gens,
                len(specs),
                len(all_results),
                results[0]["robust_score"] if results else -999,
                sum(1 for r in results if r["qualified"]),
            )
            if time.monotonic() > deadline:
                logger.warning(
                    "Spec research time budget %sm reached at gen %d",
                    config.RESEARCH_MAX_MINUTES,
                    gen + 1,
                )
                break
            elite_n = max(2, int(pop_size * config.ELITE_FRACTION))
            elites = [normalize_spec(r["spec"]) for r in results[:elite_n]]
            # Genesis drafts differ in indicator set: keep the best of each draft alive
            seen = {base_name(e.get("name")) for e in elites}
            for r in results[elite_n:]:
                name = base_name(r["spec"].get("name"))
                if name not in seen and len(elites) < pop_size // 2:
                    seen.add(name)
                    elites.append(normalize_spec(r["spec"]))
            nxt: dict[str, dict] = {spec_id(e): e for e in elites}
            attempts = 0
            while len(nxt) < pop_size and attempts < pop_size * 5:
                attempts += 1
                child = _mutate_spec(self.rng.choice(elites), self.rng)
                nxt.setdefault(spec_id(child), child)
            specs = list(nxt.values())

        return sorted(all_results.values(), key=lambda z: z["robust_score"], reverse=True)
