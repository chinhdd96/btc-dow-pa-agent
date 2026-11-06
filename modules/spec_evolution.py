"""Evolve / evaluate Strategy Specs (catalog-based Champion path)."""

from __future__ import annotations

import logging
import random
import time
from copy import deepcopy
from typing import Any

import pandas as pd

import config
from modules.backtest_engine import evaluate_spec
from modules.champion_store import get_paper_state
from modules.features import add_features as add_base_features
from modules.strategy_spec import (
    apply_patch,
    base_name,
    default_seed_specs,
    loosen_spec_for_starve,
    normalize_spec,
    spec_id,
    validate_spec,
    variant_name,
)

logger = logging.getLogger(__name__)


def maybe_promote_spec(
    ranked: list[dict],
    current: dict | None,
    *,
    starving: bool = False,
    paper_trade_count: int = 0,
) -> dict | None:
    hard = [r for r in ranked if r["qualified"]]
    soft = [r for r in ranked if r.get("soft_qualified") and not r["qualified"]]
    pool = hard or (soft if config.PAPER_SOFT_PROMOTE else [])
    if not pool:
        logger.warning("No Spec passed validation — keep current or stay temporarily flat")
        return normalize_spec(current) if current else None
    best = pool[0]
    mode = "hard" if best["qualified"] else "soft"
    if current:
        old_sid = spec_id(current)
        if starving and paper_trade_count <= 0 and best["spec_id"] != old_sid:
            logger.info(
                "Starve-replace spec %s → %s (%s)",
                old_sid,
                best["spec_id"],
                mode,
            )
            return normalize_spec(best["spec"])
        old = next(
            (
                r
                for r in ranked
                if r["spec_id"] == old_sid
                and (r["qualified"] or r.get("soft_qualified"))
            ),
            None,
        )
        margin = 3 if best["qualified"] else 8
        if starving and paper_trade_count <= 0:
            margin = 1 if best["qualified"] else 3
        # PnL-first: slightly lower bar when paper expectancy poor (explore)
        if old and best["robust_score"] < old["robust_score"] + margin:
            return normalize_spec(current)
    logger.info(
        "Promote Spec %s id=%s robust=%.2f",
        mode,
        best["spec_id"],
        best["robust_score"],
    )
    return normalize_spec(best["spec"])


def _mutate_spec(spec: dict, rng: random.Random) -> dict:
    s = normalize_spec(spec)
    # Mutate exit params
    if rng.random() < 0.4:
        s["exit"]["atr_stop"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    if rng.random() < 0.4:
        s["exit"]["reward_r"] = rng.choice([1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0])
    if rng.random() < 0.3:
        s["exit"]["leverage"] = rng.choice(list(config.LEVERAGE_CANDIDATES))
    if rng.random() < 0.25:
        s["mtf_filter"] = not s["mtf_filter"]
    # Mutate one filter/entry threshold if present
    for key in ("filters", "entry_long", "entry_short"):
        block = s.get(key) or []
        if block and rng.random() < 0.35:
            idx = rng.randrange(len(block))
            cond = deepcopy(block[idx])
            if cond.get("value") is not None and not isinstance(cond["value"], list):
                try:
                    v = float(cond["value"])
                    cond["value"] = round(v * rng.choice([0.85, 0.9, 1.1, 1.15]), 6)
                    block[idx] = cond
                    s[key] = block
                except (TypeError, ValueError):
                    pass
            break
    s["name"] = variant_name(s.get("name"), "m")
    ok, _ = validate_spec(s)
    return s if ok else normalize_spec(spec)


class SpecEvolution:
    def __init__(
        self,
        *,
        llm_specs: list[dict] | None = None,
        seed_specs: list[dict] | None = None,
        patch_candidates: list[dict] | None = None,
    ) -> None:
        self.rng = random.Random(config.RESEARCH_RANDOM_SEED)
        self.llm_specs = llm_specs or []
        self.seed_specs = seed_specs or []
        self.patch_candidates = patch_candidates or []

    def research(self, df: pd.DataFrame) -> list[dict]:
        population: list[dict] = []
        population.extend(default_seed_specs())
        population.extend(self.seed_specs)
        population.extend(self.llm_specs)
        population.extend(self.patch_candidates)

        persisted = get_paper_state()
        cur = persisted.get("champion_spec") or None
        if not cur and persisted.get("champion"):
            from modules.strategy_spec import genome_dict_to_spec

            try:
                cur = genome_dict_to_spec(persisted["champion"])
            except Exception:
                cur = None
        if cur:
            population.append(normalize_spec(cur))
            population.append(loosen_spec_for_starve(cur))

        # Deduplicate by spec_id while validating
        uniq: dict[str, dict] = {}
        for raw in population:
            try:
                s = normalize_spec(raw)
                ok, _ = validate_spec(s)
                if not ok:
                    continue
                uniq[spec_id(s)] = s
            except Exception:
                continue

        pop_size = max(8, config.POPULATION_SIZE // 2)
        specs = list(uniq.values())
        attempts = 0
        while len(uniq) < pop_size and specs and attempts < pop_size * 5:
            attempts += 1
            child = _mutate_spec(self.rng.choice(specs), self.rng)
            uniq[spec_id(child)] = child
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
            seen_family = {base_name(e.get("name")) for e in elites}
            for r in results[elite_n:]:
                fam = base_name(r["spec"].get("name"))
                if fam not in seen_family and len(elites) < pop_size // 2:
                    seen_family.add(fam)
                    elites.append(normalize_spec(r["spec"]))
            nxt: dict[str, dict] = {spec_id(e): e for e in elites}
            attempts = 0
            while len(nxt) < pop_size and attempts < pop_size * 5:
                attempts += 1
                child = _mutate_spec(self.rng.choice(elites), self.rng)
                nxt.setdefault(spec_id(child), child)
            specs = list(nxt.values())

        ranked = sorted(
            all_results.values(), key=lambda z: z["robust_score"], reverse=True
        )
        return ranked


def patched_specs_from_llm(
    current: dict,
    patches: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for p in patches[:5]:
        if not isinstance(p, dict):
            continue
        nxt = apply_patch(current, p)
        if nxt:
            out.append(nxt)
    return out
