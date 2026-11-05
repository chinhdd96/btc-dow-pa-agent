"""Strategy population evolution + champion promotion."""

from __future__ import annotations

import logging
import random
from typing import Optional

import pandas as pd

import config
from modules.backtest_engine import evaluate_genome
from modules.champion_store import get_paper_state
from modules.genome import Genome, crossover, mutate, random_genome

logger = logging.getLogger(__name__)


def maybe_promote(
    ranked: list[dict],
    current: Optional[dict],
) -> Optional[Genome]:
    hard = [r for r in ranked if r["qualified"]]
    soft = [
        r
        for r in ranked
        if r.get("soft_qualified") and not r["qualified"]
    ]
    pool = hard or (soft if config.PAPER_SOFT_PROMOTE else [])
    if not pool:
        logger.warning(
            "No candidate passed validation (hard/soft). Keep current Champion or stay flat."
        )
        return Genome(**current) if current else None
    best = pool[0]
    mode = "hard" if best["qualified"] else "soft"
    if current:
        old_id = Genome(**current).genome_id
        old = next(
            (
                r
                for r in ranked
                if r["genome_id"] == old_id
                and (r["qualified"] or r.get("soft_qualified"))
            ),
            None,
        )
        # Soft challenger must beat current by more to avoid thrash
        margin = 3 if best["qualified"] else 8
        if old and best["robust_score"] < old["robust_score"] + margin:
            return Genome(**current)
    logger.info(
        "Promote %s champion=%s robust=%.2f soft=%s",
        mode,
        best["genome_id"],
        best["robust_score"],
        bool(best.get("soft_qualified")),
    )
    return Genome(**best["genome"])


class StrategyEvolution:
    def __init__(
        self,
        *,
        llm_proposals: list[Genome] | None = None,
        seed_genomes: list[Genome] | None = None,
    ) -> None:
        self.rng = random.Random(config.RESEARCH_RANDOM_SEED)
        self.llm_proposals = llm_proposals or []
        self.seed_genomes = seed_genomes or []

    def research(self, df: pd.DataFrame) -> list[dict]:
        population = [
            random_genome(self.rng) for _ in range(config.POPULATION_SIZE)
        ]
        population.extend(self.llm_proposals)
        population.extend(self.seed_genomes)

        persisted = get_paper_state()
        if persisted.get("champion"):
            try:
                population.append(Genome(**persisted["champion"]))
            except Exception:
                pass

        hints = persisted.get("challenger_hints") or []
        for hint in hints[-8:]:
            if isinstance(hint, dict):
                try:
                    from modules.genome import genome_from_dict

                    g = genome_from_dict(hint)
                    if g:
                        population.append(g)
                except Exception:
                    continue

        all_results: dict[str, dict] = {}
        for gen in range(config.GENERATIONS):
            results = []
            for g in population:
                if g.genome_id not in all_results:
                    all_results[g.genome_id] = evaluate_genome(df, g)
                results.append(all_results[g.genome_id])
            results.sort(key=lambda z: z["robust_score"], reverse=True)
            logger.info(
                "Generation %d/%d | best=%s | qualified=%s",
                gen + 1,
                config.GENERATIONS,
                results[0]["robust_score"],
                sum(r["qualified"] for r in results),
            )
            elite_n = max(2, int(config.POPULATION_SIZE * config.ELITE_FRACTION))
            elites = [Genome(**r["genome"]) for r in results[:elite_n]]
            new_pop = elites[:]
            while len(new_pop) < config.POPULATION_SIZE:
                if self.rng.random() < 0.25:
                    child = random_genome(self.rng)
                else:
                    p1, p2 = self.rng.choice(elites), self.rng.choice(elites)
                    child = crossover(p1, p2, self.rng)
                    if self.rng.random() < config.MUTATION_RATE:
                        child = mutate(child, self.rng)
                new_pop.append(child)
            population = new_pop
        ranked = sorted(
            all_results.values(), key=lambda z: z["robust_score"], reverse=True
        )
        return ranked
