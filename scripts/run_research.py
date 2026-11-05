#!/usr/bin/env python3
"""One-shot Champion research run (debug)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from modules.research_pipeline import run_research


def main() -> None:
    p = argparse.ArgumentParser(description="Run one Champion research cycle")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--population", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.days is not None:
        config.HISTORY_DAYS = args.days
    if args.generations is not None:
        config.GENERATIONS = args.generations
    if args.population is not None:
        config.POPULATION_SIZE = args.population
    report = run_research(use_llm=not args.no_llm)
    print(
        json.dumps(
            {
                "champion_spec": report.get("champion_spec"),
                "champion_spec_id": report.get("champion_spec_id")
                or (report.get("champion_evaluation") or {}).get("spec_id"),
                "champion_evaluation": report.get("champion_evaluation"),
                "ai_review": report.get("ai_review"),
                "top_candidates": (report.get("top_candidates") or [])[:5],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
