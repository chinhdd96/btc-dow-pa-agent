"""CLI: python scripts/run_replay.py  — 30d historical paper replay."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from modules.binance_client import BinanceClient, BinanceClientError
from modules.llm_agent import LLMAgent
from modules.notifier import Notifier
from modules.replay_engine import run_replay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    notifier = Notifier()
    llm = LLMAgent()
    try:
        binance = BinanceClient()
    except BinanceClientError as exc:
        print(f"Binance init failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    summary = run_replay(binance, llm, notifier, days=config.REPLAY_DAYS)
    print(json_dumps(summary))


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
