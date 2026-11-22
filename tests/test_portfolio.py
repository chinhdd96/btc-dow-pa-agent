import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from modules import portfolio_research as research
from modules import quant_data
from modules.portfolio_engine import Portfolio, new_state
from modules.portfolio_store import PortfolioStore, writer_lock
from modules.portfolio_worker import _paper_tick
from modules.paper_champion import interval_td
from modules.strategy_spec import normalize_spec


SPEC = normalize_spec({
    "name": "fixture",
    "entry_long": [{"ind": "close", "op": "gt", "value": 0}],
    "exit": {"atr_stop": 2, "reward_r": 2, "leverage": 2},
})


class PortfolioTests(unittest.TestCase):
    def test_wallet_rejects_second_fill_when_margin_is_unavailable(self):
        portfolio = Portfolio(new_state(("AAA", "BBB"), 1000))
        for symbol in ("AAA", "BBB"):
            portfolio.signal(symbol, side=1, atr=1.0, spec=SPEC, signal_time=0, fill_time=60_000)
        with patch.object(config, "MAX_MARGIN_FRACTION", 0.6):
            self.assertTrue(portfolio.fill("AAA", 60_000, 100))
            self.assertFalse(portfolio.fill("BBB", 60_000, 100))
        self.assertEqual(portfolio.state["entries"], 1)
        self.assertGreater(portfolio.available, 0)
        portfolio.funding("AAA", 60_000, 0.001)
        cash = portfolio.state["cash"]
        portfolio.funding("AAA", 60_000, 0.001)
        self.assertEqual(portfolio.state["cash"], cash)
        portfolio.close("AAA", 120_000, "TP", 104)
        self.assertAlmostEqual(portfolio.state["cash"] - 1000, portfolio.closed[0]["pnl"])

    def test_review_boundary_survives_restart_and_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaises(RuntimeError):
                PortfolioStore(path, ("AAA", "BBB"), create=False)
            with writer_lock(path):
                store = PortfolioStore(path, ("AAA", "BBB"), create=True)
                store.save({**store.state, "champion": SPEC, "version": 1}, end_entry=0, report={})
                for i in range(1, 12):
                    portfolio = Portfolio(deepcopy(store.state))
                    symbol = ("AAA", "BBB")[i % 2]
                    portfolio.signal(symbol, side=1, atr=1.0, spec=SPEC, signal_time=i, fill_time=i + 1)
                    self.assertTrue(portfolio.fill(symbol, i + 1, 100))
                    if i != 10:
                        portfolio.close(symbol, i + 2, "TP", 104)
                    store.save(portfolio.state, portfolio.opened, portfolio.closed)
                    if i == 9:
                        self.assertIsNone(store.next_review())
                        store.db.close()
                        store = PortfolioStore(path, ("AAA", "BBB"), create=False)
                    elif i == 10:
                        self.assertEqual(store.next_review(), 10)
                self.assertEqual(store.next_review(), 10)
                batch = store.review_context(10)
                self.assertEqual(batch["closed_count"], 9)
                self.assertEqual([entry["entry_seq"] for entry in batch["entries"]], list(range(1, 11)))
                store.save(store.state, end_entry=10, report={"decision": "KEEP"})
                store.backup()
                store.db.close()
            (path / "portfolio.backup.sqlite3").replace(path / "portfolio.sqlite3")
            store = PortfolioStore(path, ("AAA", "BBB"), create=False)
            self.assertEqual(store.state["entries"], 11)
            self.assertIsNone(store.next_review())
            self.assertIn("position", store.state["markets"]["AAA"])
            store.db.close()

    def test_aligned_backtest_charges_same_wallet(self):
        times = pd.date_range("2026-10-01", periods=10, freq=interval_td(), tz="UTC")
        raw = pd.DataFrame({
            "open_time": times, "open": 100.0, "high": 104.5, "low": 99.0,
            "close": 103.0, "atr": 1.0, "funding_event_rate": np.nan,
        }).set_index("open_time", drop=False)
        with patch.object(config, "MAX_MARGIN_FRACTION", 0.75):
            report = research.simulate({"AAA": raw, "BBB": raw}, SPEC, times[1:], math.inf)
        self.assertGreater(report["symbols"]["AAA"]["trades"], 0)
        self.assertEqual(report["symbols"]["BBB"]["trades"], 0)

    def test_daily_warmup_and_incomplete_bias(self):
        times = pd.date_range("2026-01-01", periods=75, freq="D", tz="UTC")
        raw = pd.DataFrame({"open_time": times, "close": range(1, 76)})
        calls = []

        def feed(interval, days, symbol):
            calls.append((interval, days))
            return raw

        enriched = quant_data.enrich_multi_timeframe(raw, "AAA", 14, fetch_higher=feed, strict=True)
        self.assertTrue(pd.isna(enriched.iloc[25]["htf1d_bull"]))
        self.assertTrue(enriched.iloc[-1]["htf1d_bull"])
        self.assertGreaterEqual(dict(calls)["1d"], 60)

    def test_prompt_review_rejects_references_outside_input(self):
        response = {
            "summary": "Thin sample", "insufficient_evidence": True,
            "findings": [
                {"text": "Realized count known", "evidence_refs": ["batch"]},
                {"text": "Invented", "evidence_refs": ["entry:999"]},
            ],
        }
        with patch.object(research, "_llm_chat", return_value=response):
            report = research._review({"decision": "KEEP", "evidence": {"batch": {"closed_count": 3}}})
        self.assertTrue(report["insufficient_evidence"])
        self.assertEqual(len(report["findings"]), 1)

    def test_research_runs_holdout_after_candidate_selection(self):
        times = pd.date_range("2026-10-01", periods=1000, freq=interval_td(), tz="UTC")
        frame = pd.DataFrame({
            "open_time": times, "open": 100.0, "high": 104.5, "low": 99.0,
            "close": 103.0, "atr": 1.0, "funding_event_rate": np.nan,
        })
        context = {"entries": [], "end_entry": 0, "markets": {}}
        with patch.object(research.MarketData, "frame", return_value=frame):
            with patch.object(research, "prepared", side_effect=lambda data, spec: data.set_index("open_time", drop=False)):
                with patch.object(research, "_propose", return_value=[{"spec": SPEC}]):
                    with patch.object(research, "_review", return_value={"summary": "fixture"}):
                        selected, report = research.research(new_state(("AAA", "BBB"), 1000), context)
        self.assertEqual(selected, SPEC)
        self.assertEqual(report["decision"], "PROMOTE")
        self.assertLess(report["validation_window"][-1], report["holdout_window"][0])
        self.assertEqual(set(report["evidence"]["holdout:challenger"]["symbols"]), {"AAA", "BBB"})

    def test_paper_tick_applies_simultaneous_opens_in_configured_order(self):
        state = new_state(("BBB", "AAA"), 1000)
        state.update(champion=SPEC, version=1)

        def fake_market_event(symbol, *_):
            main = (0, 101.0, 99.0, 100.0, 1.0, 1.0, 1, 1, {}, True, "")
            return [(900_000, "open", symbol, 100.0), (900_000, "main", symbol, main)]

        with patch("modules.portfolio_worker._market_events", side_effect=fake_market_event):
            with patch.object(config, "MAX_MARGIN_FRACTION", 0.6):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    engine = _paper_tick(state, None, pool)
        self.assertEqual([entry["symbol"] for entry in engine.opened], ["BBB"])
