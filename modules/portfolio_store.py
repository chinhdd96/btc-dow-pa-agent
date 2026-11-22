"""SQLite ledger, review checkpoints, and single-writer ownership."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import config
from modules.portfolio_engine import Portfolio, new_state


@contextmanager
def writer_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Portfolio writer already running") from exc
        yield


class PortfolioStore:
    def __init__(self, directory: Path, symbols: tuple[str, ...], *, create: bool) -> None:
        self.directory = directory
        path = directory / "portfolio.sqlite3"
        if not path.exists() and not create:
            raise RuntimeError("Portfolio DB missing. Verify the /data mount before initializing.")
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY CHECK (id = 1), state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entries (
                seq INTEGER PRIMARY KEY, opened TEXT NOT NULL, closed TEXT
            );
            CREATE TABLE IF NOT EXISTS reviews (
                end_entry INTEGER PRIMARY KEY, report TEXT NOT NULL
            );
        """)
        row = self.db.execute("SELECT state FROM portfolio WHERE id = 1").fetchone()
        if row is None and not create:
            raise RuntimeError("Portfolio state missing. Restore a verified backup.")
        self.state = json.loads(row[0]) if row else new_state(symbols, config.INITIAL_CAPITAL)
        self.state.setdefault("review_every", config.REVIEW_EVERY_N_ENTRIES)
        if self.state["symbols"] != list(symbols) or self.state["interval"] != config.TIMEFRAME_MAIN:
            raise RuntimeError("Portfolio symbols/order or timeframe differs from its stored configuration")
        if self.state["review_every"] != config.REVIEW_EVERY_N_ENTRIES:
            raise RuntimeError("Review cadence differs from its stored configuration")
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO portfolio VALUES (1, ?)", (json.dumps(self.state),))

    def save(self, state: dict, opened=(), closed=(), *, end_entry: int | None = None, report=None) -> None:
        with self.db:
            for entry in opened:
                self.db.execute("INSERT INTO entries VALUES (?, ?, NULL)", (entry["entry_seq"], json.dumps(entry)))
            for close in closed:
                updated = self.db.execute(
                    "UPDATE entries SET closed = ? WHERE seq = ? AND closed IS NULL",
                    (json.dumps(close), close["entry_seq"]),
                ).rowcount
                if updated != 1:
                    raise RuntimeError(f"Unmatched close for entry {close['entry_seq']}")
            self.db.execute("UPDATE portfolio SET state = ? WHERE id = 1", (json.dumps(state),))
            if end_entry is not None:
                self.db.execute("INSERT INTO reviews VALUES (?, ?)", (end_entry, json.dumps(report)))
        self.state = state

    def next_review(self) -> int | None:
        if self.state["champion"] is None:
            return 0
        current = self.db.execute("SELECT COALESCE(MAX(end_entry), 0) FROM reviews").fetchone()[0]
        next_entry = current + config.REVIEW_EVERY_N_ENTRIES
        return next_entry if self.state["entries"] >= next_entry else None

    def review_context(self, end_entry: int) -> dict:
        low = max(0, end_entry - config.REVIEW_EVERY_N_ENTRIES)
        rows = self.db.execute(
            "SELECT opened, closed FROM entries WHERE seq > ? AND seq <= ? ORDER BY seq", (low, end_entry)
        )
        batch = []
        for opened, closed in rows:
            row = json.loads(closed or opened)
            item = {key: row.get(key) for key in (
                "entry_seq", "symbol", "version", "side", "entry_context", "entry_time",
                "exit_reason", "pnl", "r_multiple", "exit_time",
            )}
            item["status"] = "closed" if closed else "open"
            batch.append(item)
        last = self.db.execute("SELECT report FROM reviews ORDER BY end_entry DESC LIMIT 1").fetchone()
        prior = (json.loads(last[0]).get("review") or {}) if last else {}
        wallet = Portfolio(self.state)
        return {
            "end_entry": end_entry, "entries": batch,
            "closed_count": sum(item["status"] == "closed" for item in batch),
            "cash": self.state["cash"], "previous_hypotheses": prior.get("next_hypotheses", []),
            "equity": wallet.equity(), "available_margin": wallet.available,
            "champion_version": self.state["version"],
            "markets": {symbol: market.get("ticks", {}) for symbol, market in self.state["markets"].items()},
        }

    def backup(self) -> None:
        temporary = self.directory / "portfolio.backup.tmp"
        with closing(sqlite3.connect(temporary)) as destination:
            self.db.backup(destination)
        temporary.replace(self.directory / "portfolio.backup.sqlite3")
