"""SQLite persistence for champion research + paper state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _db_path() -> str:
    path = Path(config.RESEARCH_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def init_db(path: str | None = None) -> str:
    db = path or _db_path()
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS paper_trades "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS research_runs "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, payload TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS paper_state "
            "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
        )
    return db


def db_set(key: str, value: Any, path: str | None = None) -> None:
    db = path or _db_path()
    init_db(db)
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, default=str)),
        )


def db_get(key: str, default: Any = None, path: str | None = None) -> Any:
    db = path or _db_path()
    init_db(db)
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def save_research(report: dict, path: str | None = None) -> None:
    db = path or _db_path()
    init_db(db)
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO research_runs(ts,payload) VALUES(?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(report, default=str),
            ),
        )


def insert_paper_trade(trade: dict, path: str | None = None) -> None:
    db = path or _db_path()
    init_db(db)
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO paper_trades(payload) VALUES(?)",
            (json.dumps(trade, default=str),),
        )


def list_paper_trades(limit: int = 50, path: str | None = None) -> list[dict]:
    db = path or _db_path()
    init_db(db)
    with sqlite3.connect(db) as con:
        rows = con.execute(
            "SELECT payload FROM paper_trades ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: list[dict] = []
    for (payload,) in rows:
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out


def replace_paper_trades(trades: list[dict], path: str | None = None) -> None:
    """Replace paper_trades table (used by champion snapshot restore)."""
    db = path or _db_path()
    init_db(db)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM paper_trades")
        for trade in trades:
            con.execute(
                "INSERT INTO paper_trades(payload) VALUES(?)",
                (json.dumps(trade, default=str),),
            )


def get_paper_state(path: str | None = None) -> dict:
    return db_get(
        "paper_state",
        {
            "equity": config.INITIAL_CAPITAL,
            "position": None,
            "pending": None,
            "last_candle": None,
            "champion": None,
            "trade_count": 0,
            "champion_baseline_expectancy": None,
            "research_due": False,
            "challenger_hints": [],
        },
        path=path,
    ) or {}


def set_paper_state(state: dict, path: str | None = None) -> None:
    db_set("paper_state", state, path=path)
