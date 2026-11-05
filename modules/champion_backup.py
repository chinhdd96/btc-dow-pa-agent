"""Backup/restore Champion paper state + tick_history via GitHub (Blitz rebuild-safe).

Uses the same HISTORY_GITHUB_TOKEN / history-data branch as decision_history.
One JSON snapshot — no binary SQLite upload.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import config
from modules import champion_store
from modules import tick_history

logger = logging.getLogger(__name__)

_last_backup_ts = 0.0
_MIN_BACKUP_GAP_SEC = 60.0


def _headers() -> dict[str, str] | None:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return None
    return {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "btc-dow-pa-agent-champion",
    }


def _snapshot() -> dict[str, Any]:
    champion_store.init_db()
    state = champion_store.get_paper_state()
    trades = champion_store.list_paper_trades(limit=100)
    # list is newest-first; store oldest-first for replay clarity
    trades = list(reversed(trades))
    ticks = tick_history.load_recent(n=config.TICK_HISTORY_MAX)
    return {
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "symbol": config.SYMBOL,
        "timeframe": config.TIMEFRAME_MAIN,
        "paper_state": state,
        "paper_trades": trades,
        "tick_history": ticks,
    }


def _local_is_empty() -> bool:
    champion_store.init_db()
    st = champion_store.get_paper_state()
    has_champ = bool(st.get("champion"))
    trades = int(st.get("trade_count") or 0)
    ticks = tick_history.load_recent(n=5)
    # Fresh DB default equity + no champion + no ticks → treat as empty
    return (not has_champ) and trades <= 0 and not ticks


def _apply_snapshot(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict) or data.get("version") != 1:
        logger.warning("champion snapshot: bad version/payload")
        return False
    state = data.get("paper_state")
    if not isinstance(state, dict):
        return False
    champion_store.init_db()
    champion_store.set_paper_state(state)
    trades = data.get("paper_trades") or []
    if isinstance(trades, list):
        champion_store.replace_paper_trades(
            [t for t in trades if isinstance(t, dict)]
        )
    ticks = data.get("tick_history") or []
    if isinstance(ticks, list) and ticks:
        path = Path(config.PATHS["TICK_HISTORY"])
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = ticks[-config.TICK_HISTORY_MAX :]
        with path.open("w", encoding="utf-8") as f:
            for row in keep:
                if isinstance(row, dict):
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    logger.info(
        "champion snapshot applied: champion=%s trades=%s ticks=%s equity=%s",
        bool(state.get("champion")),
        len(trades) if isinstance(trades, list) else 0,
        len(ticks) if isinstance(ticks, list) else 0,
        state.get("equity"),
    )
    return True


def backup_remote(*, force: bool = False) -> bool:
    """Upsert champion_snapshot.json to history-data branch."""
    global _last_backup_ts
    now = time.time()
    if not force and (now - _last_backup_ts) < _MIN_BACKUP_GAP_SEC:
        return False
    headers = _headers()
    if not headers:
        return False
    payload = _snapshot()
    # Skip empty fresh state (avoid wiping remote with blank after failed restore)
    st = payload.get("paper_state") or {}
    if not st.get("champion") and int(st.get("trade_count") or 0) <= 0:
        ticks = payload.get("tick_history") or []
        if not ticks:
            logger.info("champion backup skip: empty local state")
            return False

    repo = config.HISTORY_GITHUB_REPO
    remote_path = config.CHAMPION_GITHUB_PATH
    branch = config.HISTORY_GITHUB_BRANCH or "history-data"
    content_b64 = base64.b64encode(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    ).decode("ascii")
    try:
        from modules.decision_history import (
            _ensure_history_branch,
            _github_contents_url,
        )

        if not _ensure_history_branch(headers, repo, branch):
            return False
        meta = requests.get(
            _github_contents_url(repo, remote_path, branch),
            headers=headers,
            timeout=30,
        )
        sha = None
        if meta.status_code == 200:
            sha = meta.json().get("sha")
        elif meta.status_code != 404:
            logger.warning("champion remote GET HTTP %s", meta.status_code)
            return False
        body: dict[str, Any] = {
            "message": "chore: sync champion snapshot for Blitz persistence",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        put = requests.put(
            _github_contents_url(repo, remote_path),
            headers=headers,
            json=body,
            timeout=60,
        )
        if put.status_code not in {200, 201}:
            logger.warning(
                "champion remote PUT HTTP %s: %s", put.status_code, put.text[:200]
            )
            return False
        _last_backup_ts = now
        logger.info(
            "champion snapshot backed up branch=%s champ=%s trades=%d ticks=%d",
            branch,
            bool(st.get("champion")),
            len(payload.get("paper_trades") or []),
            len(payload.get("tick_history") or []),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("champion remote backup failed: %s", exc)
        return False


def restore_from_remote() -> bool:
    """If local Champion state empty, pull snapshot from GitHub."""
    if not _local_is_empty():
        return False
    headers = _headers()
    if not headers:
        logger.info("No HISTORY_GITHUB_TOKEN — skip champion restore")
        return False
    repo = config.HISTORY_GITHUB_REPO
    remote_path = config.CHAMPION_GITHUB_PATH
    branch = config.HISTORY_GITHUB_BRANCH or "history-data"
    try:
        from modules.decision_history import _github_contents_url

        def _load(br: str) -> dict[str, Any] | None:
            resp = requests.get(
                _github_contents_url(repo, remote_path, br),
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                logger.warning("champion remote GET %s HTTP %s", br, resp.status_code)
                return None
            raw = base64.b64decode(resp.json().get("content") or "").decode("utf-8")
            return json.loads(raw)

        data = _load(branch)
        source = branch
        if data is None:
            data = _load("main")
            source = "main"
        if not data:
            logger.info("No remote champion snapshot yet (branch=%s)", branch)
            return False
        ok = _apply_snapshot(data)
        if ok:
            logger.info("champion snapshot restored from GitHub branch=%s", source)
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("champion remote restore failed: %s", exc)
        return False


def ensure_restored() -> bool:
    """Call once at startup — never overwrite non-empty local state."""
    if not _local_is_empty():
        st = champion_store.get_paper_state()
        logger.info(
            "champion local keep-as-is champ=%s trades=%s",
            bool(st.get("champion")),
            st.get("trade_count"),
        )
        return False
    return restore_from_remote()
