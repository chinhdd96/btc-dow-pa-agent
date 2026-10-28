"""Memory manager: trade stats, regime lessons, weekly compaction."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

VALID_REGIMES = {
    "REGIME_UPTREND",
    "REGIME_DOWNTREND",
    "REGIME_SIDEWAY_CHOP",
    "REGIME_HIGH_VOLATILITY_NEWS",
}

DEFAULT_MEMORY: dict[str, Any] = {
    "system_stats": {
        "total_trades": 0,
        "win_rate": 0.0,
        "consecutive_losses": 0,
        "current_min_score": config.BASE_MIN_SCORE,
    },
    "lessons_by_regime": {r: [] for r in VALID_REGIMES},
}


def _memory_regime_bucket(regime: str) -> str:
    """Map rich REGIME_* labels onto the 4 memory buckets."""
    r = (regime or "").upper()
    if not r.startswith("REGIME_"):
        r = f"REGIME_{r}" if r else "REGIME_SIDEWAY_CHOP"
    if "HIGH_VOL" in r or "NEWS" in r:
        return "REGIME_HIGH_VOLATILITY_NEWS"
    if "DOWNTREND" in r or "TREND_DOWN" in r:
        return "REGIME_DOWNTREND"
    if "UPTREND" in r or "TREND_UP" in r:
        return "REGIME_UPTREND"
    if r in VALID_REGIMES:
        return r
    # RANGE / SIDEWAY / breakout-state labels → chop bucket
    return "REGIME_SIDEWAY_CHOP"


class MemoryManager:
    def __init__(
        self,
        path: str | None = None,
        *,
        enable_remote_backup: bool = True,
    ) -> None:
        self.path = Path(path or config.PATHS["MEMORY"])
        self.enable_remote_backup = enable_remote_backup
        if not self.path.exists():
            self._write(DEFAULT_MEMORY)

    def _read(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.exception("memory read failed, resetting: %s", exc)
            data = DEFAULT_MEMORY.copy()
            data["lessons_by_regime"] = {r: [] for r in VALID_REGIMES}
            self._write(data)
            return data

        stats = data.setdefault("system_stats", {})
        stats.setdefault("total_trades", 0)
        stats.setdefault("win_rate", 0.0)
        stats.setdefault("consecutive_losses", 0)
        stats.setdefault("current_min_score", config.BASE_MIN_SCORE)
        lessons = data.setdefault("lessons_by_regime", {})
        for regime in VALID_REGIMES:
            lessons.setdefault(regime, [])
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self.path.parent),
            suffix=".tmp",
        ) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)
        if self.enable_remote_backup:
            try:
                backup_remote(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory backup skipped: %s", exc)

    def load(self) -> dict[str, Any]:
        return self._read()

    def get_min_score(self) -> float:
        stored = float(self._read()["system_stats"]["current_min_score"])
        # Migrate legacy thresholds (6.0 / 8.5) to new defaults on read
        if stored in {6.0, 8.5}:
            return config.BASE_MIN_SCORE
        return min(stored, config.PENALTY_MIN_SCORE)

    def get_lessons_for_regime(self, regime: str, top_n: int = 5) -> list[dict[str, Any]]:
        data = self._read()
        key = _memory_regime_bucket(regime)
        lessons = list(data["lessons_by_regime"].get(key, []))
        lessons.sort(key=lambda x: int(x.get("weight", 1)), reverse=True)
        return lessons[:top_n]

    def get_all_lessons_summary(self, top_n_per_regime: int = 3) -> str:
        data = self._read()
        lines: list[str] = []
        # Prefer PROCESS_ERROR first across regimes
        flat: list[tuple[int, str, dict[str, Any]]] = []
        for regime, lessons in data["lessons_by_regime"].items():
            for item in lessons:
                cls = str(item.get("classification") or "").upper()
                pri = 2 if cls == "PROCESS_ERROR" else 1
                flat.append((pri, regime, item))
        flat.sort(
            key=lambda x: (
                -x[0],
                -int(x[2].get("weight", 1) or 1),
                -int(x[2].get("evidence_n", 1) or 1),
            )
        )
        per_regime: dict[str, int] = {}
        for _pri, regime, item in flat:
            n = per_regime.get(regime, 0)
            if n >= top_n_per_regime:
                continue
            per_regime[regime] = n + 1
            rule = item.get("actionable_rule") or item.get("mistake_or_insight") or ""
            weight = item.get("weight", 1)
            ev = item.get("evidence_n", 1)
            tag = item.get("setup_tag") or ""
            if rule:
                extra = f" tag={tag}" if tag else ""
                lines.append(f"[{regime} w={weight} n={ev}{extra}] {rule}")
        return "\n".join(lines) if lines else "Chưa có bài học thực chiến."

    def record_closed_trade(self, result: str) -> dict[str, Any]:
        """
        Legacy single-trade bump. Prefer resync_stats_from_trades(paper)
        so system_stats stays SSOT with paper.closed_trades.
        """
        data = self._read()
        stats = data["system_stats"]
        total = int(stats.get("total_trades", 0)) + 1
        wins_approx = float(stats.get("win_rate", 0.0)) * max(total - 1, 0)
        result_u = result.upper()
        if result_u == "WIN":
            wins_approx += 1
            stats["consecutive_losses"] = 0
            stats["current_min_score"] = config.BASE_MIN_SCORE
        else:
            stats["consecutive_losses"] = int(stats.get("consecutive_losses", 0)) + 1
            if stats["consecutive_losses"] >= config.MAX_CONSECUTIVE_LOSSES:
                stats["current_min_score"] = config.PENALTY_MIN_SCORE

        stats["total_trades"] = total
        stats["win_rate"] = round(wins_approx / total, 4) if total else 0.0
        self._write(data)
        return stats

    def resync_stats_from_trades(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Rebuild system_stats from paper closed_trades (single source of truth)."""
        data = self._read()
        stats = data["system_stats"]
        n = len(trades)
        wins = sum(1 for t in trades if str(t.get("result") or "").upper() == "WIN")
        consec = 0
        for t in reversed(trades):
            if str(t.get("result") or "").upper() == "WIN":
                break
            consec += 1
        stats["total_trades"] = n
        stats["win_rate"] = round(wins / n, 4) if n else 0.0
        stats["consecutive_losses"] = consec
        if consec >= config.MAX_CONSECUTIVE_LOSSES:
            stats["current_min_score"] = config.PENALTY_MIN_SCORE
        else:
            stats["current_min_score"] = config.BASE_MIN_SCORE
        self._write(data)
        return stats

    def add_lesson(
        self,
        regime: str,
        mistake_or_insight: str,
        actionable_rule: str,
        weight: int = 1,
        classification: str = "",
        setup_tag: str = "",
        close_reason: str = "",
        pnl: float | None = None,
        r_multiple: float | None = None,
    ) -> None:
        data = self._read()
        key = _memory_regime_bucket(regime)
        try:
            w = int(weight) if weight else 1
        except (TypeError, ValueError):
            w = 1
        w = max(1, min(3, w))
        rule_key = (actionable_rule or "").strip().lower()
        lessons = data["lessons_by_regime"][key]
        if rule_key:
            for existing in lessons:
                if (existing.get("actionable_rule") or "").strip().lower() == rule_key:
                    existing["evidence_n"] = int(existing.get("evidence_n", 1) or 1) + 1
                    existing["weight"] = max(
                        int(existing.get("weight", 1) or 1),
                        w,
                        2 if existing["evidence_n"] >= 3 else 1,
                    )
                    if classification:
                        existing["classification"] = str(classification)
                    if setup_tag:
                        existing["setup_tag"] = str(setup_tag)
                    if close_reason:
                        existing["close_reason"] = str(close_reason)
                    if pnl is not None:
                        existing["last_pnl"] = float(pnl)
                    if r_multiple is not None:
                        existing["last_r"] = float(r_multiple)
                    if mistake_or_insight:
                        existing["mistake_or_insight"] = mistake_or_insight
                    self._write(data)
                    return
        item: dict[str, Any] = {
            "regime": key,
            "mistake_or_insight": mistake_or_insight,
            "actionable_rule": actionable_rule,
            "weight": w,
            "evidence_n": 1,
        }
        if classification:
            item["classification"] = str(classification)
        if setup_tag:
            item["setup_tag"] = str(setup_tag)
        if close_reason:
            item["close_reason"] = str(close_reason)
        if pnl is not None:
            item["last_pnl"] = float(pnl)
        if r_multiple is not None:
            item["last_r"] = float(r_multiple)
        lessons.append(item)
        self._write(data)

    def compact_memory(self) -> str:
        """Merge similar lessons via LLM when possible; hard-cap top 5 by weight per regime."""
        data = self._read()
        for regime, lessons in data["lessons_by_regime"].items():
            if not lessons:
                continue
            merged = self._merge_lessons(regime, lessons)
            merged.sort(key=lambda x: int(x.get("weight", 1)), reverse=True)
            data["lessons_by_regime"][regime] = merged[: config.MAX_LESSONS_PER_REGIME]
        self._write(data)
        return "memory compact: done"

    def _merge_lessons(
        self,
        regime: str,
        lessons: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(lessons) <= config.MAX_LESSONS_PER_REGIME:
            # Still try light dedupe by identical actionable_rule
            return self._dedupe_exact(lessons)

        if not config.LLM_API_KEY:
            return self._dedupe_exact(lessons)

        try:
            client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
            )
            payload = json.dumps(lessons, ensure_ascii=False)
            prompt = (
                f"Gộp các bài học trùng ý cho regime {regime}. "
                "Trả về JSON array các object "
                '{{"regime","mistake_or_insight","actionable_rule","weight"}} '
                "với weight đã cộng dồn. Không markdown.\n\n"
                f"LESSONS:\n{payload}"
            )
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Return only a JSON array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            merged = json.loads(raw)
            if isinstance(merged, list) and merged:
                cleaned = []
                for item in merged:
                    if not isinstance(item, dict):
                        continue
                    cleaned.append(
                        {
                            "regime": regime,
                            "mistake_or_insight": item.get("mistake_or_insight", ""),
                            "actionable_rule": item.get("actionable_rule", ""),
                            "weight": int(item.get("weight", 1) or 1),
                        }
                    )
                return cleaned or self._dedupe_exact(lessons)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM compact failed for %s: %s", regime, exc)
        return self._dedupe_exact(lessons)

    @staticmethod
    def _dedupe_exact(lessons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bucket: dict[str, dict[str, Any]] = {}
        for item in lessons:
            key = (item.get("actionable_rule") or "").strip().lower()
            if not key:
                key = (item.get("mistake_or_insight") or "").strip().lower()
            if not key:
                continue
            if key in bucket:
                bucket[key]["weight"] = int(bucket[key].get("weight", 1)) + int(
                    item.get("weight", 1) or 1
                )
            else:
                bucket[key] = dict(item)
                bucket[key]["weight"] = int(item.get("weight", 1) or 1)
        return list(bucket.values())


def _github_headers() -> dict[str, str] | None:
    tok = config.HISTORY_GITHUB_TOKEN
    if not tok:
        return None
    return {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "btc-dow-pa-agent-memory",
    }


def backup_remote(data: dict[str, Any] | None = None) -> bool:
    headers = _github_headers()
    if not headers:
        return False
    path_local = Path(config.PATHS["MEMORY"])
    if data is None:
        if not path_local.exists():
            return False
        try:
            data = json.loads(path_local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
    try:
        from modules.decision_history import (
            _ensure_history_branch,
            _github_contents_url,
        )

        repo = config.HISTORY_GITHUB_REPO
        remote = config.MEMORY_GITHUB_PATH
        branch = config.HISTORY_GITHUB_BRANCH or "history-data"
        if not _ensure_history_branch(headers, repo, branch):
            return False
        content_b64 = base64.b64encode(
            json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("ascii")
        meta = requests.get(
            _github_contents_url(repo, remote, branch), headers=headers, timeout=30
        )
        sha = meta.json().get("sha") if meta.status_code == 200 else None
        body: dict[str, Any] = {
            "message": "chore: sync memory.json for Blitz persistence",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        put = requests.put(
            _github_contents_url(repo, remote),
            headers=headers,
            json=body,
            timeout=60,
        )
        if put.status_code not in {200, 201}:
            logger.warning("memory PUT HTTP %s", put.status_code)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory backup failed: %s", exc)
        return False


def restore_from_remote() -> bool:
    path = Path(config.PATHS["MEMORY"])
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
            lessons = cur.get("lessons_by_regime") or {}
            if any(lessons.get(r) for r in lessons):
                return False
            if int((cur.get("system_stats") or {}).get("total_trades") or 0) > 0:
                return False
        except (OSError, json.JSONDecodeError):
            pass
    headers = _github_headers()
    if not headers:
        return False
    try:
        from modules.decision_history import _github_contents_url

        repo = config.HISTORY_GITHUB_REPO
        remote = config.MEMORY_GITHUB_PATH
        branch = config.HISTORY_GITHUB_BRANCH or "history-data"
        resp = requests.get(
            _github_contents_url(repo, remote, branch), headers=headers, timeout=30
        )
        if resp.status_code != 200:
            return False
        raw = base64.b64decode(resp.json().get("content") or "").decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("memory restored from remote")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory restore failed: %s", exc)
        return False


def ensure_restored() -> bool:
    return restore_from_remote()
