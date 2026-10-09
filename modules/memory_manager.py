"""Memory manager: trade stats, regime lessons, weekly compaction."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

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
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or config.PATHS["MEMORY"])
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
        for regime, lessons in data["lessons_by_regime"].items():
            ranked = sorted(lessons, key=lambda x: int(x.get("weight", 1)), reverse=True)
            for item in ranked[:top_n_per_regime]:
                rule = item.get("actionable_rule") or item.get("mistake_or_insight") or ""
                weight = item.get("weight", 1)
                if rule:
                    lines.append(f"[{regime} w={weight}] {rule}")
        return "\n".join(lines) if lines else "Chưa có bài học thực chiến."

    def record_closed_trade(self, result: str) -> dict[str, Any]:
        """result: WIN | LOSS. Updates consecutive losses and min score."""
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

    def add_lesson(
        self,
        regime: str,
        mistake_or_insight: str,
        actionable_rule: str,
        weight: int = 1,
        classification: str = "",
    ) -> None:
        data = self._read()
        key = _memory_regime_bucket(regime)
        try:
            w = int(weight) if weight else 1
        except (TypeError, ValueError):
            w = 1
        w = max(1, min(3, w))
        item: dict[str, Any] = {
            "regime": key,
            "mistake_or_insight": mistake_or_insight,
            "actionable_rule": actionable_rule,
            "weight": w,
        }
        if classification:
            item["classification"] = str(classification)
        data["lessons_by_regime"][key].append(item)
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
