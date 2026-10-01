"""RSS news crawler → distill risk context → generated/context.json via DeepSeek."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

CHUNK_SIZE = 16000

DISTILL_NEWS_PROMPT = """Bạn là AI Risk Officer cho bot BTCUSDT Futures (Dow + Price Action).

NHIỆM VỤ: Đọc tin RSS bên dưới và ĐÚC KẾT rủi ro giao dịch — không kể lại tin dài dòng.

Trả về ĐÚNG MỘT JSON (không markdown fence) theo schema:

{{
  "updated_at": "{now_iso}",
  "has_high_impact_news_next_24h": true/false,
  "news_risk_level": "LOW" | "MEDIUM" | "HIGH",
  "summary": "2-4 câu tiếng Việt: sự kiện then chốt ảnh hưởng BTC/crypto",
  "key_events": ["sự kiện 1", "sự kiện 2"],
  "risk_instruction": "Hướng dẫn hành động ngắn cho bot: giữ score / nâng min_score 8.5 / HOLD trước tin X phút / giãn SL...",
  "actionable_rules": [
    "CẤM vào lệnh nếu ...",
    "CHỈ giữ bias nếu ..."
  ]
}}

QUY TẮC CHẤM RISK:
- HIGH: CPI / FOMC / NFP / Fed speak / ETF decision / hack sàn / liquidation cascade lớn trong ~24h.
- MEDIUM: tin macro/crypto ảnh hưởng vừa, chưa rõ giờ cụ thể.
- LOW: tin thường, shill, không đổi cấu trúc thị trường.

TIN:
---
{news_blob}
---
"""

MERGE_NEWS_PROMPT = """Gộp các bản JSON risk tạm dưới đây thành ĐÚNG MỘT JSON cuối cùng cho bot BTCUSDT Futures.
Lấy news_risk_level cao nhất nếu có xung đột (HIGH > MEDIUM > LOW).
Gộp key_events / actionable_rules (bỏ trùng, tối đa 5 mỗi list).
summary ngắn 2-4 câu. updated_at = "{now_iso}".
Chỉ trả JSON, không markdown.

CÁC BẢN TẠM:
{chunks}
"""


def _load_sources() -> dict[str, Any]:
    path = Path(config.PATHS["SOURCES"])
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_rss_items(max_items: int = 20) -> list[str]:
    sources = _load_sources()
    feeds = sources.get("rss_feeds", [])
    lines: list[str] = []
    for feed in feeds:
        url = feed.get("url")
        name = feed.get("name", url)
        if not url:
            continue
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RSS parse failed %s: %s", name, exc)
            continue
        per_feed = max(1, max_items // max(len(feeds), 1))
        for entry in parsed.entries[:per_feed]:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            published = getattr(entry, "published", "") or ""
            # Strip crude HTML
            summary = re.sub(r"<[^>]+>", " ", summary)
            lines.append(f"[{name}] {published} | {title} — {summary[:400]}")
            if len(lines) >= max_items:
                return lines
    return lines


def _client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY missing (use OpenRouter from VN)")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("News distill root is not an object")
    return data


def _normalize(data: dict[str, Any], now_iso: str) -> dict[str, Any]:
    level = str(data.get("news_risk_level", "MEDIUM")).upper()
    if level not in {"LOW", "MEDIUM", "HIGH"}:
        level = "MEDIUM"
    rules = data.get("actionable_rules") or []
    if not isinstance(rules, list):
        rules = [str(rules)]
    events = data.get("key_events") or []
    if not isinstance(events, list):
        events = [str(events)]
    return {
        "updated_at": now_iso,
        "has_high_impact_news_next_24h": bool(
            data.get("has_high_impact_news_next_24h", level == "HIGH")
        ),
        "news_risk_level": level,
        "summary": str(data.get("summary", "")),
        "key_events": [str(x) for x in events][:5],
        "risk_instruction": str(data.get("risk_instruction", "")),
        "actionable_rules": [str(x) for x in rules][:5],
    }


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _distill_news(news_blob: str, now_iso: str) -> dict[str, Any]:
    client = _client()
    chunks = _chunk_text(news_blob)
    logger.info("News distill: %d chars → %d chunks", len(news_blob), len(chunks))

    partials: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, 1):
        prompt = DISTILL_NEWS_PROMPT.format(now_iso=now_iso, news_blob=chunk)
        logger.info("News distill chunk %d/%d", i, len(chunks))
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Distill market news into risk JSON only. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        partials.append(_parse_json(resp.choices[0].message.content or ""))

    if len(partials) == 1:
        return _normalize(partials[0], now_iso)

    merge_prompt = MERGE_NEWS_PROMPT.format(
        now_iso=now_iso,
        chunks=json.dumps(partials, ensure_ascii=False, indent=2),
    )
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown."},
            {"role": "user", "content": merge_prompt},
        ],
        temperature=0.1,
    )
    return _normalize(_parse_json(resp.choices[0].message.content or ""), now_iso)


def run_news_crawler() -> str:
    """Fetch RSS, distill risk context, write context.json."""
    sources = _load_sources()
    max_items = int(sources.get("max_rss_items", 20))
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = _collect_rss_items(max_items=max_items)
    out_path = Path(config.PATHS["CONTEXT_NEWS"])

    if not items:
        logger.warning("No RSS items; keeping existing context.json")
        return "news: no RSS items; file unchanged"

    try:
        data = _distill_news("\n".join(items), now_iso)
    except Exception as exc:  # noqa: BLE001
        logger.exception("News DeepSeek distill failed: %s", exc)
        return f"news: distill failed: {exc}"

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote context.json risk=%s", data.get("news_risk_level"))
    return f"news: distilled context risk={data.get('news_risk_level')}"
