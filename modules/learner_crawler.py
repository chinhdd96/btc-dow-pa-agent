"""YouTube transcript crawler → distill → daily_playbook.md via DeepSeek."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from youtube_transcript_api._errors import (  # type: ignore
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )
except ImportError:  # pragma: no cover
    NoTranscriptFound = TranscriptsDisabled = VideoUnavailable = Exception  # type: ignore

import config

logger = logging.getLogger(__name__)

YT_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

CHUNK_SIZE = 28000
TARGET_CHARS = 3500

DISTILL_DAILY_PROMPT = """Bạn là Senior Trading Coach (Dow Theory + Price Action) cho BTCUSDT Futures.

NHIỆM VỤ: Đọc transcript chuyên gia bên dưới và ĐÚC KẾT thành Daily Playbook hành động — KHÔNG tóm tắt lan man, KHÔNG copy nguyên văn.

YÊU CẦU ĐẦU RA (Markdown tiếng Việt):
1. **Bias / Regime hôm nay** (Uptrend / Downtrend / Sideway / High-vol) + 1 câu lý do
2. **Key levels** nêu được từ nội dung (nếu không có số cụ thể thì ghi vùng khái niệm)
3. **Setup hợp lệ** dạng IF ... THEN BUY/SELL (chỉ theo Dow+PA)
4. **Tín hiệu PA** cần chờ (pinbar, engulfing, fakeout, retest...)
5. **Cấm / tránh** hôm nay
6. **Heuristic** ngắn rút từ chuyên gia

RÀNG BUỘC:
- Bỏ quảng cáo, shill coin, kêu gọi FOMO, dự đoán cảm tính không có cấu trúc.
- Ưu tiên rule: "CHỈ ... KHI ..."; "CẤM ... NẾU ...".
- Độ dài mục tiêu: tối đa ~{target_chars} ký tự.
- Không bọc ``` fence quanh toàn bộ câu trả lời.

TRANSCRIPTS:
---
{transcripts}
---
"""

MERGE_DAILY_PROMPT = """Gộp các bản đúc kết Daily Playbook tạm dưới đây thành MỘT playbook thống nhất cho BTCUSDT Futures (Dow + PA).

Giữ cấu trúc: Bias → Key levels → Setup IF/THEN → PA signals → Cấm → Heuristic.
Gộp trùng, giữ rule chặt hơn khi mâu thuẫn.
Tối đa ~{target_chars} ký tự. Markdown tiếng Việt. Không dùng ``` fence.

CÁC BẢN TẠM:
{chunks}
"""


def _load_sources() -> dict[str, Any]:
    path = Path(config.PATHS["SOURCES"])
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_channel_videos(channel_id: str, limit: int = 3) -> list[dict[str, str]]:
    url = YT_RSS.format(channel_id=channel_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    videos: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        video_id_el = entry.find("yt:videoId", ATOM_NS)
        title_el = entry.find("atom:title", ATOM_NS)
        if video_id_el is None:
            continue
        videos.append(
            {
                "video_id": video_id_el.text or "",
                "title": (title_el.text if title_el is not None else "") or "",
            }
        )
        if len(videos) >= limit:
            break
    return videos


def _fetch_transcript(video_id: str) -> str:
    try:
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=["en", "vi", "en-US"])
            snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
            texts = [
                s["text"] if isinstance(s, dict) else getattr(s, "text", str(s))
                for s in snippets
            ]
            return " ".join(texts)
        except (AttributeError, TypeError):
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "vi", "en-US"]
            )
            return " ".join(item["text"] for item in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
        logger.warning("No transcript for %s: %s", video_id, exc)
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcript error for %s: %s", video_id, exc)
        return ""


def _client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY missing (use OpenRouter from VN)")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def _chat(client: OpenAI, user: str, temperature: float = 0.25) -> str:
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You distill YouTube trading content into concise actionable "
                    "daily playbooks. Vietnamese markdown only. No code fences."
                ),
            },
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        for prefix in ("markdown", "md"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
    return text.strip()


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _distill_transcripts(transcripts_blob: str) -> str:
    client = _client()
    chunks = _chunk_text(transcripts_blob)
    logger.info("Learner distill: %d chars → %d chunks", len(transcripts_blob), len(chunks))

    partials: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        prompt = DISTILL_DAILY_PROMPT.format(
            target_chars=TARGET_CHARS,
            transcripts=chunk,
        )
        logger.info("Learner distill chunk %d/%d", i, len(chunks))
        partials.append(_chat(client, prompt))

    if len(partials) == 1:
        return partials[0]

    merge = MERGE_DAILY_PROMPT.format(
        target_chars=TARGET_CHARS,
        chunks="\n\n---\n\n".join(
            f"### Bản tạm {i}\n{p}" for i, p in enumerate(partials, 1)
        ),
    )
    return _chat(client, merge)


def run_learner_crawler() -> str:
    """Crawl YouTube transcripts, distill, write daily_playbook.md."""
    sources = _load_sources()
    channels = sources.get("youtube_channels", [])
    max_videos = int(sources.get("max_videos_per_channel", 3))
    pieces: list[str] = []

    for ch in channels:
        channel_id = ch.get("channel_id", "")
        name = ch.get("name", channel_id)
        if not channel_id:
            continue
        try:
            videos = _fetch_channel_videos(channel_id, limit=max_videos)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed RSS for channel %s: %s", name, exc)
            continue
        for video in videos:
            text = _fetch_transcript(video["video_id"])
            if not text:
                continue
            # Keep more per video; distill step will chunk/merge
            pieces.append(
                f"### {name} — {video['title']} ({video['video_id']})\n{text[:20000]}"
            )

    out_path = Path(config.PATHS["DAILY_PLAYBOOK"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not pieces:
        logger.warning("No transcripts collected; keeping existing daily_playbook")
        return "learner: no transcripts; file unchanged"

    try:
        distilled = _distill_transcripts("\n\n".join(pieces))
    except Exception as exc:  # noqa: BLE001
        logger.exception("DeepSeek distill failed: %s", exc)
        return f"learner: distill failed: {exc}"

    content = (
        f"# Daily Playbook (Tầng 2A — đúc kết từ chuyên gia)\n\n"
        f"_Updated: {now}_\n\n"
        f"{distilled}\n"
    )
    out_path.write_text(content, encoding="utf-8")
    logger.info("Wrote daily_playbook.md (%d chars)", len(content))
    return f"learner: distilled daily_playbook from {len(pieces)} videos"
