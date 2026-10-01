"""
Extract PDFs from knowledge/dow + knowledge/price_action,
distill via DeepSeek into actionable Core Playbook markdown.

Usage:
  python scripts/build_core_playbook.py

Requires: pypdf, openai, LLM_API_KEY in .env (OpenRouter recommended from VN)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

try:
    from pypdf import PdfReader
except ImportError:
    print("Missing pypdf. Run: pip install pypdf")
    raise SystemExit(1)

from openai import OpenAI

import config

DOW_DIR = ROOT / "knowledge" / "dow"
PA_DIR = ROOT / "knowledge" / "price_action"
OUT = ROOT / "generated" / "core_playbook.md"

# Raw text fed to distill API (chunked if longer)
CHUNK_SIZE = 35000
MAX_TOTAL_RAW = 120000
# Final playbook kept short for trade-time context
TARGET_DOW_CHARS = 4500
TARGET_PA_CHARS = 4500

DISTILL_SECTION_PROMPT = """Bạn là Senior Trading Coach chuyên Dow Theory và Price Action cho BTCUSDT Futures.

NHIỆM VỤ: Đọc tài liệu nguồn bên dưới và ĐÚC KẾT thành playbook hành động ngắn, sắc — KHÔNG tóm tắt kiểu sách giáo khoa dài.

PHẠM VI SECTION: {section_name}
{section_hint}

YÊU CẦU ĐẦU RA (Markdown tiếng Việt):
1. **Nguyên tắc cốt lõi** (bullet ngắn)
2. **Setup hợp lệ** (điều kiện vào lệnh rõ: IF ... THEN BUY/SELL)
3. **Tín hiệu / pattern** cần nhận diện (tên + ý nghĩa 1 dòng)
4. **Cấm tuyệt đối** (anti-patterns)
5. **SL / TP / quản trị** nếu tài liệu đề cập
6. **Kinh nghiệm / heuristic** hữu ích rút từ tài liệu

RÀNG BUỘC:
- Chỉ giữ kiến thức áp dụng được khi nhìn nến 4H (xu hướng) và 1H (entry).
- Bỏ lịch sử, câu chuyện, quảng cáo, ví dụ dài dòng không thành rule.
- Ưu tiên rule dạng: "CHỈ ... KHI ..."; "CẤM ... NẾU ...".
- Độ dài mục tiêu: tối đa ~{target_chars} ký tự.
- Không bọc markdown fence ``` quanh toàn bộ câu trả lời.

TÀI LIỆU NGUỒN:
---
{source_text}
---
"""

MERGE_PROMPT = """Bạn là Senior Trading Coach. Gộp các bản đúc kết tạm dưới đây thành MỘT playbook thống nhất cho section "{section_name}".

YÊU CẦU:
- Gộp rule trùng, bỏ mâu thuẫn (giữ rule chặt / an toàn hơn).
- Giữ cấu trúc: Nguyên tắc → Setup → Pattern → Cấm → SL/TP → Heuristic.
- Tối đa ~{target_chars} ký tự. Markdown tiếng Việt. Không dùng ``` fence.

CÁC BẢN TẠM:
{chunks}
"""

SECTION_HINTS = {
    "DOW": (
        "Tập trung Lý thuyết Dow: xu hướng primary, HH/HL, LH/LL, confirmation, "
        "phá cấu trúc, sideway vs trend. Đây là NỀN XÁC ĐỊNH HƯỚNG — không chi tiết entry nến nhỏ."
    ),
    "PRICE_ACTION": (
        "Tập trung Price Action vào lệnh: pinbar, engulfing, fakeout/stop-hunt, "
        "key level, retest, rejection. Đây là TIMING ENTRY tại vùng Dow then chốt."
    ),
}


def extract_folder_text(folder: Path) -> tuple[str, list[str]]:
    """Return (combined_text, list of pdf names)."""
    pdfs = sorted(folder.glob("*.pdf"))
    names: list[str] = []
    parts: list[str] = []
    total = 0
    for pdf in pdfs:
        names.append(pdf.name)
        try:
            reader = PdfReader(str(pdf))
            pages = [(page.extract_text() or "") for page in reader.pages]
            body = "\n".join(pages).strip()
        except Exception as exc:  # noqa: BLE001
            body = f"(Lỗi đọc {pdf.name}: {exc})"
        piece = f"\n\n===== FILE: {pdf.name} =====\n{body}"
        if total + len(piece) > MAX_TOTAL_RAW:
            remain = MAX_TOTAL_RAW - total
            if remain > 500:
                parts.append(piece[:remain] + "\n...(cắt do MAX_TOTAL_RAW)...")
            break
        parts.append(piece)
        total += len(piece)
    return ("\n".join(parts).strip(), names)


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def get_client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError(
            "Thiếu LLM_API_KEY trong .env — từ VN dùng OpenRouter: "
            "LLM_BASE_URL=https://openrouter.ai/api/v1"
        )
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def call_llm(client: OpenAI, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You distill trading books into concise actionable playbooks. "
                    "Reply in Vietnamese markdown only. No surrounding code fences."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("markdown"):
            text = text[8:].strip()
        elif text.startswith("md"):
            text = text[2:].strip()
    return text.strip()


def distill_section(
    client: OpenAI,
    section_key: str,
    section_title: str,
    raw_text: str,
    target_chars: int,
) -> str:
    if not raw_text or raw_text.startswith("(Lỗi") and "=====" not in raw_text:
        return (
            f"_(Chưa có PDF hợp lệ cho {section_title}. "
            f"Thả file vào knowledge/ tương ứng rồi chạy lại script.)_"
        )

    hint = SECTION_HINTS[section_key]
    chunks = chunk_text(raw_text)
    print(f"  [{section_key}] raw={len(raw_text)} chars, chunks={len(chunks)}")

    partials: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{section_key}] distilling chunk {i}/{len(chunks)}...")
        prompt = DISTILL_SECTION_PROMPT.format(
            section_name=section_title,
            section_hint=hint,
            target_chars=target_chars,
            source_text=chunk,
        )
        partials.append(call_llm(client, prompt))

    if len(partials) == 1:
        return partials[0]

    print(f"  [{section_key}] merging {len(partials)} partials...")
    merge_prompt = MERGE_PROMPT.format(
        section_name=section_title,
        target_chars=target_chars,
        chunks="\n\n---\n\n".join(
            f"### Bản tạm {i}\n{p}" for i, p in enumerate(partials, 1)
        ),
    )
    return call_llm(client, merge_prompt)


def main() -> None:
    client = get_client()

    dow_raw, dow_files = extract_folder_text(DOW_DIR)
    pa_raw, pa_files = extract_folder_text(PA_DIR)

    if not dow_files and not pa_files:
        print("Không tìm thấy PDF nào trong knowledge/dow hoặc knowledge/price_action.")
        raise SystemExit(1)

    print("Distilling DOW section...")
    dow_md = distill_section(
        client,
        "DOW",
        "Lý thuyết Dow (xác định hướng)",
        dow_raw,
        TARGET_DOW_CHARS,
    )

    print("Distilling PRICE ACTION section...")
    pa_md = distill_section(
        client,
        "PRICE_ACTION",
        "Price Action (vào lệnh)",
        pa_raw,
        TARGET_PA_CHARS,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Core Playbook: Dow Theory & Price Action (BTCUSDT Futures)

> Auto-distilled by `scripts/build_core_playbook.py` lúc {now}.
> Nguồn Dow: {', '.join(dow_files) if dow_files else '(trống)'}
> Nguồn PA: {', '.join(pa_files) if pa_files else '(trống)'}
>
> Dow = xác định xu hướng gốc | Price Action = tín hiệu vào lệnh tại key level.

## 1. LÝ THUYẾT DOW (Gốc — xác định hướng)

{dow_md}

## 2. PRICE ACTION (Vào lệnh trade)

{pa_md}

## 3. Quy tắc vận hành bot (cố định — hard rules)

- Khung **4H**: xác định Uptrend (HH+HL) / Downtrend (LH+LL) / Sideway — quyết định hướng.
- Khung **1H**: chỉ vào lệnh **theo hướng 4H** khi có PA tại key level (Pinbar, Engulfing, Fakeout reject).
- Không đánh ngược xu hướng 4H chỉ vì 1 nến 1H đẹp.
- Bắt buộc có SL; ưu tiên R:R >= 1.5; size theo risk % tài khoản (code hard guard).
- News risk HIGH hoặc setup dưới min_score → HOLD.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(f"\nWrote {OUT} ({len(content)} chars)")
    print(f"  Dow PDFs: {dow_files}")
    print(f"  PA  PDFs: {pa_files}")


if __name__ == "__main__":
    main()
