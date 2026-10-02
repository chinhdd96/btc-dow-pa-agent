"""DeepSeek LLM agent: trade decision + post-mortem."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

LLM_DECISION_PROMPT = """Bạn là Senior Crypto Quant Trader chuyên giao dịch BTCUSDT.

MỤC TIÊU:
Phân tích thị trường thực chiến và chỉ giao dịch khi có lợi thế rõ ràng.
LÝ THUYẾT DOW và PRICE ACTION là nền tảng chính để đọc cấu trúc và tìm điểm vào,
nhưng KHÔNG được áp dụng máy móc. Dữ liệu thị trường hiện tại luôn được ưu tiên.

DỮ LIỆU:

1. DOW & PRICE ACTION PLAYBOOK
---
{core_playbook_content}
---

2. MEMORY / KINH NGHIỆM
---
{regime_memory_lessons}
---

2b. LỊCH SỬ QUYẾT ĐỊNH GẦN ĐÂY (tóm tắt, mới nhất ở dưới)
---
{decision_history}
---
Đây chỉ là ngữ cảnh ngắn hạn. Ưu tiên dữ liệu cấu trúc/PA live bên dưới.
Nếu thesis hoặc điều kiện chờ trước đó đã invalid → nêu rõ và cập nhật.

3. CẤU TRÚC THỊ TRƯỜNG

1W:
{structure_1w}

1D:
{structure_1d}

4H:
{structure_4h}

1H:
{structure_1h}

alignment: {structure_align}

ZOOM PA:
{zoom_block}

GIÁ HIỆN TẠI: {current_price}
VỊ THẾ HIỆN TẠI: {current_position}

QUY TẮC PHÂN TÍCH:

1. Dùng Dow để xác định cấu trúc chính:
   HH, HL, LH, LL, xu hướng, vùng giá quan trọng và sự thay đổi cấu trúc.

2. Dùng Price Action để tìm thời điểm vào:
   rejection, breakout, failed breakout, retest, engulfing, pinbar,
   compression, expansion và các hành vi giá đáng chú ý khác.

3. Không bắt buộc 1D/4H/1H phải cùng một hướng.
   Hãy xác định timeframe nào đang dẫn dắt và timeframe nào đang xác nhận
   hoặc mâu thuẫn.

4. TÁCH CẤU TRÚC VÀ TRẠNG THÁI THỊ TRƯỜNG:

Cấu trúc chính chỉ gồm: UPTREND / DOWNTREND / RANGE.

Nhưng RANGE không đồng nghĩa HOLD. Hãy xác định thêm trạng thái:
BREAKOUT_ATTEMPT / BREAKOUT_CONFIRMED / FAILED_BREAKOUT / RETEST / REJECTION / CONSOLIDATION.

Không máy móc:
UPTREND = BUY
DOWNTREND = SELL
RANGE = HOLD.

Khi giá gần biên range hoặc Key Level, hãy xem đó là DECISION POINT và tìm dấu hiệu:
breakout, acceptance, rejection, failed breakout, retest, liquidity sweep,
compression, expansion, momentum và thay đổi cấu trúc.

Không cần pinbar/engulfing mới được coi là Price Action.

Không BUY chỉ vì vừa breakout; ưu tiên breakout + acceptance/retest.
Không SELL chỉ vì bị từ chối một lần; cần rejection/failed breakout + xác nhận cấu trúc.

Nếu 1D tăng nhưng 4H range → mô tả "bullish bias + 4H consolidation",
không gọi toàn bộ thị trường là SIDEWAY.

5. Luôn đánh giá:
- Giá đang ở đâu trong cấu trúc?
- Đang ở Key Level hay giữa range?
- Thị trường đang tích lũy, breakout hay rejection?
- Có TRIGGER + INVALIDATION + R:R hợp lý chưa?
- Điều gì ủng hộ và điều gì làm thesis sai?

6. MEMORY và LỊCH SỬ QUYẾT ĐỊNH chỉ là tham khảo.
   Dữ liệu thị trường hiện tại luôn được ưu tiên; không neo cứng vào narrative cũ.

7. Không ép giao dịch. Nếu chưa có edge rõ → HOLD và ghi rõ điều kiện chờ:
BREAKOUT / RETEST / REJECTION / PULLBACK.

Chỉ BUY/SELL khi có:
THESIS + KEY LEVEL + TRIGGER + INVALIDATION + R:R.

setup_score là đánh giá chất lượng setup, không phải điều kiện bắt buộc để BUY/SELL.
Tầng Python chịu trách nhiệm kiểm tra min_score, risk và execution.

LƯU Ý EXECUTION:
Tầng Python có thể chặn lệnh ngược primary_bias 1D hoặc khi primary SIDEWAY.
Vẫn mô tả transition/edge trung thực; nếu bị chặn thì action=HOLD và giải thích trong reasoning.
Ngưỡng điểm tối thiểu hiện tại (Python): {min_score_required} — chỉ để tham chiếu, không tự ép HOLD vì điểm.

ĐIỂM QUAN TRỌNG:
Hãy suy nghĩ như một trader thực chiến:
không cố chứng minh lý thuyết đúng,
mà dùng lý thuyết để hiểu thị trường và tìm cơ hội có xác suất/lợi thế tốt.

CẤM: Không trả lời kiểu "thiếu dữ liệu / không đủ nến" —
luôn suy luận từ block pre-compute + CSV đã cung cấp.

OUTPUT:
Chỉ trả về đúng 1 JSON object, không markdown, không ```.
Bắt đầu bằng {{ và kết thúc bằng }}.
action chỉ nhận: BUY | SELL | HOLD.
market_regime = REGIME_{{CẤU_TRÚC}}_{{TRẠNG_THÁI}} khi có trạng thái
(ví dụ REGIME_RANGE_BREAKOUT_ATTEMPT, REGIME_UPTREND_RETEST),
hoặc REGIME_UPTREND / REGIME_DOWNTREND / REGIME_RANGE nếu chưa rõ trạng thái.
Các trường mô tả viết bằng tiếng Việt.

FORMAT:
{{
  "market_regime": "REGIME_RANGE_BREAKOUT_ATTEMPT",
  "dow_structure_analysis": "Mô tả cấu trúc Dow bằng tiếng Việt",
  "price_action_signal": "Mô tả tín hiệu PA bằng tiếng Việt",
  "market_location": "Giá đang ở đâu trong cấu trúc (tiếng Việt)",
  "setup_score": 7.5,
  "action": "HOLD",
  "entry_price": null,
  "stop_loss_price": null,
  "take_profit_price": null,
  "risk_reward_ratio": null,
  "reasoning": "Lý do BUY/SELL/HOLD bằng tiếng Việt"
}}

Nếu HOLD: entry_price, stop_loss_price, take_profit_price, risk_reward_ratio = null.
"""

POST_MORTEM_PROMPT = """Bạn là AI Risk Officer phụ trách phân tích hậu phẫu (Post-Mortem) lệnh giao dịch vừa đóng.

THÔNG TIN LỆNH VỪA ĐÓNG:
- Cặp tiền: BTCUSDT | Hành động: {action}
- Giá Entry: {entry} | Giá SL: {sl} | Giá TP: {tp}
- Kết quả: {result} (WIN / LOSS) | PnL: {pnl} USDT
- Market Regime lúc vào lệnh: {regime}
- Lý do vào lệnh ban đầu: {entry_reasoning}
- Diễn biến các cây nến sau khi vào lệnh: {post_trade_candles}

NHIỆM VỤ:
1. Xác định nguyên nhân cốt lõi khiến lệnh WIN hoặc LOSS (Do đánh đúng Dow? Do nến PA giả? Do tin tức giật?).
2. Rút ra 01 BÀI HỌC HÀNH ĐỘNG (Actionable Rule) cực kỳ ngắn gọn, sắc bén để ngăn lặp lại lỗi sai hoặc phát huy lệnh thắng.

TRẢ VỀ KẾT QUẢ DƯỚI DẠNG JSON NGUYÊN BẢN.
mistake_or_insight và actionable_rule BẮT BUỘC viết bằng TIẾNG VIỆT:
{{
  "regime": "{regime}",
  "mistake_or_insight": "Nguyên nhân thắng/thua 1-2 câu tiếng Việt",
  "actionable_rule": "CẤM... hoặc CHỈ VÀO LỆNH KHI... (tiếng Việt)",
  "weight": 1
}}
"""

HOLD_DECISION: dict[str, Any] = {
    "market_regime": "REGIME_SIDEWAY_CHOP",
    "dow_structure_analysis": "Không phân tích được (fallback)",
    "price_action_signal": "Không có",
    "market_location": "",
    "setup_score": 0.0,
    "action": "HOLD",
    "entry_price": None,
    "stop_loss_price": None,
    "take_profit_price": None,
    "risk_reward_ratio": None,
    "reasoning": "HOLD do lỗi LLM/parse/validation",
}


def _read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _strip_noise(text: str) -> str:
    raw = (text or "").strip()
    # gpt-oss / reasoning wrappers
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<thinking>[\s\S]*?</thinking>", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<\|.*?\|>", "", raw)
    if "```" in raw:
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        if fenced:
            raw = fenced[-1].strip()
        else:
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _balanced_json_slices(text: str) -> list[str]:
    """Extract candidate {...} slices with brace balancing."""
    slices: list[str] = []
    start = -1
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    slices.append(text[start : i + 1])
                    start = -1
    return slices


def extract_json_object(text: str) -> dict[str, Any]:
    raw = _strip_noise(text)
    if not raw:
        raise ValueError("Empty LLM response")

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    candidates = _balanced_json_slices(raw)
    # Prefer objects that look like a trade decision
    scored: list[tuple[int, dict[str, Any]]] = []
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            # trailing commas / soft fix
            try:
                fixed = re.sub(r",\s*}", "}", cand)
                fixed = re.sub(r",\s*]", "]", fixed)
                data = json.loads(fixed)
            except json.JSONDecodeError:
                continue
        if not isinstance(data, dict):
            continue
        score = 0
        if "action" in data:
            score += 3
        if "setup_score" in data:
            score += 2
        if "market_regime" in data:
            score += 1
        if "reasoning" in data:
            score += 1
        scored.append((score, data))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > 0:
            return scored[0][1]
        return scored[0][1]

    raise ValueError(f"No JSON object found in LLM response: {raw[:240]!r}")


def _compact_candles(
    candles: list[dict[str, Any]],
    last_n: int | None = None,
    with_volume: bool = False,
) -> str:
    """CSV denser — optional volume khi còn budget."""
    n = last_n if last_n is not None else config.CANDLE_PROMPT_BARS
    if n <= 0 or not candles:
        return "(không gửi CSV — dùng structure)"
    slim = candles[-n:]
    if with_volume and any("volume" in c for c in slim):
        lines = ["o,h,l,c,v"]
        for c in slim:
            lines.append(
                f"{round(float(c.get('open', 0)), 1)},"
                f"{round(float(c.get('high', 0)), 1)},"
                f"{round(float(c.get('low', 0)), 1)},"
                f"{round(float(c.get('close', 0)), 1)},"
                f"{round(float(c.get('volume', 0)), 1)}"
            )
    else:
        lines = ["o,h,l,c"]
        for c in slim:
            lines.append(
                f"{round(float(c.get('open', 0)), 1)},"
                f"{round(float(c.get('high', 0)), 1)},"
                f"{round(float(c.get('low', 0)), 1)},"
                f"{round(float(c.get('close', 0)), 1)}"
            )
    return "\n".join(lines)


def _zoom_block(
    candles_4h: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
    *,
    n4: int,
    n1: int,
    near_s: Any = None,
    near_r: Any = None,
    with_volume: bool = False,
) -> str:
    parts = [
        f"Key level gần (từ structure): S={near_s} | R={near_r}",
        f"- 1H zoom ({min(n1, len(candles_1h))} bars) — PA entry:",
        _compact_candles(candles_1h, last_n=n1, with_volume=with_volume),
    ]
    if n4 > 0:
        parts.extend(
            [
                f"- 4H zoom ({min(n4, len(candles_4h))} bars) — confirm cấu trúc gần:",
                _compact_candles(candles_4h, last_n=n4, with_volume=False),
            ]
        )
    return "\n".join(parts)


def _build_decision_prompt(
    *,
    core: str,
    memory_lessons: str,
    decision_history: str,
    candles_1d: list[dict[str, Any]],
    candles_4h: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
    current_price: float,
    current_position: dict[str, Any] | None,
    min_score_required: float,
) -> str:
    """
    Wide view = structure FULL fetch.
    Zoom CSV: mặc định 1H; còn token → +4H + volume + thêm bars 1H.
    """
    from modules.market_structure import build_multi_tf_structure

    core_txt = (core or "").strip()
    if len(core_txt) > config.CORE_PLAYBOOK_MAX_CHARS:
        core_txt = core_txt[: config.CORE_PLAYBOOK_MAX_CHARS]
    if not core_txt:
        core_txt = "(THIẾU FILE core_playbook.md — dùng quy tắc Dow/PA mặc định trong prompt)"

    max_prompt = config.LLM_MAX_PROMPT_CHARS
    bars_1h = config.CANDLE_PROMPT_BARS_1H
    mem_budget = config.MEMORY_LESSONS_MAX_CHARS

    struct_1w, struct_1d, struct_4h, struct_1h, meta = build_multi_tf_structure(
        candles_1d, candles_4h, candles_1h
    )
    near_s = meta.get("nearest_support")
    near_r = meta.get("nearest_resistance")

    align = (
        f"history_source={meta.get('history_source')} | "
        f"primary={meta.get('primary_bias')} | "
        f"1W={meta.get('bias_1w')} 1D={meta.get('bias_1d')} "
        f"4H={meta.get('bias_4h')} 1H={meta.get('bias_1h')} | "
        f"aligned_htf={meta.get('aligned_htf')} | "
        f"key_S={meta.get('key_supports')} key_R={meta.get('key_resistances')} | "
        f"bars_used 1W={meta.get('bars_1w')} 1D={meta.get('bars_1d')} "
        f"4H={meta.get('bars_4h')} 1H={meta.get('bars_1h')}"
    )
    pos = (
        json.dumps(current_position, ensure_ascii=False)
        if current_position
        else "NONE"
    )

    hist_budget = config.DECISION_HISTORY_PROMPT_MAX_CHARS
    hist_txt = (decision_history or "").strip() or "(chưa có lịch sử quyết định gần đây)"
    if len(hist_txt) > hist_budget:
        hist_txt = hist_txt[-hist_budget:]

    def _assemble(
        m_budget: int,
        n1: int,
        n4: int = 0,
        with_vol: bool = False,
        hist: str | None = None,
    ) -> str:
        return LLM_DECISION_PROMPT.format(
            core_playbook_content=core_txt,
            regime_memory_lessons=(memory_lessons or "")[:m_budget] or "(chưa có bài học)",
            decision_history=hist if hist is not None else hist_txt,
            structure_1w=struct_1w,
            structure_1d=struct_1d,
            structure_4h=struct_4h,
            structure_1h=struct_1h,
            structure_align=align,
            zoom_block=_zoom_block(
                candles_4h,
                candles_1h,
                n4=n4,
                n1=n1,
                near_s=near_s,
                near_r=near_r,
                with_volume=with_vol,
            ),
            current_price=current_price,
            current_position=pos,
            min_score_required=min_score_required,
        )

    prompt = _assemble(mem_budget, bars_1h, n4=0, with_vol=False)

    if len(prompt) < max_prompt - 800:
        enriched = _assemble(mem_budget, bars_1h, n4=16, with_vol=False)
        if len(enriched) <= max_prompt:
            prompt = enriched
            logger.info("Zoom enrich: +4H×16 (prompt=%d)", len(prompt))
    if len(prompt) < max_prompt - 600:
        enriched = _assemble(mem_budget, min(bars_1h + 12, 36), n4=16, with_vol=True)
        if len(enriched) <= max_prompt:
            prompt = enriched
            logger.info("Zoom enrich: +1H bars + volume (prompt=%d)", len(prompt))
    if len(prompt) < max_prompt - 500:
        enriched = _assemble(mem_budget, min(bars_1h + 12, 36), n4=24, with_vol=True)
        if len(enriched) <= max_prompt:
            prompt = enriched
            logger.info("Zoom enrich: +4H×24 (prompt=%d)", len(prompt))

    if len(prompt) <= max_prompt:
        return prompt, meta

    for m in (400, 200, 0):
        for h in (hist_txt, hist_txt[-700:] if len(hist_txt) > 700 else hist_txt, "(đã rút gọn)"):
            prompt = _assemble(m, bars_1h, n4=0, with_vol=False, hist=h)
            if len(prompt) <= max_prompt:
                logger.warning("Prompt trimmed memory/history/zoom to fit %d", max_prompt)
                return prompt, meta

    for n1 in (20, 16, 12):
        prompt = _assemble(0, n1, n4=0, with_vol=False, hist="(đã rút gọn)")
        if len(prompt) <= max_prompt:
            logger.warning("Prompt trimmed 1H CSV to %d to fit %d", n1, max_prompt)
            return prompt, meta

    logger.warning("Prompt still large (%d) — hard truncate tail", len(prompt))
    return prompt[:max_prompt], meta


class LLMAgent:
    def __init__(self) -> None:
        if not config.LLM_API_KEY:
            logger.warning("LLM_API_KEY is empty")
        self.client = OpenAI(
            api_key=config.LLM_API_KEY or "missing",
            base_url=config.LLM_BASE_URL,
        )

    def _chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        max_tokens = max_tokens if max_tokens is not None else max(
            config.LLM_MAX_COMPLETION_TOKENS, 2000
        )

        def _create(use_json: bool):
            kwargs: dict[str, Any] = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            return self.client.chat.completions.create(**kwargs)

        # gpt-oss on Groq often fails json_object on long prompts (empty failed_generation)
        prefer_json = "gpt-oss" not in (config.LLM_MODEL or "").lower()
        resp = None
        if prefer_json:
            try:
                resp = _create(True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("json_object mode failed (%s) — plain retry", exc)
                resp = None
        if resp is None:
            resp = _create(False)

        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning", None) or "").strip()

        if not content:
            logger.warning(
                "Empty content finish=%s reasoning_len=%d — retry plain higher tokens",
                getattr(resp.choices[0], "finish_reason", None),
                len(reasoning),
            )
            resp = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": "Your previous answer was empty. Reply with ONLY one JSON object starting with {",
                    }
                ],
                temperature=0,
                max_tokens=max(max_tokens, 3000),
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning", None) or "").strip()

        if not content and reasoning:
            # Last resort: try extract JSON buried in reasoning
            content = reasoning

        logger.info(
            "LLM chat content_len=%d reasoning_len=%d finish=%s",
            len(content),
            len(reasoning),
            getattr(resp.choices[0], "finish_reason", None),
        )
        return content

    def summarize_decision_for_history(
        self,
        decision: dict[str, Any],
        price: float,
    ) -> str:
        """Compress a verbose decision into one fixed-length line for history."""
        slim = {
            "market_regime": decision.get("market_regime"),
            "action": decision.get("action"),
            "setup_score": decision.get("setup_score"),
            "market_location": decision.get("market_location"),
            "price_action_signal": (decision.get("price_action_signal") or "")[:240],
            "reasoning": (decision.get("reasoning") or "")[:320],
        }
        system = (
            "Nén quyết định trading thành ĐÚNG 1 dòng tiếng Việt/ASCII. "
            f"Tối đa {config.DECISION_SUMMARY_MAX_CHARS} ký tự. "
            "Format: regime | action | score | level/wait | status | 1 câu. "
            "Chỉ nén nội dung đã cho, không suy luận thêm, không markdown."
        )
        user = (
            f"Giá={price}. Quyết định JSON:\n"
            f"{json.dumps(slim, ensure_ascii=False)}"
        )
        raw = self._chat(system=system, user=user, temperature=0, max_tokens=120)
        line = (raw or "").strip().splitlines()[0] if raw else ""
        line = line.strip("` ").strip()
        if len(line) > config.DECISION_SUMMARY_MAX_CHARS:
            line = line[: config.DECISION_SUMMARY_MAX_CHARS - 1] + "…"
        return line

    def get_decision(
        self,
        candles_4h: list[dict[str, Any]],
        candles_1h: list[dict[str, Any]],
        current_price: float,
        current_position: dict[str, Any] | None,
        memory_lessons: str,
        min_score_required: float,
        candles_1d: list[dict[str, Any]] | None = None,
        decision_history: str = "",
    ) -> dict[str, Any]:
        core = _read_text(config.PATHS["CORE_PLAYBOOK"])
        candles_1d = candles_1d or []

        if len(candles_1d) < 30 or len(candles_4h) < 48 or len(candles_1h) < 24:
            logger.warning(
                "Thin candle history: 1D=%d 4H=%d 1H=%d",
                len(candles_1d),
                len(candles_4h),
                len(candles_1h),
            )

        prompt, struct_meta = _build_decision_prompt(
            core=core,
            memory_lessons=memory_lessons,
            decision_history=decision_history,
            candles_1d=candles_1d,
            candles_4h=candles_4h,
            candles_1h=candles_1h,
            current_price=current_price,
            current_position=current_position,
            min_score_required=min_score_required,
        )
        logger.info(
            "Decision prompt chars=%d core=%d hist=%d 1D=%d 4H=%d 1H=%d primary=%s",
            len(prompt),
            min(len(core), config.CORE_PLAYBOOK_MAX_CHARS),
            len(decision_history or ""),
            len(candles_1d),
            len(candles_4h),
            len(candles_1h),
            struct_meta.get("primary_bias"),
        )

        system = (
            "Bạn là Senior Crypto Quant Trader Dow/PA. Chỉ trả về 1 JSON hợp lệ, "
            "không markdown. Phân tích thực chiến: tách cấu trúc "
            "(UPTREND/DOWNTREND/RANGE) và trạng thái "
            "(BREAKOUT_ATTEMPT/BREAKOUT_CONFIRMED/FAILED_BREAKOUT/RETEST/...). "
            "market_regime dạng REGIME_RANGE_BREAKOUT_ATTEMPT khi phù hợp. "
            "Các trường mô tả PHẢI tiếng Việt. "
            "BUY/SELL chỉ khi thesis+level+trigger+invalidation+R:R. "
            "setup_score là chất lượng setup; Python tự kiểm min_score. "
            "Lịch sử quyết định chỉ là ngữ cảnh — ưu tiên dữ liệu live. "
            "Không nói thiếu dữ liệu."
        )

        try:
            raw = self._chat(system=system, user=prompt, temperature=0.1)
            try:
                decision = extract_json_object(raw)
            except ValueError:
                logger.warning(
                    "JSON parse failed, retry strict. Raw preview: %r",
                    (raw or "")[:400],
                )
                repair = (
                    f"Giá={current_price}. primary_bias={struct_meta.get('primary_bias')}.\n"
                    "Chỉ trả về 1 JSON với các key: "
                    "market_regime, dow_structure_analysis, price_action_signal, "
                    "market_location, setup_score, action, entry_price, "
                    "stop_loss_price, take_profit_price, risk_reward_ratio, reasoning.\n"
                    "market_regime ví dụ REGIME_RANGE_BREAKOUT_ATTEMPT. "
                    "Text tiếng Việt. Chưa edge → HOLD, giá = null. "
                    "setup_score chỉ phản ánh chất lượng, không tự ép HOLD theo ngưỡng."
                )
                raw2 = self._chat(system=system, user=repair, temperature=0)
                decision = extract_json_object(raw2)
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_decision failed: %s", exc)
            hold = dict(HOLD_DECISION)
            hold["reasoning"] = f"Lỗi LLM: {exc}"
            hold["primary_bias"] = struct_meta.get("primary_bias")
            return hold

        out = self._normalize_decision(decision, min_score_required)
        out["primary_bias"] = struct_meta.get("primary_bias")
        out["aligned_htf"] = struct_meta.get("aligned_htf")
        out["structure_meta"] = {
            "bars_1d": struct_meta.get("bars_1d"),
            "bars_4h": struct_meta.get("bars_4h"),
            "bars_1h": struct_meta.get("bars_1h"),
            "key_supports": struct_meta.get("key_supports"),
            "key_resistances": struct_meta.get("key_resistances"),
        }
        return out

    def _normalize_decision(
        self,
        decision: dict[str, Any],
        min_score_required: float,
    ) -> dict[str, Any]:
        action = str(decision.get("action", "HOLD")).upper()
        if action not in {"BUY", "SELL", "HOLD"}:
            action = "HOLD"

        try:
            score = float(decision.get("setup_score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0

        regime = str(decision.get("market_regime", "REGIME_SIDEWAY_CHOP"))
        if not regime.startswith("REGIME_"):
            regime = f"REGIME_{regime}" if regime else "REGIME_SIDEWAY_CHOP"

        out = {
            "market_regime": regime,
            "dow_structure_analysis": decision.get("dow_structure_analysis", ""),
            "price_action_signal": decision.get("price_action_signal", ""),
            "market_location": decision.get("market_location", ""),
            "setup_score": score,
            "action": action,
            "entry_price": _to_float(decision.get("entry_price")),
            "stop_loss_price": _to_float(decision.get("stop_loss_price")),
            "take_profit_price": _to_float(decision.get("take_profit_price")),
            "risk_reward_ratio": _to_float(decision.get("risk_reward_ratio")),
            "reasoning": decision.get("reasoning", ""),
        }

        if action in {"BUY", "SELL"} and score < min_score_required:
            out["action"] = "HOLD"
            out["reasoning"] = (
                f"Điểm {score} < ngưỡng tối thiểu {min_score_required}. "
                f"Gốc: {out['reasoning']}"
            )
        return out

    def run_post_mortem(
        self,
        action: str,
        entry: float,
        sl: float | None,
        tp: float | None,
        result: str,
        pnl: float,
        regime: str,
        entry_reasoning: str,
        post_trade_candles: list[dict[str, Any]] | str,
    ) -> dict[str, Any]:
        candles_str = (
            post_trade_candles
            if isinstance(post_trade_candles, str)
            else _compact_candles(post_trade_candles, last_n=20)
        )
        prompt = POST_MORTEM_PROMPT.format(
            action=action,
            entry=entry,
            sl=sl,
            tp=tp,
            result=result,
            pnl=pnl,
            regime=regime,
            entry_reasoning=entry_reasoning,
            post_trade_candles=candles_str,
        )
        try:
            raw = self._chat(
                system=(
                    "Bạn là Risk Officer. Chỉ trả về 1 JSON. "
                    "mistake_or_insight và actionable_rule viết TIẾNG VIỆT."
                ),
                user=prompt,
                temperature=0.3,
            )
            data = extract_json_object(raw)
            return {
                "regime": data.get("regime") or regime,
                "mistake_or_insight": data.get("mistake_or_insight", ""),
                "actionable_rule": data.get("actionable_rule", ""),
                "weight": int(data.get("weight", 1) or 1),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("post_mortem failed: %s", exc)
            return {
                "regime": regime,
                "mistake_or_insight": f"Hậu phẫu LLM lỗi: {exc}",
                "actionable_rule": "Xem lại thủ công; giữ nguyên quy tắc rủi ro nghiêm ngặt.",
                "weight": 1,
            }


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
