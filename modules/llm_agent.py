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
Tìm lệnh có xác suất thắng và kỳ vọng dương — ưu tiên BUY/SELL khi thấy edge.
Dow và Price Action là nền tảng đọc thị trường, KHÔNG phải luật cứng.
Chỉ HOLD khi thật sự không có edge hoặc không đặt được SL/TP hợp lý.

DỮ LIỆU:

1. DOW & PRICE ACTION PLAYBOOK
---
{core_playbook_content}
---

2. MEMORY / KINH NGHIỆM
---
{regime_memory_lessons}
---

2b. LỊCH SỬ QUYẾT ĐỊNH (mới nhất ở dưới)
---
{decision_history}
---
Chỉ tham khảo. Số liệu live + trade_flags luôn ưu tiên hơn history.

3. CẤU TRÚC THỊ TRƯỜNG

1W: {structure_1w}
1D: {structure_1d}
4H: {structure_4h}
1H: {structure_1h}
alignment: {structure_align}

4. CỜ SỐ LIỆU (Python pre-compute — kiểm tra, không suy diễn ngược)
---
{trade_flags}
---

ZOOM PA:
{zoom_block}

GIÁ HIỆN TẠI: {current_price}
VỊ THẾ HIỆN TẠI: {current_position}

QUY TẮC PHÂN TÍCH:

1. Chuỗi: STRUCTURE → STATE → BIAS → TRIGGER → THESIS + INVALIDATION → R:R.
   Structure: UPTREND/DOWNTREND/RANGE. State: FAILED_BREAKOUT/RETEST/SWEEP/...
   Bias ngắn hạn có thể ngược 1D nếu nêu lý do.

2. RANGE ≠ HOLD. Giữa range → HOLD. Biên range + PA xác nhận → trade.

3. FAILED BREAKOUT / LIQUIDITY SWEEP là trigger hợp lệ — không cần pinbar/engulfing.
   PA: failed breakout, rejection, sweep, break local structure, LH/HL, retest fail,
   compression→expansion, reclaim/loss of level, displacement, consecutive closes.

4. Kiểm tra số (trade_flags + OHLC zoom):
   - KHÔNG gọi breakout nếu close chưa vượt level.
   - KHÔNG gọi breakdown nếu close chưa dưới level.

5. signal_level: L1 (<5 điểm, quan sát) | L2 (5–7, có trigger) | L3 (>7, nhiều xác nhận).
   setup_score = chất lượng 0–10. win_probability = xác suất thắng 0–1.
   Ngưỡng tham chiếu: {min_score_required}. R:R tối thiểu: {min_rr}.

6. BUY/SELL: entry + SL (sau invalidation + buffer ATR) + TP + R:R ≥ {min_rr}.
   HOLD → entry/SL/TP/R:R = null.

CẤM: "thiếu dữ liệu" — suy luận từ structure + trade_flags + CSV.

OUTPUT: 1 JSON, không markdown. action: BUY|SELL|HOLD.
market_regime = REGIME_{{CẤU_TRÚC}}_{{TRẠNG_THÁI}}. Text tiếng Việt.

FORMAT:
{{
  "structure": "RANGE",
  "state": "FAILED_BREAKOUT",
  "short_term_bias": "bearish",
  "trigger": "Sweep R + close back inside + break local low",
  "invalidation": "Reclaim resistance",
  "target_logic": "Range midpoint / range low",
  "signal_level": 2,
  "win_probability": 0.62,
  "market_regime": "REGIME_RANGE_FAILED_BREAKOUT",
  "dow_structure_analysis": "Mô tả Dow",
  "price_action_signal": "Mô tả PA",
  "market_location": "Vị trí giá",
  "setup_score": 6.5,
  "action": "SELL",
  "entry_price": 86500.0,
  "stop_loss_price": 87200.0,
  "take_profit_price": 84800.0,
  "risk_reward_ratio": 2.4,
  "reasoning": "Lý do BUY/SELL/HOLD"
}}
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
    "structure": "RANGE",
    "state": "UNKNOWN",
    "short_term_bias": "neutral",
    "trigger": "",
    "invalidation": "",
    "target_logic": "",
    "signal_level": 1,
    "win_probability": 0.0,
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


def _normalize_setup_score(raw: Any) -> float:
    """Stabilize LLM score onto [0, 10]."""
    try:
        score = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    if score != score:  # NaN
        return 0.0
    # Common scale mistakes: 0–1 → ×10; 0–100 → ÷10
    if 0 < score <= 1.0:
        score *= 10.0
    elif score > 10.0:
        if score <= 100.0:
            score /= 10.0
        else:
            score = 10.0
    return max(0.0, min(10.0, round(score, 2)))


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "rate_limit" in text
        or "rate limit" in text
        or "tokens per day" in text
        or "tpd" in text
    )


def _friendly_llm_error(exc: BaseException) -> str:
    if _is_rate_limit_error(exc):
        return (
            "HOLD: hết hạn mức token ngày của Groq (TPD). "
            "Bot sẽ thử lại chu kỳ sau — không phải lỗi phân tích thị trường."
        )
    return f"Lỗi LLM: {exc}"


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

    hist_placeholder = "(chưa có lịch sử quyết định gần đây)"
    hist_budget = config.DECISION_HISTORY_PROMPT_MAX_CHARS
    raw_hist = (decision_history or "").strip()
    hist_lines: list[str] = []
    if raw_hist and raw_hist != hist_placeholder:
        hist_lines = [ln.strip() for ln in raw_hist.splitlines() if ln.strip()]

    def _join_hist(lines: list[str]) -> str:
        return "\n".join(lines) if lines else hist_placeholder

    def _fit_hist_budget(lines: list[str]) -> list[str]:
        """Keep newest lines within char budget (drop oldest first)."""
        if not lines:
            return []
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            add = len(line) + (1 if kept else 0)
            if total + add > hist_budget:
                break
            kept.append(line)
            total += add
        kept.reverse()
        return kept

    hist_lines = _fit_hist_budget(hist_lines)
    hist_txt = _join_hist(hist_lines)
    trade_flags_txt = str(meta.get("trade_flags_text") or "(không có trade_flags)")
    min_rr = config.MIN_RR

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
            trade_flags=trade_flags_txt,
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
            min_rr=min_rr,
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

    # Over budget: drop oldest decision-history lines first (keep OUTPUT/JSON intact).
    # e.g. 20 → 18 → 15 → 12 … (file history trên disk không bị xóa).
    n_hist = len(hist_lines)
    keep_targets = []
    for k in (18, 15, 12, 10, 8, 5, 3, 1, 0):
        if k < n_hist:
            keep_targets.append(k)
    # Also step down one-by-one near the top if n is between ladder rungs
    for k in range(n_hist - 1, 0, -1):
        if k not in keep_targets and k >= 15:
            keep_targets.append(k)
    keep_targets = sorted(set(keep_targets), reverse=True)

    for keep_n in keep_targets:
        h_lines = hist_lines[-keep_n:] if keep_n else []
        h = _join_hist(h_lines)
        prompt = _assemble(mem_budget, bars_1h, n4=0, with_vol=False, hist=h)
        if len(prompt) <= max_prompt:
            logger.warning(
                "Prompt fit by reducing decision history %d → %d lines (chars=%d)",
                n_hist,
                keep_n,
                len(prompt),
            )
            return prompt, meta

    # Still over: shrink memory, then 1H zoom; history already minimal/empty
    h_min = _join_hist(hist_lines[-1:] if hist_lines else [])
    for m in (400, 200, 0):
        prompt = _assemble(m, bars_1h, n4=0, with_vol=False, hist=h_min)
        if len(prompt) <= max_prompt:
            logger.warning(
                "Prompt trimmed memory to %d (+ hist≤1) to fit %d", m, max_prompt
            )
            return prompt, meta

    for n1 in (20, 16, 12):
        prompt = _assemble(0, n1, n4=0, with_vol=False, hist=hist_placeholder)
        if len(prompt) <= max_prompt:
            logger.warning(
                "Prompt trimmed 1H CSV to %d + dropped history to fit %d",
                n1,
                max_prompt,
            )
            return prompt, meta

    # Last resort: shrink core playbook from its end (never cut prompt OUTPUT tail)
    for core_cap in (3000, 2500, 2000, 1500):
        core_txt = core_txt[:core_cap]
        prompt = _assemble(0, 12, n4=0, with_vol=False, hist=hist_placeholder)
        if len(prompt) <= max_prompt:
            logger.warning(
                "Prompt trimmed core_playbook to %d chars to fit %d",
                core_cap,
                max_prompt,
            )
            return prompt, meta

    logger.error(
        "Prompt still over budget (%d > %d) after history/memory/core trim — sending as-is",
        len(prompt),
        max_prompt,
    )
    return prompt, meta


class LLMAgent:
    def __init__(self) -> None:
        if not config.LLM_API_KEY:
            logger.warning("LLM_API_KEY is empty")
        # Avoid burning daily TPD on automatic 429 retries
        self.client = OpenAI(
            api_key=config.LLM_API_KEY or "missing",
            base_url=config.LLM_BASE_URL,
            max_retries=0,
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
        max_tokens = (
            max_tokens
            if max_tokens is not None
            else max(256, int(config.LLM_MAX_COMPLETION_TOKENS))
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
                if _is_rate_limit_error(exc):
                    raise
                logger.warning("json_object mode failed (%s) — plain retry", exc)
                resp = None
        if resp is None:
            resp = _create(False)

        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning", None) or "").strip()

        # Empty retry burns more TPD — skip when already rate-limited risk / keep once only
        if not content and not _is_rate_limit_error(
            Exception(getattr(resp.choices[0], "finish_reason", "") or "")
        ):
            logger.warning(
                "Empty content finish=%s reasoning_len=%d — one plain retry",
                getattr(resp.choices[0], "finish_reason", None),
                len(reasoning),
            )
            try:
                resp = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "Your previous answer was empty. "
                                "Reply with ONLY one JSON object starting with {"
                            ),
                        }
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                reasoning = (getattr(msg, "reasoning", None) or "").strip()
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc):
                    raise
                logger.warning("empty-content retry failed: %s", exc)

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
            "Bạn là Senior Crypto Quant Trader. Trả về 1 JSON hợp lệ, không markdown. "
            "Mục tiêu: tìm cơ hội có edge và vào lệnh khi hợp lý — không quá an toàn. "
            "Dow/PA là nền tảng, không phải luật cứng. RANGE có thể trade ở biên. "
            "Failed breakout/sweep là trigger hợp lệ. "
            "Đối chiếu trade_flags trước khi kết luận breakout/breakdown. "
            "setup_score 0–10, win_probability 0–1, signal_level 1–3. "
            "BUY/SELL cần SL/TP hợp lý. Text tiếng Việt."
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
                    "Chỉ trả về 1 JSON với các key: structure, state, short_term_bias, "
                    "trigger, invalidation, target_logic, signal_level, win_probability, "
                    "market_regime, dow_structure_analysis, price_action_signal, "
                    "market_location, setup_score, action, entry_price, "
                    "stop_loss_price, take_profit_price, risk_reward_ratio, reasoning.\n"
                    "setup_score 0–10, signal_level 1–3, win_probability 0–1. "
                    "Text tiếng Việt. Chưa edge → HOLD, giá = null."
                )
                raw2 = self._chat(system=system, user=repair, temperature=0)
                decision = extract_json_object(raw2)
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc):
                logger.warning("get_decision rate-limited: %s", exc)
            else:
                logger.exception("get_decision failed: %s", exc)
            hold = dict(HOLD_DECISION)
            hold["reasoning"] = _friendly_llm_error(exc)
            hold["primary_bias"] = struct_meta.get("primary_bias")
            hold["llm_error"] = "rate_limit" if _is_rate_limit_error(exc) else "error"
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

        score = _normalize_setup_score(decision.get("setup_score", 0))

        regime = str(decision.get("market_regime", "REGIME_SIDEWAY_CHOP"))
        if not regime.startswith("REGIME_"):
            regime = f"REGIME_{regime}" if regime else "REGIME_SIDEWAY_CHOP"

        try:
            signal_level = int(decision.get("signal_level") or 1)
        except (TypeError, ValueError):
            signal_level = 1
        signal_level = max(1, min(3, signal_level))

        try:
            win_prob = float(decision.get("win_probability") or 0)
        except (TypeError, ValueError):
            win_prob = 0.0
        win_prob = max(0.0, min(1.0, round(win_prob, 3)))

        out = {
            "structure": str(decision.get("structure") or ""),
            "state": str(decision.get("state") or ""),
            "short_term_bias": str(decision.get("short_term_bias") or ""),
            "trigger": str(decision.get("trigger") or ""),
            "invalidation": str(decision.get("invalidation") or ""),
            "target_logic": str(decision.get("target_logic") or ""),
            "signal_level": signal_level,
            "win_probability": win_prob,
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
