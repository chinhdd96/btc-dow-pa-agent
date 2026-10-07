"""Tính HH/HL, S/R, channel trên FULL nến đã fetch — gửi LLM bản tóm tắt ngắn."""

from __future__ import annotations

from typing import Any

# Ước lượng khoảng thời gian từ bars_ago
_HOURS_PER_BAR = {"1W": 168.0, "1D": 24.0, "4H": 4.0, "1H": 1.0}


def _ohlc(candles: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for c in candles:
        out.append(
            (
                float(c.get("open", 0)),
                float(c.get("high", 0)),
                float(c.get("low", 0)),
                float(c.get("close", 0)),
            )
        )
    return out


def _age_label(bars_ago: int, timeframe: str) -> str:
    hours = bars_ago * _HOURS_PER_BAR.get(timeframe, 1.0)
    if hours >= 24:
        return f"-{hours / 24:.0f}d"
    if hours >= 1:
        return f"-{hours:.0f}h"
    return f"-{bars_ago}b"


def _ema(closes: list[float], period: int = 20) -> float | None:
    if not closes:
        return None
    if len(closes) < period:
        return round(sum(closes) / len(closes), 1)
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 1)


def _atr(
    ohlc: list[tuple[float, float, float, float]],
    period: int = 14,
) -> float | None:
    """Wilder ATR — ổn định hơn SMA của TR."""
    if len(ohlc) < period + 1:
        if len(ohlc) < 2:
            return None
        trs = []
        for i in range(1, len(ohlc)):
            h, l, prev_c = ohlc[i][1], ohlc[i][2], ohlc[i - 1][3]
            trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
        return round(sum(trs) / len(trs), 1) if trs else None

    trs: list[float] = []
    for i in range(1, len(ohlc)):
        h, l, prev_c = ohlc[i][1], ohlc[i][2], ohlc[i - 1][3]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 1)


def _find_swings(
    ohlc: list[tuple[float, float, float, float]],
    left: int = 2,
    right: int = 2,
) -> list[dict[str, Any]]:
    swings: list[dict[str, Any]] = []
    n = len(ohlc)
    if n < left + right + 1:
        return swings
    for i in range(left, n - right):
        hi = ohlc[i][1]
        lo = ohlc[i][2]
        is_high = all(
            hi >= ohlc[j][1] for j in range(i - left, i + right + 1) if j != i
        )
        is_low = all(
            lo <= ohlc[j][2] for j in range(i - left, i + right + 1) if j != i
        )
        if is_high:
            swings.append({"type": "H", "price": round(hi, 1), "bars_ago": n - 1 - i})
        if is_low:
            swings.append({"type": "L", "price": round(lo, 1), "bars_ago": n - 1 - i})
    swings.sort(key=lambda s: s["bars_ago"], reverse=True)
    return swings


def _annotate_swings(
    swings: list[dict[str, Any]],
    timeframe: str,
) -> list[dict[str, Any]]:
    """Gán HH/HL/LH/LL trên TOÀN BỘ chuỗi swing (không cắt trước khi gán nhãn)."""
    last_h: float | None = None
    last_l: float | None = None
    out: list[dict[str, Any]] = []
    for s in swings:
        price = float(s["price"])
        if s["type"] == "H":
            if last_h is None:
                label = "H"
            elif price > last_h:
                label = "HH"
            elif price < last_h:
                label = "LH"
            else:
                label = "EH"
            last_h = price
        else:
            if last_l is None:
                label = "L"
            elif price > last_l:
                label = "HL"
            elif price < last_l:
                label = "LL"
            else:
                label = "EL"
            last_l = price
        out.append(
            {
                **s,
                "label": label,
                "age": _age_label(int(s["bars_ago"]), timeframe),
            }
        )
    return out


def _labels_str(annotated: list[dict[str, Any]]) -> list[str]:
    return [f"{a['label']}@{a['price']}({a['age']})" for a in annotated]


def _bias_from_labels(labels: list[str]) -> str:
    text = " ".join(labels)
    hh = text.count("HH")
    hl = text.count("HL")
    lh = text.count("LH")
    ll = text.count("LL")
    if hh + hl >= 3 and hh >= 1 and hl >= 1 and lh + ll <= 1:
        return "UPTREND"
    if lh + ll >= 3 and lh >= 1 and ll >= 1 and hh + hl <= 1:
        return "DOWNTREND"
    return "SIDEWAY_CHOP"


def _pct(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return round((a - b) / b * 100, 3)


def _detect_channel(
    annotated: list[dict[str, Any]],
    atr: float | None,
) -> dict[str, Any]:
    """
    Channel từ 2–3 swing H gần + 2–3 swing L gần.
    ASCENDING / DESCENDING / HORIZONTAL / NONE
    """
    highs = [a for a in annotated if a["type"] == "H"][-3:]
    lows = [a for a in annotated if a["type"] == "L"][-3:]
    if len(highs) < 2 or len(lows) < 2:
        return {"type": "NONE", "upper": None, "lower": None, "width_atr": None}

    h_slope = highs[-1]["price"] - highs[0]["price"]
    l_slope = lows[-1]["price"] - lows[0]["price"]
    upper = round(sum(h["price"] for h in highs) / len(highs), 1)
    lower = round(sum(l["price"] for l in lows) / len(lows), 1)
    width = upper - lower
    thr = (atr or abs(upper) * 0.005) * 0.35

    if h_slope > thr and l_slope > thr:
        ctype = "ASCENDING"
    elif h_slope < -thr and l_slope < -thr:
        ctype = "DESCENDING"
    elif abs(h_slope) <= thr and abs(l_slope) <= thr:
        ctype = "HORIZONTAL"
    else:
        ctype = "EXPANDING_OR_MIXED"

    width_atr = round(width / atr, 2) if atr and atr > 0 else None
    return {
        "type": ctype,
        "upper": upper,
        "lower": lower,
        "width": round(width, 1),
        "width_atr": width_atr,
        "h_slope": round(h_slope, 1),
        "l_slope": round(l_slope, 1),
    }


def _cluster_levels(
    prices: list[float],
    tol_pct: float = 0.004,
) -> list[float]:
    """Gộp mức gần nhau thành S/R cluster."""
    if not prices:
        return []
    sorted_p = sorted(prices)
    clusters: list[list[float]] = [[sorted_p[0]]]
    for p in sorted_p[1:]:
        if abs(p - clusters[-1][-1]) / max(p, 1) <= tol_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [round(sum(c) / len(c), 1) for c in clusters]


def _last_bar_pa(ohlc: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    if not ohlc:
        return {"hint": "NONE"}
    o, h, l, c = ohlc[-1]
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body_pct = body / rng
    hints: list[str] = []
    if lower >= 0.5 * rng and c >= o and body_pct < 0.45:
        hints.append("PINBAR_BULL")
    if upper >= 0.5 * rng and c <= o and body_pct < 0.45:
        hints.append("PINBAR_BEAR")
    if body_pct >= 0.8:
        hints.append("MARUBOZU_BULL" if c > o else "MARUBOZU_BEAR")
    if len(ohlc) >= 2:
        po, ph, pl, pc = ohlc[-2]
        if c > o and c >= ph and o <= pl and pc < po:
            hints.append("ENGULFING_BULL")
        if c < o and c <= pl and o >= ph and pc > po:
            hints.append("ENGULFING_BEAR")
        if h <= ph and l >= pl:
            hints.append("INSIDE_BAR")
    return {
        "hint": ",".join(hints) if hints else "NONE",
        "last_bar": "BULL" if c > o else ("BEAR" if c < o else "DOJI"),
        "body_pct": round(body_pct, 2),
        "upper_wick_pct": round(upper / rng, 2),
        "lower_wick_pct": round(lower / rng, 2),
        "range": round(rng, 1),
    }


def _micro_channel(
    ohlc: list[tuple[float, float, float, float]],
    lookback: int = 5,
) -> str:
    if len(ohlc) < lookback:
        return "NONE"
    window = ohlc[-lookback:]
    bull = all(window[i][2] >= window[i - 1][2] for i in range(1, lookback))
    bear = all(window[i][1] <= window[i - 1][1] for i in range(1, lookback))
    if bull and not bear:
        return "BULL_MICRO"
    if bear and not bull:
        return "BEAR_MICRO"
    return "NONE"


def aggregate_to_weekly(candles_1d: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gộp nến Daily → Weekly (7 nến/1 tuần) để có nhìn siêu rộng."""
    if not candles_1d:
        return []
    weeks: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []
    for c in candles_1d:
        bucket.append(c)
        if len(bucket) == 7:
            weeks.append(
                {
                    "open": float(bucket[0]["open"]),
                    "high": max(float(x["high"]) for x in bucket),
                    "low": min(float(x["low"]) for x in bucket),
                    "close": float(bucket[-1]["close"]),
                }
            )
            bucket = []
    if len(bucket) >= 3:
        weeks.append(
            {
                "open": float(bucket[0]["open"]),
                "high": max(float(x["high"]) for x in bucket),
                "low": min(float(x["low"]) for x in bucket),
                "close": float(bucket[-1]["close"]),
            }
        )
    return weeks


def summarize_structure(
    candles: list[dict[str, Any]],
    timeframe: str,
    swing_left: int = 2,
    swing_right: int = 2,
    swing_keep: int = 12,
) -> dict[str, Any]:
    """
    Tính trên TOÀN BỘ `candles` đã fetch.
    Output gọn để nhét prompt — không phụ thuộc CSV mỏng.
    """
    ohlc = _ohlc(candles)
    if not ohlc:
        return {"timeframe": timeframe, "bars": 0, "error": "empty_candles"}

    # Swing lớn hơn trên HTF (ít nhiễu)
    if timeframe in {"1W", "1D"}:
        swing_left = max(swing_left, 3)
        swing_right = max(swing_right, 3)

    highs = [x[1] for x in ohlc]
    lows = [x[2] for x in ohlc]
    closes = [x[3] for x in ohlc]
    price = closes[-1]
    range_high = max(highs)
    range_low = min(lows)
    ema20 = _ema(closes, 20)
    ema25 = _ema(closes, 25)
    atr14 = _atr(ohlc, 14)

    swings = _find_swings(ohlc, left=swing_left, right=swing_right)
    annotated_all = _annotate_swings(swings, timeframe)
    annotated = (
        annotated_all[-swing_keep:]
        if len(annotated_all) > swing_keep
        else annotated_all
    )
    labels = _labels_str(annotated)
    bias = _bias_from_labels(_labels_str(annotated_all[-max(swing_keep, 8) :]))
    channel = _detect_channel(annotated, atr14)
    pa = _last_bar_pa(ohlc)
    micro = _micro_channel(ohlc, 5)

    all_h = [s["price"] for s in swings if s["type"] == "H"]
    all_l = [s["price"] for s in swings if s["type"] == "L"]
    r_clusters = [x for x in _cluster_levels(all_h) if x >= price]
    s_clusters = [x for x in _cluster_levels(all_l) if x <= price]
    s_clusters = sorted(s_clusters, reverse=True)
    r_clusters = sorted(r_clusters)

    if not r_clusters:
        r_clusters = [round(range_high, 1)]
    if not s_clusters:
        s_clusters = [round(range_low, 1)]

    round_levels: list[int] = []
    base = int(price // 1000) * 1000
    for offset in (-2000, -1000, 0, 1000, 2000):
        lvl = base + offset
        if abs(lvl - price) / max(price, 1) < 0.06:
            round_levels.append(lvl)

    near_r = r_clusters[0]
    near_s = s_clusters[0]
    span = range_high - range_low
    pos_in_range = round((price - range_low) / span, 3) if span > 0 else None

    # Vị trí giá trong channel
    ch_pos = None
    if channel.get("upper") and channel.get("lower"):
        cw = channel["upper"] - channel["lower"]
        if cw > 0:
            ch_pos = round((price - channel["lower"]) / cw, 3)

    cover = _age_label(len(ohlc) - 1, timeframe).lstrip("-")

    return {
        "timeframe": timeframe,
        "bars_used": len(ohlc),
        "history_cover": cover,
        "price": round(price, 1),
        "range_high": round(range_high, 1),
        "range_low": round(range_low, 1),
        "pos_in_range": pos_in_range,
        "ema20": ema20,
        "ema25": ema25,
        "dist_ema20_pct": _pct(price, ema20) if ema20 else None,
        "dist_ema25_pct": _pct(price, ema25) if ema25 else None,
        "atr14": atr14,
        "atr_pct": round(atr14 / price * 100, 3) if atr14 and price else None,
        "bias_hint": bias,
        "swing_timeline": labels,
        "swing_count_total": len(annotated_all),
        "channel": channel,
        "channel_pos": ch_pos,
        "nearest_resistance": near_r,
        "nearest_support": near_s,
        "dist_r_pct": _pct(near_r, price) if near_r else None,
        "dist_s_pct": _pct(near_s, price) if near_s else None,
        "supports": s_clusters[:4],
        "resistances": r_clusters[:4],
        "round_levels": round_levels,
        "pa_last": pa,
        "micro_channel": micro,
    }


def structure_to_text(summary: dict[str, Any], *, role: str = "htf") -> str:
    """role=htf: Dow bias+S/R+channel (no PA). role=entry: include PA_last."""
    if summary.get("error"):
        return f"{summary.get('timeframe')}: KHÔNG CÓ NẾN"
    ch = summary.get("channel") or {}
    lines = [
        (
            f"TF={summary['timeframe']} | role={role} | "
            f"computed_on={summary['bars_used']} bars (~{summary.get('history_cover')}) | "
            f"price={summary['price']} | bias={summary.get('bias_hint')}"
        ),
        (
            f"range={summary['range_low']}..{summary['range_high']} "
            f"pos={summary.get('pos_in_range')} | "
            f"ATR14={summary.get('atr14')} ({summary.get('atr_pct')}%)"
        ),
        (
            f"channel={ch.get('type')} upper={ch.get('upper')} lower={ch.get('lower')} "
            f"width={ch.get('width')} ({ch.get('width_atr')}×ATR) "
            f"pos_in_ch={summary.get('channel_pos')}"
        ),
        (
            f"EMA20={summary.get('ema20')}({summary.get('dist_ema20_pct')}%) "
            f"EMA25={summary.get('ema25')}({summary.get('dist_ema25_pct')}%)"
        ),
        f"Dow timeline: {' → '.join(summary.get('swing_timeline') or [])}",
        (
            f"S={summary.get('supports')} gần={summary.get('nearest_support')} "
            f"({summary.get('dist_s_pct')}%) | "
            f"R={summary.get('resistances')} gần={summary.get('nearest_resistance')} "
            f"({summary.get('dist_r_pct')}%)"
        ),
        f"round={summary.get('round_levels')}",
    ]
    if role == "entry":
        pa = summary.get("pa_last") or {}
        lines.append(
            f"PA_last={pa.get('hint')} bar={pa.get('last_bar')} "
            f"micro={summary.get('micro_channel')}"
        )
    return "\n".join(lines)


def compute_trade_flags(
    candles_4h: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
    near_s: float | None,
    near_r: float | None,
) -> dict[str, Any]:
    """Pre-computed numeric flags for LLM (avoid misreading OHLC)."""
    ohlc_4h = _ohlc(candles_4h)
    ohlc_1h = _ohlc(candles_1h)
    out: dict[str, Any] = {
        "range_4h_low": None,
        "range_4h_high": None,
        "pos_in_range_4h": None,
        "range_1h_low": None,
        "range_1h_high": None,
        "pos_in_range_1h": None,
        "last_close_1h": None,
        "close_above_R": False,
        "close_below_S": False,
        "sweep_high_close_back": False,
        "sweep_low_close_back": False,
        "local_high_1h": None,
        "local_low_1h": None,
        "broke_local_high": False,
        "broke_local_low": False,
        "atr_1h": None,
        "atr_4h": None,
        "near_S": near_s,
        "near_R": near_r,
    }
    if not ohlc_1h:
        return out

    def _range_pos(ohlc: list[tuple[float, float, float, float]]) -> tuple[float, float, float | None]:
        highs = [x[1] for x in ohlc]
        lows = [x[2] for x in ohlc]
        rh, rl = max(highs), min(lows)
        price = ohlc[-1][3]
        span = rh - rl
        pos = round((price - rl) / span, 3) if span > 0 else None
        return rl, rh, pos

    if ohlc_4h:
        rl4, rh4, pos4 = _range_pos(ohlc_4h[-48:] if len(ohlc_4h) > 48 else ohlc_4h)
        out["range_4h_low"] = round(rl4, 1)
        out["range_4h_high"] = round(rh4, 1)
        out["pos_in_range_4h"] = pos4
        out["atr_4h"] = _atr(ohlc_4h, 14)

    rl1, rh1, pos1 = _range_pos(ohlc_1h[-24:] if len(ohlc_1h) > 24 else ohlc_1h)
    out["range_1h_low"] = round(rl1, 1)
    out["range_1h_high"] = round(rh1, 1)
    out["pos_in_range_1h"] = pos1
    out["atr_1h"] = _atr(ohlc_1h, 14)

    last_close = round(ohlc_1h[-1][3], 1)
    out["last_close_1h"] = last_close
    if near_r is not None:
        out["close_above_R"] = last_close > float(near_r)
    if near_s is not None:
        out["close_below_S"] = last_close < float(near_s)

    recent = ohlc_1h[-3:]
    for o, h, l, c in recent:
        if near_r is not None and h > float(near_r) and c <= float(near_r):
            out["sweep_high_close_back"] = True
        if near_s is not None and l < float(near_s) and c >= float(near_s):
            out["sweep_low_close_back"] = True

    lookback = ohlc_1h[-11:-1] if len(ohlc_1h) > 11 else ohlc_1h[:-1]
    if lookback:
        local_h = max(x[1] for x in lookback)
        local_l = min(x[2] for x in lookback)
        out["local_high_1h"] = round(local_h, 1)
        out["local_low_1h"] = round(local_l, 1)
        out["broke_local_high"] = last_close > local_h
        out["broke_local_low"] = last_close < local_l

    return out


def trade_flags_to_text(flags: dict[str, Any]) -> str:
    """Compact block (<600 chars) for prompt injection."""
    if not flags:
        return "(không có trade_flags)"
    parts = [
        f"4H range={flags.get('range_4h_low')}..{flags.get('range_4h_high')} "
        f"pos={flags.get('pos_in_range_4h')} ATR={flags.get('atr_4h')}",
        f"1H range={flags.get('range_1h_low')}..{flags.get('range_1h_high')} "
        f"pos={flags.get('pos_in_range_1h')} ATR={flags.get('atr_1h')}",
        f"S={flags.get('near_S')} R={flags.get('near_R')} "
        f"close={flags.get('last_close_1h')} "
        f"above_R={flags.get('close_above_R')} below_S={flags.get('close_below_S')}",
        f"sweep_hi_back={flags.get('sweep_high_close_back')} "
        f"sweep_lo_back={flags.get('sweep_low_close_back')}",
        f"local_H/L={flags.get('local_high_1h')}/{flags.get('local_low_1h')} "
        f"broke_H={flags.get('broke_local_high')} broke_L={flags.get('broke_local_low')}",
    ]
    text = "\n".join(parts)
    return text[:580] + "…" if len(text) > 580 else text


def build_multi_tf_structure(
    candles_1d: list[dict[str, Any]],
    candles_4h: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
) -> tuple[str, str, str, str, dict[str, Any]]:
    """
    Trả (text_1w, text_1d, text_4h, text_1h, meta).
    primary_bias LUÔN từ 1D (Dow). 1H không đảo primary.
    """
    weekly = aggregate_to_weekly(candles_1d)
    s1w = summarize_structure(weekly, "1W", swing_keep=8)
    s1d = summarize_structure(candles_1d, "1D", swing_keep=12)
    s4 = summarize_structure(candles_4h, "4H", swing_keep=12)
    s1 = summarize_structure(candles_1h, "1H", swing_keep=10)

    bias_1w = s1w.get("bias_hint")
    bias_1d = s1d.get("bias_hint") or "SIDEWAY_CHOP"
    bias_4h = s4.get("bias_hint")
    bias_1h = s1.get("bias_hint")
    trend = {"UPTREND", "DOWNTREND"}
    primary = bias_1d
    aligned_htf = bias_1d in trend and bias_1d == bias_4h

    supports: list[float] = []
    resistances: list[float] = []
    for src in (s1d, s4):
        supports.extend(float(x) for x in (src.get("supports") or []) if x is not None)
        resistances.extend(
            float(x) for x in (src.get("resistances") or []) if x is not None
        )
    supports = sorted(set(supports), reverse=True)[:5]
    resistances = sorted(set(resistances))[:5]

    near_s = supports[0] if supports else s4.get("nearest_support")
    near_r = resistances[0] if resistances else s4.get("nearest_resistance")
    trade_flags = compute_trade_flags(candles_4h, candles_1h, near_s, near_r)

    meta = {
        "bias_1w": bias_1w,
        "bias_1d": bias_1d,
        "bias_4h": bias_4h,
        "bias_1h": bias_1h,
        "primary_bias": primary,
        "aligned_htf": aligned_htf,
        "aligned_all": aligned_htf and bias_4h == bias_1h,
        "bars_1w": s1w.get("bars_used", 0),
        "bars_1d": s1d.get("bars_used", 0),
        "bars_4h": s4.get("bars_used", 0),
        "bars_1h": s1.get("bars_used", 0),
        "channel_1d": (s1d.get("channel") or {}).get("type"),
        "channel_4h": (s4.get("channel") or {}).get("type"),
        "key_supports": supports,
        "key_resistances": resistances,
        "nearest_support": near_s,
        "nearest_resistance": near_r,
        "trade_flags": trade_flags,
        "trade_flags_text": trade_flags_to_text(trade_flags),
        "history_source": "local_cache",
    }
    return (
        structure_to_text(s1w, role="htf"),
        structure_to_text(s1d, role="htf"),
        structure_to_text(s4, role="htf"),
        structure_to_text(s1, role="entry"),
        meta,
    )


def build_dual_structure(
    candles_4h: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    _, _, t4, t1, meta = build_multi_tf_structure([], candles_4h, candles_1h)
    return t4, t1, meta
