"""Strategy Spec: catalog conditions → compiled signal (Champion Spec path)."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import config
from modules.indicator_catalog import OPS, col_key, validate_indicator

logger = logging.getLogger(__name__)

SignalFn = Callable[[pd.Series, pd.Series], Optional[int]]


def spec_id(spec: dict[str, Any]) -> str:
    raw = json.dumps(normalize_spec(spec), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _fmt_cond(cond: dict[str, Any]) -> str:
    ind = str(cond.get("ind") or "?")
    params = cond.get("params") or {}
    op = str(cond.get("op") or "")
    param_s = ",".join(f"{k}={v}" for k, v in sorted(params.items())) if params else ""
    head = f"{ind}({param_s})" if param_s else ind
    vs = cond.get("vs")
    if isinstance(vs, dict) and vs.get("ind"):
        vp = vs.get("params") or {}
        vp_s = ",".join(f"{k}={v}" for k, v in sorted(vp.items())) if vp else ""
        right = f"{vs['ind']}({vp_s})" if vp_s else str(vs["ind"])
        return f"{head} {op} {right}"
    val = cond.get("value")
    if val is None:
        return f"{head} {op}".strip()
    return f"{head} {op} {val}"


def format_spec_telegram(
    spec: dict[str, Any] | None,
    *,
    sid: str | None = None,
    promote_mode: str | None = None,
    research_mode: str | None = None,
    source: str = "research",
) -> str:
    """Human-readable Telegram note for a chosen indicator family/Spec."""
    if not spec:
        return "SPEC: (none)"
    s = normalize_spec(spec)
    sid = sid or spec_id(s)
    lines = [
        "FAMILY/SPEC ĐÃ CHỌN (chỉ báo)",
        f"name={s.get('name') or '—'} id={sid}",
        f"source={source} mode={research_mode or '—'} promote={promote_mode or '—'}",
        f"mtf={bool(s.get('mtf_filter'))}",
    ]
    long_c = s.get("entry_long") or []
    short_c = s.get("entry_short") or []
    filters = s.get("filters") or []
    if long_c:
        lines.append("LONG: " + " AND ".join(_fmt_cond(c) for c in long_c if isinstance(c, dict)))
    else:
        lines.append("LONG: (none)")
    if short_c:
        lines.append("SHORT: " + " AND ".join(_fmt_cond(c) for c in short_c if isinstance(c, dict)))
    else:
        lines.append("SHORT: (none)")
    if filters:
        lines.append("FILTER: " + " AND ".join(_fmt_cond(c) for c in filters if isinstance(c, dict)))
    else:
        lines.append("FILTER: (none)")
    ex = s.get("exit") or {}
    lines.append(
        f"EXIT: atr×{ex.get('atr_stop')} RR={ex.get('reward_r')} lev={ex.get('leverage')}x"
    )
    return "\n".join(lines)


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(spec)
    out.setdefault("name", "unnamed")
    out.setdefault("entry_long", [])
    out.setdefault("entry_short", [])
    out.setdefault("filters", [])
    exit_cfg = dict(out.get("exit") or {})
    exit_cfg.setdefault("atr_period", 14)
    exit_cfg.setdefault("atr_stop", 2.0)
    exit_cfg.setdefault("reward_r", 2.0)
    lev = int(exit_cfg.get("leverage") or 2)
    if lev not in config.LEVERAGE_CANDIDATES:
        lev = config.LEVERAGE_CANDIDATES[0]
    exit_cfg["leverage"] = lev
    out["exit"] = exit_cfg
    out["mtf_filter"] = bool(out.get("mtf_filter", False))
    return out


def _validate_cond(cond: dict[str, Any]) -> tuple[bool, str]:
    ind = str(cond.get("ind") or "")
    params = cond.get("params") or {}
    if not isinstance(params, dict):
        return False, "params must be object"
    ok, msg = validate_indicator(ind, params if params else None)
    if not ok:
        return False, msg
    op = str(cond.get("op") or "")
    if op not in OPS:
        return False, f"bad op: {op}"
    vs = cond.get("vs")
    if isinstance(vs, dict) and vs.get("ind"):
        ok2, msg2 = validate_indicator(
            str(vs["ind"]), dict(vs.get("params") or {}) or None
        )
        if not ok2:
            return False, msg2
    return True, ""


def validate_spec(spec: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(spec, dict):
        return False, "spec must be object"
    s = normalize_spec(spec)
    for key in ("entry_long", "entry_short", "filters"):
        block = s.get(key) or []
        if not isinstance(block, list):
            return False, f"{key} must be list"
        for cond in block:
            if not isinstance(cond, dict):
                return False, f"{key} cond must be object"
            ok, msg = _validate_cond(cond)
            if not ok:
                return False, f"{key}: {msg}"
    if not s["entry_long"] and not s["entry_short"]:
        return False, "need entry_long or entry_short"
    return True, ""


def _series_val(row: pd.Series, ind: str, params: dict | None) -> float:
    key = col_key(ind, params if params else None)
    if key in row.index:
        v = row.get(key)
    elif ind in row.index:
        v = row.get(ind)
    else:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _eval_cond(row: pd.Series, prev: pd.Series, cond: dict[str, Any]) -> bool:
    ind = str(cond["ind"])
    params = dict(cond.get("params") or {})
    op = str(cond["op"])
    left = _series_val(row, ind, params if params else None)
    prev_left = _series_val(prev, ind, params if params else None)

    vs = cond.get("vs")
    if isinstance(vs, dict) and vs.get("ind"):
        right = _series_val(
            row, str(vs["ind"]), dict(vs.get("params") or {}) or None
        )
        prev_right = _series_val(
            prev, str(vs["ind"]), dict(vs.get("params") or {}) or None
        )
        if op == "cross_up":
            return (
                np.isfinite(left)
                and np.isfinite(right)
                and np.isfinite(prev_left)
                and np.isfinite(prev_right)
                and prev_left <= prev_right
                and left > right
            )
        if op == "cross_down":
            return (
                np.isfinite(left)
                and np.isfinite(right)
                and np.isfinite(prev_left)
                and np.isfinite(prev_right)
                and prev_left >= prev_right
                and left < right
            )
        if not np.isfinite(left) or not np.isfinite(right):
            return False
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        if op == "lte":
            return left <= right
        if op == "eq":
            return abs(left - right) < 1e-12
        return False

    if op == "is_true":
        return bool(left) if np.isfinite(left) else False
    if op == "is_false":
        return (not bool(left)) if np.isfinite(left) else False

    thr = cond.get("value")
    if op == "between":
        if not isinstance(thr, (list, tuple)) or len(thr) != 2:
            return False
        if not np.isfinite(left):
            return False
        return float(thr[0]) <= left <= float(thr[1])

    if thr is None or not np.isfinite(left):
        return False
    t = float(thr)
    if op == "lt":
        return left < t
    if op == "lte":
        return left <= t
    if op == "gt":
        return left > t
    if op == "gte":
        return left >= t
    if op == "eq":
        return abs(left - t) < 1e-12
    if op == "cross_up":
        # cross above threshold
        return np.isfinite(prev_left) and prev_left <= t < left
    if op == "cross_down":
        return np.isfinite(prev_left) and prev_left >= t > left
    return False


def _all_conds(row: pd.Series, prev: pd.Series, conds: list) -> bool:
    if not conds:
        return True
    return all(_eval_cond(row, prev, c) for c in conds if isinstance(c, dict))


def mtf_ok(row: pd.Series, side: int, enabled: bool) -> bool:
    if not enabled:
        return True
    biases: list[bool] = []
    for col in ("htf_1h_bull", "htf_4h_bull", "htf1_bull", "htf4_bull", "htf1d_bull"):
        if col in row.index and pd.notna(row.get(col)):
            biases.append(bool(row.get(col)))
    # also catalog keys
    for ind in ("htf_1h_bull", "htf_4h_bull"):
        k = col_key(ind, None)
        if k in row.index and pd.notna(row.get(k)):
            biases.append(bool(row.get(k)))
    if not biases:
        return False
    return all(biases) if side == 1 else all(not b for b in biases)


def compile_signal(spec: dict[str, Any]) -> SignalFn:
    s = normalize_spec(spec)
    ok, msg = validate_spec(s)
    if not ok:
        raise ValueError(msg)

    def signal_fn(row: pd.Series, prev: pd.Series) -> Optional[int]:
        if not _all_conds(row, prev, s.get("filters") or []):
            return None
        long_ok = bool(s["entry_long"]) and _all_conds(row, prev, s["entry_long"])
        short_ok = bool(s["entry_short"]) and _all_conds(row, prev, s["entry_short"])
        side = None
        if long_ok and not short_ok:
            side = 1
        elif short_ok and not long_ok:
            side = -1
        elif long_ok and short_ok:
            return None
        if side is None:
            return None
        if not mtf_ok(row, side, s["mtf_filter"]):
            return None
        return side

    return signal_fn


def exit_params(spec: dict[str, Any]) -> dict[str, Any]:
    s = normalize_spec(spec)
    return dict(s["exit"])


def diagnose_spec_entry(
    row: pd.Series,
    prev: pd.Series,
    spec: dict[str, Any],
) -> dict[str, Any]:
    s = normalize_spec(spec)
    try:
        fn = compile_signal(s)
    except ValueError as exc:
        return {"side": None, "action": "SKIP", "why": f"spec invalid: {exc}"}
    if not _all_conds(row, prev, s.get("filters") or []):
        return {"side": None, "action": "SKIP", "why": "filter chưa đạt"}
    side = fn(row, prev)
    if side is None:
        long_raw = bool(s["entry_long"]) and _all_conds(row, prev, s["entry_long"])
        short_raw = bool(s["entry_short"]) and _all_conds(row, prev, s["entry_short"])
        if long_raw or short_raw:
            return {
                "side": None,
                "action": "SKIP",
                "why": "signal bị chặn MTF hoặc xung đột L/S",
            }
        return {
            "side": None,
            "action": "SKIP",
            "why": f"spec={s.get('name')}: chưa đủ điều kiện setup",
        }
    return {
        "side": side,
        "action": "PENDING",
        "why": f"spec {s.get('name')} signal {'LONG' if side == 1 else 'SHORT'} → chờ fill",
    }


# --- Legacy family → Spec seeds -------------------------------------------------

def family_to_spec(family: str, **overrides: Any) -> dict[str, Any]:
    """Map old 11 families into Strategy Spec equivalents."""
    rsi_low = int(overrides.get("rsi_low", 30))
    rsi_high = int(overrides.get("rsi_high", 70))
    fast = int(overrides.get("fast", 12))
    slow = int(overrides.get("slow", 50))
    breakout_n = int(overrides.get("breakout_n", 20))
    vol_f = float(overrides.get("volume_filter", 0) or 0)
    max_atr = float(overrides.get("max_atr_pct", 0.06) or 0.06)
    atr_stop = float(overrides.get("atr_stop", 2.0))
    reward_r = float(overrides.get("reward_r", 2.0))
    leverage = int(overrides.get("leverage", 2))
    mtf = bool(overrides.get("mtf_filter", False))
    funding_abs = float(overrides.get("funding_abs_max", 0.001))
    oi_min = float(overrides.get("oi_change_min", 0.0))

    filters: list[dict] = [
        {"ind": "atr_pct", "params": {"period": 14}, "op": "lte", "value": max_atr},
    ]
    if vol_f > 0:
        filters.append(
            {"ind": "vol_ratio", "params": {"period": 20}, "op": "gte", "value": vol_f}
        )

    exit_cfg = {
        "atr_period": 14,
        "atr_stop": atr_stop,
        "reward_r": reward_r,
        "leverage": leverage,
    }

    def base(name: str, entry_long: list, entry_short: list) -> dict:
        return normalize_spec(
            {
                "name": name,
                "entry_long": entry_long,
                "entry_short": entry_short,
                "filters": filters,
                "exit": exit_cfg,
                "mtf_filter": mtf,
                "legacy_family": family,
            }
        )

    if family == "trend_cross":
        return base(
            "trend_cross",
            [
                {
                    "ind": "ema",
                    "params": {"period": fast},
                    "op": "cross_up",
                    "vs": {"ind": "ema", "params": {"period": slow}},
                }
            ],
            [
                {
                    "ind": "ema",
                    "params": {"period": fast},
                    "op": "cross_down",
                    "vs": {"ind": "ema", "params": {"period": slow}},
                }
            ],
        )
    if family == "breakout":
        return base(
            "breakout",
            [
                {
                    "ind": "close",
                    "params": {},
                    "op": "gt",
                    "vs": {"ind": "donchian_high", "params": {"period": breakout_n}},
                }
            ],
            [
                {
                    "ind": "close",
                    "params": {},
                    "op": "lt",
                    "vs": {"ind": "donchian_low", "params": {"period": breakout_n}},
                }
            ],
        )
    if family == "pullback":
        return base(
            "pullback",
            [
                {"ind": "dist_to_ema", "params": {"period": 200}, "op": "gt", "value": 0},
                {
                    "ind": "close",
                    "params": {},
                    "op": "cross_up",
                    "vs": {"ind": "ema", "params": {"period": fast}},
                },
                {
                    "ind": "rsi",
                    "params": {"period": 14},
                    "op": "between",
                    "value": [rsi_low, rsi_high],
                },
            ],
            [
                {"ind": "dist_to_ema", "params": {"period": 200}, "op": "lt", "value": 0},
                {
                    "ind": "close",
                    "params": {},
                    "op": "cross_down",
                    "vs": {"ind": "ema", "params": {"period": fast}},
                },
                {
                    "ind": "rsi",
                    "params": {"period": 14},
                    "op": "between",
                    "value": [100 - rsi_high, 100 - rsi_low],
                },
            ],
        )
    if family == "mean_revert":
        return base(
            "mean_revert",
            [
                {"ind": "bb_z", "params": {"period": 20, "std": 2.0}, "op": "lte", "value": -1.8},
                {"ind": "rsi", "params": {"period": 14}, "op": "lt", "value": rsi_low},
            ],
            [
                {"ind": "bb_z", "params": {"period": 20, "std": 2.0}, "op": "gte", "value": 1.8},
                {"ind": "rsi", "params": {"period": 14}, "op": "gt", "value": rsi_high},
            ],
        )
    if family == "momentum":
        return base(
            "momentum",
            [
                {"ind": "ret", "params": {"period": 5}, "op": "gt", "value": 0.012},
                {"ind": "ret", "params": {"period": 20}, "op": "gt", "value": 0.02},
            ],
            [
                {"ind": "ret", "params": {"period": 5}, "op": "lt", "value": -0.012},
                {"ind": "ret", "params": {"period": 20}, "op": "lt", "value": -0.02},
            ],
        )
    if family == "volatility_breakout":
        fl = [
            {"ind": "atr_pct", "params": {"period": 14}, "op": "gte", "value": 0.012},
        ] + filters
        return normalize_spec(
            {
                "name": "volatility_breakout",
                "entry_long": [
                    {
                        "ind": "close",
                        "params": {},
                        "op": "gt",
                        "vs": {"ind": "donchian_high", "params": {"period": breakout_n}},
                    }
                ],
                "entry_short": [
                    {
                        "ind": "close",
                        "params": {},
                        "op": "lt",
                        "vs": {"ind": "donchian_low", "params": {"period": breakout_n}},
                    }
                ],
                "filters": fl,
                "exit": exit_cfg,
                "mtf_filter": mtf,
                "legacy_family": family,
            }
        )
    if family == "vwap_reversion":
        return base(
            "vwap_reversion",
            [
                {"ind": "vwap_z", "params": {"window": 48}, "op": "lt", "value": -1.5},
                {"ind": "rsi", "params": {"period": 14}, "op": "lt", "value": rsi_low},
            ],
            [
                {"ind": "vwap_z", "params": {"window": 48}, "op": "gt", "value": 1.5},
                {"ind": "rsi", "params": {"period": 14}, "op": "gt", "value": rsi_high},
            ],
        )
    if family == "funding_contrarian":
        return base(
            "funding_contrarian",
            [
                {"ind": "funding_rate", "params": {}, "op": "lte", "value": -funding_abs},
                {"ind": "rsi", "params": {"period": 14}, "op": "lt", "value": 45},
            ],
            [
                {"ind": "funding_rate", "params": {}, "op": "gte", "value": funding_abs},
                {"ind": "rsi", "params": {"period": 14}, "op": "gt", "value": 55},
            ],
        )
    if family == "oi_breakout":
        return base(
            "oi_breakout",
            [
                {"ind": "oi_change", "params": {}, "op": "gte", "value": oi_min},
                {
                    "ind": "close",
                    "params": {},
                    "op": "gt",
                    "vs": {"ind": "donchian_high", "params": {"period": breakout_n}},
                },
            ],
            [
                {"ind": "oi_change", "params": {}, "op": "gte", "value": oi_min},
                {
                    "ind": "close",
                    "params": {},
                    "op": "lt",
                    "vs": {"ind": "donchian_low", "params": {"period": breakout_n}},
                },
            ],
        )
    if family == "range_compression":
        fl = [
            {"ind": "atr_pct", "params": {"period": 14}, "op": "lt", "value": 0.012},
        ]
        if vol_f > 0:
            fl.append(
                {"ind": "vol_ratio", "params": {"period": 20}, "op": "gte", "value": vol_f}
            )
        return normalize_spec(
            {
                "name": "range_compression",
                "entry_long": [
                    {
                        "ind": "close",
                        "params": {},
                        "op": "gt",
                        "vs": {"ind": "donchian_high", "params": {"period": breakout_n}},
                    }
                ],
                "entry_short": [
                    {
                        "ind": "close",
                        "params": {},
                        "op": "lt",
                        "vs": {"ind": "donchian_low", "params": {"period": breakout_n}},
                    }
                ],
                "filters": fl,
                "exit": exit_cfg,
                "mtf_filter": mtf,
                "legacy_family": family,
            }
        )
    if family == "orderflow_imbalance":
        return base(
            "orderflow_imbalance",
            [
                {"ind": "taker_buy_ratio", "params": {}, "op": "gt", "value": 0.58},
                {"ind": "vol_ratio", "params": {"period": 20}, "op": "gt", "value": 1.1},
            ],
            [
                {"ind": "taker_buy_ratio", "params": {}, "op": "lt", "value": 0.42},
                {"ind": "vol_ratio", "params": {"period": 20}, "op": "gt", "value": 1.1},
            ],
        )
    # default: RSI mean-ish
    return base(
        family or "rsi_default",
        [{"ind": "rsi", "params": {"period": 14}, "op": "lt", "value": rsi_low}],
        [{"ind": "rsi", "params": {"period": 14}, "op": "gt", "value": rsi_high}],
    )


def genome_dict_to_spec(g: dict[str, Any]) -> dict[str, Any]:
    family = str(g.get("family") or "trend_cross")
    return family_to_spec(family, **g)


def default_seed_specs() -> list[dict[str, Any]]:
    from modules.genome import FAMILIES

    return [family_to_spec(f) for f in FAMILIES]


def loosen_spec_for_starve(spec: dict[str, Any]) -> dict[str, Any]:
    """PnL-first: nới filter khi starve để có tín hiệu học."""
    s = normalize_spec(spec)
    new_filters = []
    for cond in s.get("filters") or []:
        if not isinstance(cond, dict):
            continue
        c = deepcopy(cond)
        ind = str(c.get("ind") or "")
        if ind == "vol_ratio" and c.get("value") is not None:
            c["value"] = max(0.0, float(c["value"]) - 0.3)
        elif ind == "atr_pct" and c.get("op") in {"lte", "lt"} and c.get("value"):
            c["value"] = min(0.08, float(c["value"]) * 1.5)
        new_filters.append(c)
    s["filters"] = new_filters
    s["mtf_filter"] = False
    s["name"] = str(s.get("name") or "spec") + "_loose"
    return s


def apply_patch(spec: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any] | None:
    """Apply one LLM patch: tune|add|remove on entry_long/short/filters."""
    s = normalize_spec(spec)
    action = str(patch.get("action") or "").lower()
    target = str(patch.get("target") or "filters")  # entry_long|entry_short|filters
    if target not in {"entry_long", "entry_short", "filters"}:
        return None
    block = list(s.get(target) or [])

    if action == "remove":
        idx = int(patch.get("index", -1))
        if idx < 0 or idx >= len(block):
            # remove by ind match
            ind = str(patch.get("ind") or "")
            block = [c for c in block if str(c.get("ind")) != ind]
        else:
            block.pop(idx)
    elif action == "add":
        cond = patch.get("condition")
        if not isinstance(cond, dict):
            return None
        ok, _ = _validate_cond(cond)
        if not ok:
            return None
        block.append(cond)
    elif action == "tune":
        idx = int(patch.get("index", 0))
        if idx < 0 or idx >= len(block):
            return None
        cond = deepcopy(block[idx])
        if "params" in patch and isinstance(patch["params"], dict):
            cond["params"] = {**(cond.get("params") or {}), **patch["params"]}
        if "value" in patch:
            cond["value"] = patch["value"]
        if "op" in patch:
            cond["op"] = patch["op"]
        ok, _ = _validate_cond(cond)
        if not ok:
            return None
        block[idx] = cond
    else:
        return None

    s[target] = block
    ok, msg = validate_spec(s)
    if not ok:
        logger.info("patch rejected: %s", msg)
        return None
    s["name"] = str(s.get("name") or "spec") + f"_p{action[:3]}"
    return s
