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


_VARIANT_TOKENS = {"m", "loose", "patch", "ptun", "padd", "prem"}
_ID_EXCLUDE_KEYS = {"name", "legacy_family", "variant"}


def spec_id(spec: dict[str, Any]) -> str:
    """Hash of trading logic only — renaming a Spec does not change its id."""
    logic = {
        k: v for k, v in normalize_spec(spec).items() if k not in _ID_EXCLUDE_KEYS
    }
    raw = json.dumps(logic, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def base_name(name: str | None) -> str:
    tokens = str(name or "spec").split("_")
    while len(tokens) > 1 and tokens[-1] in _VARIANT_TOKENS:
        tokens.pop()
    return "_".join(tokens) or "spec"


def variant_name(name: str | None, variant: str) -> str:
    return f"{base_name(name)}_{variant}"


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
    lines.append("EXIT: " + format_exit(s.get("exit") or {}))
    return "\n".join(lines)


def format_exit(ex: dict[str, Any]) -> str:
    parts = [f"atr×{ex.get('atr_stop')} RR={ex.get('reward_r')} lev={ex.get('leverage')}x"]
    if ex.get("breakeven_r"):
        parts.append(f"BE@{ex['breakeven_r']}R")
    if ex.get("trail_atr"):
        parts.append(f"trail {ex['trail_atr']}×ATR")
    if ex.get("max_bars"):
        parts.append(f"max {ex['max_bars']} nến")
    if ex.get("exit_on_opposite"):
        parts.append("đóng khi tín hiệu ngược")
    return " | ".join(parts)


def spec_requirements(spec: dict[str, Any]) -> list[tuple[str, dict]]:
    from modules.features_catalog import collect_requirements

    s = normalize_spec(spec)
    conds = [
        c
        for key in ("entry_long", "entry_short", "filters")
        for c in (s.get(key) or [])
        if isinstance(c, dict)
    ]
    return collect_requirements(conds)


def short_conditions(spec: dict[str, Any], max_len: int = 220) -> str:
    s = normalize_spec(spec)
    parts = []
    for key, tag in (("entry_long", "L"), ("entry_short", "S"), ("filters", "F")):
        conds = [c for c in (s.get(key) or []) if isinstance(c, dict)]
        if conds:
            parts.append(f"{tag}: " + " & ".join(_fmt_cond(c) for c in conds))
    text = " | ".join(parts) + f" | X: {format_exit(s.get('exit') or {})}"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


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
    # Optional exits are omitted when off so older Specs keep their spec_id
    for key, cast in (("trail_atr", float), ("breakeven_r", float), ("max_bars", int)):
        try:
            val = cast(exit_cfg.get(key) or 0)
        except (TypeError, ValueError):
            val = 0
        if val > 0:
            exit_cfg[key] = val
        else:
            exit_cfg.pop(key, None)
    if exit_cfg.get("exit_on_opposite"):
        exit_cfg["exit_on_opposite"] = True
    else:
        exit_cfg.pop("exit_on_opposite", None)
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


def _fmt_num(v: float, ind: str = "") -> str:
    if not np.isfinite(v):
        return "?"
    if ind == "atr_pct":
        return f"{v * 100:.2f}%"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _cond_live(row: pd.Series, prev: pd.Series, cond: dict[str, Any]) -> str:
    """One condition with live value, required rule, and ✓/✗."""
    ok = "✓" if _eval_cond(row, prev, cond) else "✗"
    ind = str(cond.get("ind") or "?")
    params = dict(cond.get("params") or {})
    op = str(cond.get("op") or "")
    now = _series_val(row, ind, params if params else None)
    was = _series_val(prev, ind, params if params else None)
    vs = cond.get("vs")
    if isinstance(vs, dict) and vs.get("ind"):
        right = _series_val(row, str(vs["ind"]), dict(vs.get("params") or {}) or None)
        live = f"{ind}={_fmt_num(now, ind)} vs {vs['ind']}={_fmt_num(right, str(vs['ind']))}"
        if op in {"cross_up", "cross_down"}:
            live += f" ({_fmt_num(was, ind)}→{_fmt_num(now, ind)})"
        return f"{ok} {live} cần {op}"
    thr = cond.get("value")
    if op in {"cross_up", "cross_down"}:
        return (
            f"{ok} {ind}={_fmt_num(was, ind)}→{_fmt_num(now, ind)} cần {op} "
            f"{_fmt_num(float(thr), ind) if thr is not None else '?'}"
        )
    if op == "between" and isinstance(thr, (list, tuple)) and len(thr) == 2:
        return f"{ok} {ind}={_fmt_num(now, ind)} cần {thr[0]}–{thr[1]}"
    if op in {"is_true", "is_false"}:
        return f"{ok} {ind}={_fmt_num(now, ind)} cần {op}"
    need = _fmt_num(float(thr), ind) if thr is not None else "?"
    return f"{ok} {ind}={_fmt_num(now, ind)} cần {op} {need}"


def _block_live(row: pd.Series, prev: pd.Series, conds: list, tag: str) -> str:
    parts = [_cond_live(row, prev, c) for c in (conds or []) if isinstance(c, dict)]
    return f"{tag}: " + " | ".join(parts) if parts else f"{tag}: (none)"


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
    live = "\n".join(
        [
            _block_live(row, prev, s.get("entry_long") or [], "LONG"),
            _block_live(row, prev, s.get("entry_short") or [], "SHORT"),
            _block_live(row, prev, s.get("filters") or [], "FILTER"),
        ]
    )
    if not _all_conds(row, prev, s.get("filters") or []):
        return {"side": None, "action": "SKIP", "why": "filter chưa đạt\n" + live}
    side = fn(row, prev)
    if side is None:
        long_raw = bool(s["entry_long"]) and _all_conds(row, prev, s["entry_long"])
        short_raw = bool(s["entry_short"]) and _all_conds(row, prev, s["entry_short"])
        if long_raw or short_raw:
            extra = " (MTF chặn)" if s.get("mtf_filter") else " (LONG và SHORT cùng lúc)"
            return {
                "side": None,
                "action": "SKIP",
                "why": "signal bị chặn MTF hoặc xung đột L/S" + extra + "\n" + live,
            }
        return {
            "side": None,
            "action": "SKIP",
            "why": "chưa đủ điều kiện setup\n" + live,
        }
    return {
        "side": side,
        "action": "PENDING",
        "why": f"{'LONG' if side == 1 else 'SHORT'} đủ điều kiện → chờ fill nến sau\n" + live,
    }


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
    s["name"] = variant_name(s.get("name"), "loose")
    return s


_PRICE_INDS = {"close"}
_IND_TYPE = {
    "donchian_high": "donchian",
    "donchian_low": "donchian",
    "htf_1h_bull": "htf_trend",
    "htf_4h_bull": "htf_trend",
    "ema_cross": "ema",
}
_EXIT_KEYS = {"atr_stop", "reward_r", "leverage", "breakeven_r", "trail_atr", "max_bars", "exit_on_opposite"}


def indicator_types(spec: dict[str, Any]) -> list[str]:
    """Distinct indicator types a Spec uses (ema12 + ema50 = ema; raw price not counted)."""
    s = normalize_spec(spec)
    names: set[str] = set()
    for key in ("entry_long", "entry_short", "filters"):
        for cond in s.get(key) or []:
            if not isinstance(cond, dict):
                continue
            names.add(str(cond.get("ind") or ""))
            vs = cond.get("vs")
            if isinstance(vs, dict) and vs.get("ind"):
                names.add(str(vs["ind"]))
    if s.get("mtf_filter"):
        names.add("htf_trend")
    return sorted({_IND_TYPE.get(n, n) for n in names if n and n not in _PRICE_INDS})


def family_shape_ok(spec: dict[str, Any]) -> tuple[bool, str]:
    ok, msg = validate_spec(spec)
    if not ok:
        return False, msg
    n = len(indicator_types(spec))
    if not config.FAMILY_MIN_INDICATORS <= n <= config.FAMILY_MAX_INDICATORS:
        return False, (
            f"{n} chỉ báo, cần {config.FAMILY_MIN_INDICATORS}-{config.FAMILY_MAX_INDICATORS}"
        )
    return True, ""


def _index(patch: dict[str, Any], block: list) -> int | None:
    try:
        idx = int(patch.get("index"))
    except (TypeError, ValueError):
        idx = None
    if idx is not None and 0 <= idx < len(block):
        return idx
    ind = str(patch.get("ind") or "")
    return next((i for i, c in enumerate(block) if str(c.get("ind")) == ind), None) if ind else None


def apply_patch(spec: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any] | None:
    """One edit: add|remove|replace|tune a condition, or set exit params / mtf."""
    s = normalize_spec(spec)
    action = str(patch.get("action") or "").lower()

    if action == "exit":
        changes = patch.get("exit") if isinstance(patch.get("exit"), dict) else {}
        changes = {k: v for k, v in changes.items() if k in _EXIT_KEYS}
        if not changes:
            return None
        s["exit"] = {**s["exit"], **changes}
        return normalize_spec(s)
    if action == "mtf":
        s["mtf_filter"] = bool(patch.get("value"))
        return s

    target = str(patch.get("target") or "filters")
    if target not in {"entry_long", "entry_short", "filters"}:
        return None
    block = list(s.get(target) or [])
    idx = _index(patch, block)

    if action == "remove":
        if idx is None:
            return None
        block.pop(idx)
    elif action in {"add", "replace"}:
        cond = patch.get("condition")
        if not isinstance(cond, dict) or not _validate_cond(cond)[0]:
            return None
        if action == "add":
            block.append(cond)
        elif idx is None:
            return None
        else:
            block[idx] = cond
    elif action == "tune":
        if idx is None:
            return None
        cond = deepcopy(block[idx])
        if isinstance(patch.get("params"), dict):
            cond["params"] = {**(cond.get("params") or {}), **patch["params"]}
        if "value" in patch:
            cond["value"] = patch["value"]
        if "op" in patch:
            cond["op"] = patch["op"]
        if not _validate_cond(cond)[0]:
            return None
        block[idx] = cond
    else:
        return None

    s[target] = block
    ok, msg = validate_spec(s)
    if not ok:
        logger.info("patch rejected: %s", msg)
        return None
    return s


def apply_ops(spec: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Apply a sequence of edits; the result must still be a valid 3–5 indicator family."""
    s = normalize_spec(spec)
    for op in ops:
        if not isinstance(op, dict):
            return None
        nxt = apply_patch(s, op)
        if nxt is None:
            logger.info("family op rejected: %s", json.dumps(op, default=str)[:200])
            return None
        s = nxt
    ok, msg = family_shape_ok(s)
    if not ok:
        logger.info("family variant rejected: %s", msg)
        return None
    return s
