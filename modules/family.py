"""The bot's single indicator family.

The LLM drafts it (genesis), Python's walk-forward picks the best draft as v1,
and every later research round the LLM adds / removes / replaces / tunes
indicators; a change only becomes the next version if it wins out-of-sample
(or forward on paper as a shadow). The changelog is fed back to the LLM.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

import config
from modules.strategy_spec import _fmt_cond, base_name, format_exit, normalize_spec, spec_id

_BLOCKS = (("entry_long", "LONG"), ("entry_short", "SHORT"), ("filters", "FILTER"))


def get(state: dict) -> Optional[dict]:
    fam = state.get("family")
    return fam if isinstance(fam, dict) and fam.get("name") else None


def needs_genesis(state: dict) -> bool:
    return get(state) is None or not state.get("champion_spec")


def label(state: dict) -> Optional[str]:
    fam = get(state)
    return f"{fam['name']} v{fam['version']}" if fam else None


def pop_meta(spec: dict) -> tuple[dict, dict]:
    """Split the research note (source / reason / ops) from the trading logic."""
    s = normalize_spec(spec)
    meta = s.pop("variant", None)
    return s, meta if isinstance(meta, dict) else {}


def diff_specs(old: dict | None, new: dict) -> list[str]:
    if not old:
        return ["(phiên bản đầu)"]
    a, b = normalize_spec(old), normalize_spec(new)
    out: list[str] = []
    for key, tag in _BLOCKS:
        before = [_fmt_cond(c) for c in a.get(key) or [] if isinstance(c, dict)]
        after = [_fmt_cond(c) for c in b.get(key) or [] if isinstance(c, dict)]
        out += [f"- {tag}: {c}" for c in before if c not in after]
        out += [f"+ {tag}: {c}" for c in after if c not in before]
    if a["exit"] != b["exit"]:
        out.append(f"~ EXIT: {format_exit(a['exit'])} → {format_exit(b['exit'])}")
    if a.get("mtf_filter") != b.get("mtf_filter"):
        out.append(f"~ MTF: {a.get('mtf_filter')} → {b.get('mtf_filter')}")
    return out or ["(không đổi logic)"]


def _eval_digest(row: dict | None) -> dict | None:
    if not row:
        return None
    oos = row.get("oos") or row.get("test") or {}
    return {
        "robust": round(float(row.get("robust_score") or -999), 2),
        "qualified": "hard" if row.get("qualified") else ("soft" if row.get("soft_qualified") else "no"),
        "exp": round(float(oos.get("expectancy") or 0), 4),
        "pf": round(float(oos.get("profit_factor") or 0), 2),
        "n": int(oos.get("trade_count") or 0),
        "folds_pos": row.get("positive_folds"),
    }


def _entry(version: int, spec: dict, changes: list[str], meta: dict, source: str, row: dict | None) -> dict:
    return {
        "version": version,
        "spec_id": spec_id(spec),
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": source,
        "changes": changes,
        "reason": str(meta.get("reason") or "")[:400],
        "backtest": _eval_digest(row),
    }


def start(state: dict, spec: dict, *, meta: dict, row: dict | None) -> dict:
    s = normalize_spec(spec)
    fam = {
        "name": base_name(s.get("name")),
        "version": 1,
        "spec_id": spec_id(s),
        "created_ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "history": [_entry(1, s, diff_specs(None, s), meta, "genesis", row)],
    }
    state["family"] = fam
    return fam


def record_version(
    state: dict, old: dict | None, new: dict, *, meta: dict, source: str, row: dict | None
) -> dict:
    fam = state["family"]
    fam["version"] = int(fam.get("version") or 0) + 1
    fam["spec_id"] = spec_id(new)
    hist = list(fam.get("history") or [])
    hist.append(_entry(fam["version"], new, diff_specs(old, new), meta, source, row))
    fam["history"] = hist[-config.FAMILY_HISTORY_KEEP :]
    return fam


def history_for_prompt(state: dict, limit: int = 10) -> list[dict[str, Any]]:
    """Version changelog with backtest + real paper results, so the LLM sees which edits helped."""
    fam = get(state)
    if not fam:
        return []
    from modules.research_memory import paper_results_by_spec

    paper = {p["id"]: p for p in paper_results_by_spec()}
    out = []
    for h in (fam.get("history") or [])[-limit:]:
        p = paper.get(h.get("spec_id"))
        out.append(
            {
                **{k: h.get(k) for k in ("version", "source", "changes", "reason", "backtest")},
                "paper": (
                    {k: p[k] for k in ("n", "pnl", "win_rate", "expectancy")} if p else {"n": 0}
                ),
            }
        )
    return out
