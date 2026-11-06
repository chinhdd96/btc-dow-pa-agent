"""Shadow Specs: forward-test challengers on the same live candles, no real equity.

Shadows live in paper_state["shadows"] (so they are backed up with the Champion)
and survive across research runs until retired or promoted.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

import config
from modules import champion_store
from modules.strategy_spec import base_name, normalize_spec, spec_id

logger = logging.getLogger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def new_shadow(
    spec: dict,
    eval_row: Optional[dict] = None,
    *,
    label: str | None = None,
    champion: dict | None = None,
) -> dict[str, Any]:
    from modules.family import diff_specs

    s = normalize_spec(spec)
    s["name"] = base_name(s.get("name"))
    sid = spec_id(s)
    return {
        "champion_spec": s,
        "champion_spec_id": sid,
        "name": label or f"ứng viên {sid[:6]}",
        "changes": diff_specs(champion, s)[:4] if champion else [],
        "equity": config.INITIAL_CAPITAL,
        "position": None,
        "pending": None,
        "last_candle": None,
        "trade_count": 0,
        "trades": [],
        "added_ts": _now().isoformat(),
        "timeframe": config.TIMEFRAME_MAIN,
        "backtest_ok": bool(eval_row and (eval_row.get("qualified") or eval_row.get("soft_qualified"))),
        "robust_at_add": float(eval_row["robust_score"]) if eval_row else None,
    }


def shadows(state: dict) -> list[dict]:
    items = state.get("shadows")
    if not isinstance(items, list):
        items = []
        state["shadows"] = items
    return items


def run_on_slice(state: dict, df: pd.DataFrame) -> None:
    from modules.paper_champion import PaperChampion

    for sh in shadows(state):
        try:
            PaperChampion(state=sh).process(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shadow %s process failed: %s", sh.get("name"), exc)


def stats(trades: list[dict]) -> dict[str, Any]:
    pnls = [float(t.get("pnl") or 0) for t in trades]
    n = len(pnls)
    return {
        "n": n,
        "pnl": round(sum(pnls), 4),
        "expectancy": round(sum(pnls) / n, 4) if n else None,
        "win_rate": round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n else None,
    }


def champion_stats_since(champion_id: str | None, since_iso: str) -> dict[str, Any]:
    if not champion_id:
        return stats([])
    since = pd.Timestamp(since_iso)
    trades = [
        t
        for t in champion_store.list_paper_trades(limit=300)
        if str(t.get("spec_id") or t.get("genome_id")) == champion_id
        and t.get("exit_time")
        and pd.Timestamp(t["exit_time"]) >= since
    ]
    return stats(trades)


def retire(state: dict) -> list[str]:
    keep, gone = [], []
    now = _now()
    for sh in shadows(state):
        st = stats(sh.get("trades") or [])
        age_days = (now - pd.Timestamp(sh["added_ts"])).total_seconds() / 86400
        losing = st["n"] >= config.SHADOW_MIN_TRADES and (st["expectancy"] or 0) <= 0
        stale = age_days >= config.SHADOW_MAX_DAYS and st["n"] < config.SHADOW_MIN_TRADES
        if (losing or stale) and sh.get("position") is None:
            gone.append(f"{sh.get('name')} ({'lỗ' if losing else 'quá ít lệnh'}, n={st['n']})")
        else:
            keep.append(sh)
    state["shadows"] = keep
    return gone


def refill(state: dict, ranked: list[dict], champion: dict | None) -> list[str]:
    """Fill free slots with the best qualified family variants not already running."""
    champion_id = spec_id(champion) if champion else None
    existing = {sh.get("champion_spec_id") for sh in shadows(state)}
    added = []
    for r in ranked:
        if len(shadows(state)) >= config.SHADOW_SLOTS:
            break
        if not (r.get("qualified") or r.get("soft_qualified")):
            continue
        sid = r["spec_id"]
        if sid == champion_id or sid in existing:
            continue
        sh = new_shadow(r["spec"], r, champion=champion)
        state["shadows"].append(sh)
        existing.add(sid)
        added.append(f"{sh['name']}: {'; '.join(sh['changes'][:2])}")
    return added


def add_spec(
    state: dict,
    spec: dict,
    eval_row: Optional[dict] = None,
    *,
    label: str | None = None,
    champion: dict | None = None,
) -> bool:
    sid = spec_id(spec)
    if any(sh.get("champion_spec_id") == sid for sh in shadows(state)):
        return False
    if len(shadows(state)) >= config.SHADOW_SLOTS:
        return False
    state["shadows"].append(new_shadow(spec, eval_row, label=label, champion=champion))
    return True


def ensure(state: dict, row: dict, *, champion: dict | None = None) -> bool:
    """Make room for a candidate that must prove itself forward (evicts the weakest flat shadow)."""
    sid = row["spec_id"]
    if any(sh.get("champion_spec_id") == sid for sh in shadows(state)):
        return False
    if len(shadows(state)) >= config.SHADOW_SLOTS:
        flat = [sh for sh in shadows(state) if sh.get("position") is None and not sh.get("pending")]
        if not flat:
            return False
        weakest = min(flat, key=lambda sh: stats(sh.get("trades") or [])["expectancy"] or 0.0)
        remove(state, weakest["champion_spec_id"])
    state["shadows"].append(new_shadow(row["spec"], row, champion=champion))
    return True


def best_challenger(state: dict) -> Optional[dict]:
    """Shadow with enough forward trades that beats the Champion over the same window."""
    champ_id = state.get("champion_spec_id")
    best, best_exp = None, None
    for sh in shadows(state):
        st = stats(sh.get("trades") or [])
        if st["n"] < config.SHADOW_MIN_TRADES or (st["expectancy"] or 0) <= 0:
            continue
        if not sh.get("backtest_ok"):
            continue
        champ = champion_stats_since(champ_id, sh["added_ts"])
        champ_exp = champ["expectancy"] if champ["n"] else 0.0
        if st["expectancy"] <= champ_exp:
            continue
        if best_exp is None or st["expectancy"] > best_exp:
            best, best_exp = sh, st["expectancy"]
    return best


def remove(state: dict, sid: str) -> None:
    state["shadows"] = [sh for sh in shadows(state) if sh.get("champion_spec_id") != sid]


def summary_lines(state: dict) -> list[str]:
    out = []
    for sh in shadows(state):
        st = stats(sh.get("trades") or [])
        pos = "pos" if sh.get("position") else ("pend" if sh.get("pending") else "flat")
        out.append(
            f"- {sh.get('name')} [{sh.get('champion_spec_id')}] n={st['n']} "
            f"pnl={st['pnl']:+.2f} exp={st['expectancy']} win={st['win_rate']}% {pos}"
        )
        if sh.get("changes"):
            out.append("    " + "; ".join(sh["changes"][:2]))
    return out
