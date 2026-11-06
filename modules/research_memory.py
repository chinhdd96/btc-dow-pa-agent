"""What the agent already tried and how it went — fed back into LLM research prompts."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import config
from modules import champion_store
from modules.strategy_spec import normalize_spec, short_conditions, spec_requirements

logger = logging.getLogger(__name__)


def digest_row(r: dict) -> dict[str, Any]:
    """Compact record of one evaluated Spec (stored in each research report)."""
    spec = r.get("spec") or {}
    oos = r.get("oos") or r.get("test") or {}
    try:
        inds = sorted({ind for ind, _ in spec_requirements(spec)})
        cond = short_conditions(spec, max_len=200)
    except Exception:  # noqa: BLE001
        inds, cond = [], ""
    return {
        "id": r.get("spec_id"),
        "name": r.get("name") or spec.get("name"),
        "cond": cond,
        "inds": inds,
        "robust": round(float(r.get("robust_score") or -999), 2),
        "q": bool(r.get("qualified")),
        "sq": bool(r.get("soft_qualified")),
        "exp": round(float(oos.get("expectancy") or 0), 4),
        "pf": round(float(oos.get("profit_factor") or 0), 2),
        "n": int(oos.get("trade_count") or 0),
        "folds_pos": r.get("positive_folds"),
    }


def _rows_from_run(run: dict) -> list[dict]:
    rows = run.get("evaluated_digest")
    if isinstance(rows, list) and rows:
        return [x for x in rows if isinstance(x, dict)]
    out = []
    for c in run.get("top_candidates") or []:
        if isinstance(c, dict) and c.get("spec"):
            try:
                out.append(digest_row({**c, "spec": normalize_spec(c["spec"])}))
            except Exception:  # noqa: BLE001
                continue
    return out


def paper_results_by_spec(limit: int = 300) -> list[dict[str, Any]]:
    trades = champion_store.list_paper_trades(limit=limit)
    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        sid = str(t.get("spec_id") or t.get("genome_id") or "?")
        a = agg[sid]
        a["n"] += 1
        a["pnl"] += float(t.get("pnl") or 0)
        a["wins"] += 1 if float(t.get("pnl") or 0) > 0 else 0
        a["name"] = t.get("family") or a.get("name")
    out = []
    for sid, a in agg.items():
        out.append(
            {
                "id": sid,
                "name": a.get("name"),
                "n": a["n"],
                "pnl": round(a["pnl"], 4),
                "win_rate": round(a["wins"] / a["n"] * 100, 1) if a["n"] else 0,
                "expectancy": round(a["pnl"] / a["n"], 4) if a["n"] else None,
            }
        )
    out.sort(key=lambda z: z["n"], reverse=True)
    return out


def summarize(max_chars: int = 1500, runs: int = 10) -> str:
    try:
        reports = champion_store.list_research_runs(limit=runs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_memory read failed: %s", exc)
        reports = []
    latest: dict[str, dict] = {}
    for run in reversed(reports):  # oldest → newest so newest wins
        for row in _rows_from_run(run):
            if row.get("id"):
                latest[row["id"]] = row
    if not latest:
        cache = champion_store.get_paper_state().get("research_memory_cache")
        return str(cache or "(chưa có lịch sử research)")

    rows = list(latest.values())
    rows.sort(key=lambda z: (z["q"], z["sq"], z["robust"]), reverse=True)
    good = [r for r in rows if r["q"] or r["sq"]]
    bad = sorted(
        (r for r in rows if not (r["q"] or r["sq"]) and r["n"] >= 5),
        key=lambda z: z["exp"],
    )
    silent = [r for r in rows if r["n"] < 5]

    ind_good: Counter = Counter()
    ind_all: Counter = Counter()
    for r in rows:
        for ind in r.get("inds") or []:
            ind_all[ind] += 1
            if r["q"] or r["sq"]:
                ind_good[ind] += 1
    ind_line = ", ".join(
        f"{ind} {ind_good[ind]}/{n}"
        for ind, n in sorted(ind_all.items(), key=lambda kv: -(ind_good[kv[0]] / kv[1]))
        if n >= 3
    )

    def fmt(r: dict) -> str:
        return (
            f"- {r['name']} [{r['id']}] q={'H' if r['q'] else ('S' if r['sq'] else '-')} "
            f"exp={r['exp']} pf={r['pf']} n={r['n']} folds+={r.get('folds_pos')} :: {r['cond']}"
        )

    lines = [f"Đã đánh giá {len(rows)} Spec trong {len(reports)} lần research."]
    if good:
        lines.append("TỐT NHẤT (out-of-sample):")
        lines += [fmt(r) for r in good[:5]]
    if bad:
        lines.append("TỆ NHẤT — có lệnh nhưng lỗ (đừng lặp lại):")
        lines += [fmt(r) for r in bad[:5]]
    if silent:
        lines.append(
            f"{len(silent)} Spec gần như không vào lệnh (<5 lệnh OOS), ví dụ: "
            + ", ".join(sorted({str(r['name']) for r in silent})[:6])
            + " → điều kiện quá chặt"
        )
    if ind_line:
        lines.append("Chỉ báo — số Spec đạt chuẩn/tổng: " + ind_line)
    paper = paper_results_by_spec()
    if paper:
        lines.append("KẾT QUẢ PAPER THẬT theo Spec:")
        lines += [
            f"- {p['name']} [{p['id']}] n={p['n']} pnl={p['pnl']} win={p['win_rate']}% exp={p['expectancy']}"
            for p in paper[:5]
        ]
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def recent_trades_for_prompt(limit: int | None = None) -> list[dict[str, Any]]:
    """Recent closed paper trades with the indicator snapshot taken at signal time."""
    trades = champion_store.list_paper_trades(limit=limit or config.LLM_RECENT_TRADES)
    out = []
    for t in reversed(trades):
        ctx = t.get("entry_context") or {}
        out.append(
            {
                "spec": t.get("family"),
                "side": "LONG" if t.get("side") == 1 else "SHORT",
                "entry_time": t.get("entry_time"),
                "exit_reason": t.get("exit_reason"),
                "pnl": round(float(t.get("pnl") or 0), 4),
                "r": round(float(t["r_multiple"]), 2) if t.get("r_multiple") is not None else None,
                "bars_held": t.get("bars_held"),
                "regime": (ctx.get("regime") or {}).get("tag"),
                "indicators": ctx.get("indicators"),
            }
        )
    return out
