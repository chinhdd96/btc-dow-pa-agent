"""Live Binance USD-M public market data for Champion research/paper."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

FAPI_BASES = (
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
)
SPOT_DATA_BASE = "https://data-api.binance.vision"
USER_AGENT = "BTC-Champion-Research-Agent/1.0"

_session = requests.Session()
_preferred_fapi: str | None = None
_use_spot_klines = False


_thread_sessions = threading.local()


def _session_get(url: str, params: dict | None = None, timeout: int = 25) -> requests.Response:
    # requests.Session is not thread-safe; research runs in a background thread
    sess = getattr(_thread_sessions, "s", None)
    if sess is None:
        sess = _session if threading.current_thread() is threading.main_thread() else requests.Session()
        _thread_sessions.s = sess
    sess.verify = config.BINANCE_SSL_VERIFY
    sess.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return sess.get(url, params=params, timeout=timeout)


def fetch_recent_1m(
    symbol: str | None = None,
    *,
    start_ms: int | None = None,
    limit: int = 2,
) -> list[dict]:
    """Latest 1m klines incl. the still-forming one (live SL/TP watch)."""
    params: dict = {"symbol": symbol or config.SYMBOL, "interval": "1m", "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    rows = _get("/fapi/v1/klines", params, timeout=10) or []
    return [
        {
            "open_time_ms": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "close_time_ms": int(r[6]),
        }
        for r in rows
    ]


def _spot_klines(params: dict) -> object:
    resp = _session_get(
        f"{SPOT_DATA_BASE}/api/v3/klines",
        {
            "symbol": params.get("symbol", config.SYMBOL),
            "interval": params.get("interval", "1h"),
            "limit": params.get("limit", 1500),
            **({"startTime": params["startTime"]} if "startTime" in params else {}),
            **({"endTime": params["endTime"]} if "endTime" in params else {}),
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: dict | None = None, timeout: int = 25) -> object:
    """Public GET with multi-host Futures retry; klines may fall back to Spot."""
    global _preferred_fapi, _use_spot_klines
    params = params or {}
    if _use_spot_klines and path == "/fapi/v1/klines":
        return _spot_klines(params)

    errors: list[str] = []
    bases = list(FAPI_BASES)
    if _preferred_fapi:
        bases = [_preferred_fapi] + [b for b in bases if b != _preferred_fapi]
    for base in bases:
        try:
            resp = _session_get(f"{base}{path}", params, timeout=timeout)
            if resp.status_code == 403:
                errors.append(f"{base}: 403")
                continue
            resp.raise_for_status()
            _preferred_fapi = base
            return resp.json()
        except requests.RequestException as exc:
            errors.append(f"{base}: {exc}")
            continue

    if path == "/fapi/v1/klines":
        logger.warning(
            "Futures klines blocked (%s) — Spot data-api fallback",
            "; ".join(errors[-3:]) or "unknown",
        )
        _use_spot_klines = True
        return _spot_klines(params)

    raise RuntimeError("Binance public API failed: " + "; ".join(errors[-4:]))


def fetch_klines(
    interval: str,
    days: int,
    symbol: str | None = None,
) -> pd.DataFrame:
    symbol = symbol or config.SYMBOL
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows: list = []
    # Spot data-api max 1000/page; Futures allows 1500
    page_limit = 1000 if _use_spot_klines else 1500
    while start_ms < end_ms:
        batch = _get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": page_limit,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 1
        if nxt <= start_ms:
            break
        start_ms = nxt
        page_limit = 1000 if _use_spot_klines else 1500
        if len(batch) < page_limit:
            break
        time.sleep(0.10)
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise RuntimeError("Binance returned no candles")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df.open_time), unit="ms", utc=True)
    df["close_time"] = pd.to_numeric(df.close_time)
    now_ms = int(time.time() * 1000)
    df = df[df.close_time < now_ms].drop_duplicates("open_time").sort_values("open_time")
    df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df["taker_buy_quote"] = pd.to_numeric(df["taker_buy_quote"], errors="coerce")
    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_quote",
        ]
    ].reset_index(drop=True)


def fetch_market_snapshot(symbol: str | None = None) -> dict:
    symbol = symbol or config.SYMBOL
    out: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
    }
    try:
        premium = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
        out.update(
            {
                "mark_price": float(premium["markPrice"]),
                "index_price": float(premium.get("indexPrice") or premium["markPrice"]),
                "funding_rate": float(premium.get("lastFundingRate") or 0),
                "next_funding_time": int(premium.get("nextFundingTime") or 0),
            }
        )
    except Exception as exc:
        logger.warning("premiumIndex unavailable: %s", exc)
        try:
            resp = _session_get(
                f"{SPOT_DATA_BASE}/api/v3/ticker/price",
                {"symbol": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            px = float(resp.json()["price"])
            out["mark_price"] = px
            out["index_price"] = px
        except Exception as exc2:
            logger.warning("spot price fallback failed: %s", exc2)
    try:
        oi = _get("/fapi/v1/openInterest", {"symbol": symbol})
        out["open_interest"] = float(oi["openInterest"])
    except Exception as exc:
        logger.warning("openInterest unavailable: %s", exc)
    try:
        ticker = _get("/fapi/v1/ticker/24hr", {"symbol": symbol})
        out["change_24h_pct"] = float(ticker["priceChangePercent"])
        out["volume_24h"] = float(ticker["volume"])
    except Exception as exc:
        logger.warning("24hr ticker unavailable: %s", exc)
    return out


def fetch_funding_history(
    symbol: str | None = None,
    days: int = 730,
    limit: int = 1000,
) -> pd.DataFrame:
    symbol = symbol or config.SYMBOL
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows: list = []
    while start_ms < end_ms:
        batch = _get(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= start_ms or len(batch) < limit:
            break
        start_ms = nxt
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_event_rate"])
    out = pd.DataFrame(rows)
    out["funding_time"] = pd.to_datetime(
        pd.to_numeric(out["fundingTime"]), unit="ms", utc=True
    )
    out["funding_event_rate"] = pd.to_numeric(out["fundingRate"], errors="coerce")
    return (
        out[["funding_time", "funding_event_rate"]]
        .drop_duplicates("funding_time")
        .sort_values("funding_time")
    )


def fetch_open_interest_history(
    symbol: str | None = None,
    period: str = "1h",
) -> pd.DataFrame:
    symbol = symbol or config.SYMBOL
    try:
        rows = _get(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": 500},
        )
    except Exception as exc:
        logger.warning("OI history unavailable: %s", exc)
        return pd.DataFrame(columns=["open_time", "oi_change"])
    if not rows:
        return pd.DataFrame(columns=["open_time", "oi_change"])
    out = pd.DataFrame(rows)
    out["open_time"] = pd.to_datetime(
        pd.to_numeric(out["timestamp"]), unit="ms", utc=True
    )
    out["sum_open_interest"] = pd.to_numeric(out["sumOpenInterest"], errors="coerce")
    out["oi_change"] = out["sum_open_interest"].pct_change(3)
    return out[["open_time", "oi_change"]].dropna().sort_values("open_time")


def enrich_derivatives_data(
    candles: pd.DataFrame,
    symbol: str | None = None,
    days: int = 730,
) -> pd.DataFrame:
    symbol = symbol or config.SYMBOL
    x = candles.sort_values("open_time").copy()
    try:
        funding = fetch_funding_history(symbol, days)
        if not funding.empty:
            x = x.merge(
                funding, left_on="open_time", right_on="funding_time", how="left"
            )
            x = x.drop(columns=["funding_time"])
            funding_curve = funding.rename(
                columns={
                    "funding_time": "open_time",
                    "funding_event_rate": "funding_rate",
                }
            )
            x = pd.merge_asof(
                x.sort_values("open_time"),
                funding_curve.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            x["funding_event_rate"] = np.nan
            x["funding_rate"] = np.nan
    except Exception as exc:
        logger.warning("Funding history unavailable: %s", exc)
        x["funding_event_rate"] = np.nan
        x["funding_rate"] = np.nan
    try:
        oi = fetch_open_interest_history(symbol)
        if not oi.empty:
            x = pd.merge_asof(
                x.sort_values("open_time"),
                oi.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            x["oi_change"] = np.nan
    except Exception as exc:
        logger.warning("OI enrichment skipped: %s", exc)
        x["oi_change"] = np.nan
    return x.reset_index(drop=True)


_INTERVAL_TD = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def _htf_enrich_stack(main_interval: str | None = None) -> list[tuple[str, pd.Timedelta, str]]:
    """Higher-TF bias columns attached to the main trading interval."""
    main = (main_interval or config.TIMEFRAME_MAIN).strip()
    stack: list[tuple[str, pd.Timedelta, str]] = []
    # Sub-hour entries (e.g. 15m): also attach 1h mid-TF
    if main in {"1m", "3m", "5m", "15m", "30m"}:
        mid = getattr(config, "TIMEFRAME_MID", "1h") or "1h"
        if mid != main:
            stack.append((mid, _INTERVAL_TD.get(mid, pd.Timedelta(hours=1)), "htf1"))
    trend = getattr(config, "TIMEFRAME_TREND", "4h") or "4h"
    htf = getattr(config, "TIMEFRAME_HTF", "1d") or "1d"
    if trend != main:
        stack.append((trend, _INTERVAL_TD.get(trend, pd.Timedelta(hours=4)), "htf4"))
    if htf != main:
        stack.append((htf, _INTERVAL_TD.get(htf, pd.Timedelta(days=1)), "htf1d"))
    seen: set[str] = set()
    out: list[tuple[str, pd.Timedelta, str]] = []
    for item in stack:
        if item[0] in seen:
            continue
        seen.add(item[0])
        out.append(item)
    return out


def enrich_multi_timeframe(
    candles: pd.DataFrame,
    symbol: str | None = None,
    days: int = 730,
) -> pd.DataFrame:
    symbol = symbol or config.SYMBOL
    x = candles.sort_values("open_time").copy()
    for interval, duration, prefix in _htf_enrich_stack():
        try:
            higher = fetch_klines(interval, days, symbol)
            higher["fast"] = higher.close.ewm(
                span=20, adjust=False, min_periods=20
            ).mean()
            higher["slow"] = higher.close.ewm(
                span=50, adjust=False, min_periods=50
            ).mean()
            higher["bull"] = higher.fast > higher.slow
            higher["available_at"] = higher.open_time + duration
            higher = higher[["available_at", "bull"]].dropna().sort_values("available_at")
            higher = higher.rename(columns={"bull": f"{prefix}_bull"})
            x = pd.merge_asof(
                x.sort_values("open_time"),
                higher,
                left_on="open_time",
                right_on="available_at",
                direction="backward",
            )
            x = x.drop(columns=["available_at"])
        except Exception as exc:
            logger.warning("%s context unavailable: %s", interval, exc)
            x[f"{prefix}_bull"] = np.nan
    return x.reset_index(drop=True)


def estimate_depth_impact(symbol: str, side: int, qty: float) -> float | None:
    """Fractional cost of walking the public book vs best price (never submits)."""
    try:
        book = _get("/fapi/v1/depth", {"symbol": symbol, "limit": 100}, timeout=10)
        levels = book["asks"] if side == 1 else book["bids"]
        best = float(levels[0][0])
        remaining, notional = qty, 0.0
        for px_s, q_s in levels:
            px, avail = float(px_s), float(q_s)
            take = min(remaining, avail)
            notional += take * px
            remaining -= take
            if remaining <= 1e-12:
                break
        filled = qty - remaining
        if filled <= 0 or best <= 0:
            return None
        return max(0.0, (notional / filled / best - 1) * side)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Depth snapshot unavailable: %s", exc)
        return None


def estimate_depth_fill(
    symbol: str,
    side: int,
    qty: float,
    fallback_price: float,
    fallback_slippage: float,
) -> tuple[float, float, str]:
    """Estimate market fill from public order-book depth; never submits an order."""
    try:
        book = _get("/fapi/v1/depth", {"symbol": symbol, "limit": 100}, timeout=10)
        levels = book["asks"] if side == 1 else book["bids"]
        remaining, notional = qty, 0.0
        for px_s, q_s in levels:
            px, avail = float(px_s), float(q_s)
            take = min(remaining, avail)
            notional += take * px
            remaining -= take
            if remaining <= 1e-12:
                break
        filled = qty - remaining
        if filled > 0:
            vwap = notional / filled
            if remaining > 1e-12:
                tail_px = fallback_price * (1 + fallback_slippage * side)
                vwap = (notional + remaining * tail_px) / qty
            return vwap, qty, "public_depth_vwap_with_tail_estimate"
    except Exception as exc:
        logger.debug("Depth snapshot unavailable, using fallback slippage: %s", exc)
    return (
        fallback_price * (1 + fallback_slippage * side),
        qty,
        "fallback_slippage_model",
    )


def load_research_frame(days: int | None = None, interval: str | None = None) -> pd.DataFrame:
    """Fetch + enrich candles ready for feature engineering."""
    from modules.features import add_features

    days = days if days is not None else config.HISTORY_DAYS
    interval = interval or config.TIMEFRAME_MAIN
    raw = fetch_klines(interval, days, config.SYMBOL)
    raw = enrich_derivatives_data(raw, config.SYMBOL, days)
    raw = enrich_multi_timeframe(raw, config.SYMBOL, days)
    return add_features(raw)
