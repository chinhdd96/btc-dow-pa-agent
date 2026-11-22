"""Incremental Futures candles for the worker and full history for research."""

from __future__ import annotations

import pandas as pd

import config
from modules import quant_data
from modules.paper_champion import interval_td


def ms(timestamp: pd.Timestamp) -> int:
    return int(timestamp.value // 1_000_000)


def check_symbols(symbols: tuple[str, ...]) -> None:
    info = quant_data._get("/fapi/v1/exchangeInfo")
    supported = {
        row["symbol"] for row in info["symbols"]
        if row.get("status") == "TRADING" and row.get("contractType") == "PERPETUAL"
        and row.get("quoteAsset") == "USDT"
    }
    if missing := set(symbols) - supported:
        raise ValueError(f"Unavailable USDT perpetual symbols: {', '.join(sorted(missing))}")


class MarketData:
    def __init__(self, days: int) -> None:
        self.days = days
        self.raw: dict[tuple[str, str], pd.DataFrame] = {}
        self.funding: dict[str, dict[int, float]] = {}

    def candles(self, interval: str, days: int, symbol: str) -> pd.DataFrame:
        key = (symbol, interval)
        old = self.raw.get(key)
        if old is None:
            result = quant_data.fetch_klines(interval, days, symbol)
        else:
            start = ms(old.open_time.iloc[-1] + interval_td(interval))
            now = pd.Timestamp.now(tz="UTC")
            if now < pd.Timestamp(start, unit="ms", tz="UTC") + interval_td(interval):
                return old
            new = quant_data.fetch_klines(interval, days, symbol, start_ms=start, allow_empty=True)
            if new.empty:
                return old
            result = pd.concat([old, new], ignore_index=True)
            result = result[result.open_time >= now - pd.Timedelta(days=days)]
        if result.empty:
            raise RuntimeError(f"No closed Futures candles for {symbol} {interval}")
        self.raw[key] = result
        return result

    def frame(self, symbol: str) -> pd.DataFrame:
        raw = self.candles(config.TIMEFRAME_MAIN, self.days, symbol)
        funding_history = quant_data.fetch_funding_history(symbol, self.days if symbol not in self.funding else 2)
        frame = quant_data.enrich_derivatives_data(raw, symbol, funding_history=funding_history)
        funding = self.funding.setdefault(symbol, {})
        funding.update(
            (ms(t.floor("min")), float(rate))
            for t, rate in zip(funding_history.funding_time, funding_history.funding_event_rate)
            if pd.notna(rate)
        )
        funding = self.funding[symbol] = {t: rate for t, rate in funding.items() if t >= ms(raw.open_time.iloc[0])}
        frame["funding_event_rate"] = frame.open_time.map(lambda stamp: funding.get(ms(stamp)))
        frame["funding_rate"] = frame["funding_event_rate"].ffill().fillna(frame["funding_rate"])
        return quant_data.enrich_multi_timeframe(frame, symbol, self.days, fetch_higher=self.candles, strict=True)
