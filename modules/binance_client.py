"""Binance USD-M Futures client with hard risk guards."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import InsecureRequestWarning

import config

# Paper mode often sets BINANCE_SSL_VERIFY=false on Windows / geo fallback
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

logger = logging.getLogger(__name__)

FAPI_BASES = (
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
)
# Spot public mirror — fallback khi Futures bị 403 (geo)
SPOT_DATA_BASE = "https://data-api.binance.vision"


class BinanceClientError(Exception):
    pass


class BinanceClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        paper_mode: bool | None = None,
    ) -> None:
        key = api_key if api_key is not None else config.BINANCE_API_KEY
        secret = api_secret if api_secret is not None else config.BINANCE_API_SECRET
        self.paper_mode = config.PAPER_MODE if paper_mode is None else paper_mode
        self.symbol = config.SYMBOL
        self._exchange_info: dict[str, Any] | None = None
        self._session = requests.Session()
        self._session.verify = config.BINANCE_SSL_VERIFY
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; btc-dow-pa-agent/1.0; paper-mode)"
                ),
                "Accept": "application/json",
            }
        )
        self.client: Client | None = None
        self._fapi_base: str | None = None
        self._use_spot_fallback = False

        if self.paper_mode:
            logger.warning(
                "PAPER_MODE ON — public market data, orders → Telegram only (no real trade)"
            )
        else:
            if not key or not secret:
                raise BinanceClientError("Missing BINANCE_API_KEY / BINANCE_API_SECRET")
            self.client = Client(key, secret)

    def _public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET public market data; try Futures hosts then Spot vision fallback."""
        params = params or {}
        if self._use_spot_fallback and self.paper_mode:
            return self._spot_fallback(path, params)

        errors: list[str] = []
        bases = list(FAPI_BASES)
        if self._fapi_base:
            bases = [self._fapi_base] + [b for b in bases if b != self._fapi_base]

        for base in bases:
            url = f"{base}{path}"
            try:
                resp = self._session.get(url, params=params, timeout=30)
                if resp.status_code == 403:
                    errors.append(f"{base}: 403")
                    continue
                resp.raise_for_status()
                self._fapi_base = base
                return resp.json()
            except requests.RequestException as exc:
                errors.append(f"{base}: {exc}")
                continue

        if self.paper_mode and path in {
            "/fapi/v1/klines",
            "/fapi/v1/premiumIndex",
            "/fapi/v1/exchangeInfo",
        }:
            self._use_spot_fallback = True
            logger.warning(
                "Futures public API blocked (%s) — using Spot data-api fallback",
                "; ".join(errors[-3:]) or "unknown",
            )
            return self._spot_fallback(path, params)

        raise BinanceClientError(
            "Binance public API failed: " + "; ".join(errors[-4:])
        )

    def _spot_fallback(self, path: str, params: dict[str, Any]) -> Any:
        symbol = params.get("symbol", self.symbol)
        if path == "/fapi/v1/klines":
            resp = self._session.get(
                f"{SPOT_DATA_BASE}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": params.get("interval", "1h"),
                    "limit": params.get("limit", 50),
                    **(
                        {"startTime": params["startTime"]}
                        if "startTime" in params
                        else {}
                    ),
                    **(
                        {"endTime": params["endTime"]}
                        if "endTime" in params
                        else {}
                    ),
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        if path == "/fapi/v1/premiumIndex":
            resp = self._session.get(
                f"{SPOT_DATA_BASE}/api/v3/ticker/price",
                params={"symbol": symbol},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return {"markPrice": data["price"], "symbol": symbol}
        if path == "/fapi/v1/exchangeInfo":
            resp = self._session.get(
                f"{SPOT_DATA_BASE}/api/v3/exchangeInfo",
                params={"symbol": symbol},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Normalize to futures-like shape used by _get_symbol_filters
            return {"symbols": data.get("symbols", [])}
        raise BinanceClientError(f"No spot fallback for {path}")

    def _get_symbol_filters(self) -> dict[str, Any]:
        if self._exchange_info is None:
            if self.paper_mode or self.client is None:
                self._exchange_info = self._public_get("/fapi/v1/exchangeInfo")
            else:
                self._exchange_info = self.client.futures_exchange_info()
        for s in self._exchange_info.get("symbols", []):
            if s.get("symbol") == self.symbol:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                return {
                    "quantityPrecision": int(s.get("quantityPrecision", 3)),
                    "pricePrecision": int(s.get("pricePrecision", 2)),
                    "LOT_SIZE": filters.get("LOT_SIZE", {}),
                    "MIN_NOTIONAL": filters.get("MIN_NOTIONAL")
                    or filters.get("NOTIONAL", {}),
                }
        raise BinanceClientError(f"Symbol {self.symbol} not found on Futures")

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        floored = math.floor(value / step) * step
        return float(f"{floored:.{precision}f}")

    def get_klines(
        self,
        interval: str,
        limit: int = 50,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(max(int(limit), 1), 1500),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        try:
            if self.paper_mode or self.client is None:
                raw = self._public_get("/fapi/v1/klines", params)
            else:
                raw = self.client.futures_klines(**params)
        except BinanceAPIException as exc:
            logger.exception("get_klines failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

        candles = []
        for row in raw:
            candles.append(
                {
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": int(row[6]),
                }
            )
        return candles

    def get_klines_range(
        self,
        interval: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        max_bars: int | None = None,
        max_pages: int = 8,
        page_sleep_sec: float = 0.12,
    ) -> list[dict[str, Any]]:
        """Paginate klines; dừng sớm nếu không thêm timestamp mới."""
        import time

        page_limit = 1000
        by_ts: dict[int, dict[str, Any]] = {}
        cursor = start_time
        prev_n = 0
        for _ in range(max(1, max_pages)):
            batch = self.get_klines(
                interval,
                limit=page_limit,
                start_time=cursor,
                end_time=end_time,
            )
            if not batch:
                break
            for c in batch:
                by_ts[int(c["open_time"])] = c
            if len(by_ts) == prev_n:
                break
            prev_n = len(by_ts)
            if max_bars is not None and len(by_ts) >= max_bars:
                break
            if len(batch) < page_limit:
                break
            cursor = int(batch[-1]["open_time"]) + 1
            if end_time is not None and cursor > end_time:
                break
            if page_sleep_sec > 0:
                time.sleep(page_sleep_sec)

        out = [by_ts[k] for k in sorted(by_ts)]
        if max_bars is not None and len(out) > max_bars:
            out = out[-max_bars:]
        return out

    def get_mark_price(self) -> float:
        try:
            if self.paper_mode or self.client is None:
                data = self._public_get(
                    "/fapi/v1/premiumIndex", {"symbol": self.symbol}
                )
            else:
                data = self.client.futures_mark_price(symbol=self.symbol)
            return float(data["markPrice"])
        except BinanceAPIException as exc:
            logger.exception("get_mark_price failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

    def get_balance_usdt(self) -> float:
        if self.paper_mode:
            return float(config.PAPER_BALANCE_USDT)
        try:
            balances = self.client.futures_account_balance()
            for item in balances:
                if item.get("asset") == "USDT":
                    return float(item.get("availableBalance", item.get("balance", 0)))
            return 0.0
        except BinanceAPIException as exc:
            logger.exception("get_balance_usdt failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

    def get_position(self) -> dict[str, Any] | None:
        # Paper: no real position tracking — mỗi chu kỳ có thể ra signal mới
        if self.paper_mode:
            return None
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
        except BinanceAPIException as exc:
            logger.exception("get_position failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

        for pos in positions:
            amt = float(pos.get("positionAmt", 0))
            if abs(amt) > 0:
                side = "BUY" if amt > 0 else "SELL"
                return {
                    "side": side,
                    "amount": abs(amt),
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                    "leverage": int(float(pos.get("leverage", config.LEVERAGE))),
                    "raw": pos,
                }
        return None

    def set_leverage(self, leverage: int | None = None) -> None:
        if self.paper_mode:
            return
        if self.client is None:
            raise BinanceClientError("Binance client not initialized")
        lev = int(leverage or config.LEVERAGE)
        lev = min(lev, config.LEVERAGE)
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=lev)
            logger.info("Leverage set to %sx for %s", lev, self.symbol)
        except BinanceAPIException as exc:
            if getattr(exc, "code", None) == -4028:
                return
            logger.exception("set_leverage failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

    def validate_stop_loss(
        self,
        action: str,
        entry: float,
        stop_loss: float | None,
    ) -> tuple[bool, str]:
        if stop_loss is None or stop_loss <= 0:
            return False, "Thiếu giá cắt lỗ (SL)"
        if entry <= 0:
            return False, "Giá vào lệnh không hợp lệ"
        distance_pct = abs(entry - stop_loss) / entry
        if distance_pct > config.MAX_SL_DISTANCE_PCT:
            return (
                False,
                f"Khoảng cách SL {distance_pct:.2%} > tối đa {config.MAX_SL_DISTANCE_PCT:.2%}",
            )
        if action == "BUY" and stop_loss >= entry:
            return False, "BUY yêu cầu SL < giá vào"
        if action == "SELL" and stop_loss <= entry:
            return False, "SELL yêu cầu SL > giá vào"
        return True, "ok"

    def calc_position_size(
        self,
        balance: float,
        entry: float,
        stop_loss: float,
        risk_pct: float | None = None,
    ) -> float:
        risk_pct = risk_pct if risk_pct is not None else config.RISK_PER_TRADE_PCT
        risk_amount = balance * risk_pct
        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            raise BinanceClientError("Stop loss distance must be > 0")

        raw_qty = risk_amount / sl_distance
        filters = self._get_symbol_filters()
        lot = filters.get("LOT_SIZE", {})
        step = float(lot.get("stepSize", "0.001") or "0.001")
        min_qty = float(lot.get("minQty", "0.001") or "0.001")
        qty = self._round_step(raw_qty, step)
        if qty < min_qty:
            raise BinanceClientError(
                f"Calculated qty {qty} < minQty {min_qty}; risk too small for SL distance"
            )

        min_notional_filter = filters.get("MIN_NOTIONAL", {})
        min_notional = float(
            min_notional_filter.get("notional")
            or min_notional_filter.get("minNotional")
            or 5
        )
        notional = qty * entry
        if notional < min_notional:
            raise BinanceClientError(
                f"Notional {notional:.2f} < minNotional {min_notional}"
            )
        return qty

    def cancel_all_open_orders(self) -> None:
        if self.paper_mode:
            return
        if self.client is None:
            return
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.symbol)
        except BinanceAPIException as exc:
            logger.warning("cancel_all_open_orders: %s", exc)

    def close_and_cancel(self) -> dict[str, Any] | None:
        if self.paper_mode:
            return None
        if self.client is None:
            raise BinanceClientError("Binance client not initialized")
        position = self.get_position()
        self.cancel_all_open_orders()
        if not position:
            return None
        side = "SELL" if position["side"] == "BUY" else "BUY"
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=position["amount"],
                reduceOnly=True,
            )
            logger.info("Closed position via market: %s", order.get("orderId"))
            return order
        except BinanceAPIException as exc:
            logger.exception("close_and_cancel failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

    def open_bracket(
        self,
        action: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        balance: float | None = None,
    ) -> dict[str, Any]:
        action = action.upper()
        if action not in {"BUY", "SELL"}:
            raise BinanceClientError(f"Invalid action for open_bracket: {action}")

        ok, reason = self.validate_stop_loss(action, entry_price, stop_loss_price)
        if not ok:
            raise BinanceClientError(f"Hard SL guard: {reason}")

        if take_profit_price is None or take_profit_price <= 0:
            raise BinanceClientError("Missing take_profit_price")

        if action == "BUY" and take_profit_price <= entry_price:
            raise BinanceClientError("BUY requires take_profit > entry")
        if action == "SELL" and take_profit_price >= entry_price:
            raise BinanceClientError("SELL requires take_profit < entry")

        if not self.paper_mode:
            existing = self.get_position()
            if existing:
                raise BinanceClientError(
                    f"Already in position {existing['side']} qty={existing['amount']}"
                )

        bal = balance if balance is not None else self.get_balance_usdt()
        qty = self.calc_position_size(bal, entry_price, stop_loss_price)

        filters = self._get_symbol_filters()
        price_precision = filters["pricePrecision"]
        sl_price = round(float(stop_loss_price), price_precision)
        tp_price = round(float(take_profit_price), price_precision)

        # Paper: validate + size only — caller sends Telegram, no exchange order
        if self.paper_mode:
            logger.info(
                "PAPER open_bracket %s qty=%s entry=%s SL=%s TP=%s",
                action,
                qty,
                entry_price,
                sl_price,
                tp_price,
            )
            return {
                "paper": True,
                "quantity": qty,
                "entry_order": None,
                "sl_order": None,
                "tp_order": None,
                "stop_loss": sl_price,
                "take_profit": tp_price,
            }

        self.set_leverage(config.LEVERAGE)
        self.cancel_all_open_orders()

        assert self.client is not None
        try:
            entry_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=action,
                type="MARKET",
                quantity=qty,
            )
        except BinanceAPIException as exc:
            logger.exception("Entry MARKET failed: %s", exc)
            raise BinanceClientError(str(exc)) from exc

        close_side = "SELL" if action == "BUY" else "BUY"

        sl_order = None
        tp_order = None
        try:
            sl_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=sl_price,
                closePosition=True,
                workingType="MARK_PRICE",
            )
        except BinanceAPIException as exc:
            logger.exception("STOP_MARKET failed: %s — attempting emergency close", exc)
            self.close_and_cancel()
            raise BinanceClientError(f"SL order failed, position closed: {exc}") from exc

        try:
            tp_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_price,
                closePosition=True,
                workingType="MARK_PRICE",
            )
        except BinanceAPIException as exc:
            logger.exception("TAKE_PROFIT_MARKET failed: %s", exc)
            raise BinanceClientError(
                f"TP order failed (SL is live): {exc}"
            ) from exc

        return {
            "quantity": qty,
            "entry_order": entry_order,
            "sl_order": sl_order,
            "tp_order": tp_order,
            "stop_loss": sl_price,
            "take_profit": tp_price,
        }
