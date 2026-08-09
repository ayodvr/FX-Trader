"""
Thin wrapper around pybit's unified trading HTTP client.

Keeping this separate from strategy/risk means swapping exchanges later
(or adding a second one) only requires a new file like this — not a rewrite.
"""
import logging
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

import pandas as pd
from pybit.unified_trading import HTTP

from config import ExchangeConfig

logger = logging.getLogger("exchange")


class BybitExchange:
    def __init__(self, cfg: ExchangeConfig):
        self.cfg = cfg
        self.client = HTTP(
            testnet=cfg.testnet,
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
            domain="bytick",
        )
        # symbol -> {tick_size, qty_step, min_qty}; instrument specs never change
        # mid-session, so fetch once per symbol and reuse.
        self._instrument_cache: dict[str, dict] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Instrument precision
    #
    # Every symbol has its own tickSize (price increment) and qtyStep (size
    # increment). Hardcoding round(price, 2) works for BTCUSDT (~$64,000) but
    # destroys sub-dollar altcoins: a 1000PEPE stop at 0.002886 rounds to 0.0
    # and is rejected outright, and a COTI stop at 0.0152 rounds to 0.02 --
    # above the entry price, i.e. the wrong side of the trade. The scanner
    # trades whatever the top-30 by volume happen to be, so it must ask the
    # exchange for each symbol's real precision.
    # ──────────────────────────────────────────────────────────────────────────

    def get_instrument_info(self, symbol: str) -> dict:
        """Fetch (and cache) tickSize / qtyStep / minOrderQty for a symbol."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        info = {"tick_size": None, "qty_step": None, "min_qty": None,
                "min_notional": Decimal("5")}
        try:
            resp = self.client.get_instruments_info(category=self.cfg.category, symbol=symbol)
            item = resp["result"]["list"][0]
            lot = item["lotSizeFilter"]
            info = {
                "tick_size": Decimal(str(item["priceFilter"]["tickSize"])),
                "qty_step":  Decimal(str(lot["qtyStep"])),
                "min_qty":   Decimal(str(lot["minOrderQty"])),
                # Bybit also enforces a minimum order VALUE (5 USDT on linear
                # perps) independently of minOrderQty -- a size can clear the
                # quantity floor and still be rejected on notional.
                "min_notional": Decimal(str(lot.get("minNotionalValue") or "5")),
            }
            self._instrument_cache[symbol] = info
        except Exception as e:
            # Don't cache failures -- a transient error shouldn't poison every
            # later order for this symbol.
            logger.warning("Could not fetch instrument info for %s: %s", symbol, e)
        return info

    def format_price(self, symbol: str, price: float) -> str:
        """Round a price to the symbol's tickSize and render it without exponent notation."""
        tick = self.get_instrument_info(symbol)["tick_size"]
        if tick is None or tick <= 0:
            logger.warning("No tickSize for %s -- falling back to 2dp, which may be wrong", symbol)
            return f"{price:.2f}"
        d = Decimal(str(price))
        snapped = (d / tick).to_integral_value(rounding=ROUND_HALF_UP) * tick
        # format(..., 'f') avoids str()'s scientific notation (e.g. '1e-06'),
        # which Bybit rejects.
        return format(snapped.quantize(tick), "f")

    def format_qty(self, symbol: str, qty: float) -> str:
        """Round a quantity DOWN to the symbol's qtyStep (never round up past what we hold)."""
        step = self.get_instrument_info(symbol)["qty_step"]
        if step is None or step <= 0:
            logger.warning("No qtyStep for %s -- sending qty unrounded", symbol)
            return format(Decimal(str(qty)), "f")
        d = Decimal(str(qty))
        snapped = (d / step).to_integral_value(rounding=ROUND_DOWN) * step
        return format(snapped.quantize(step), "f")

    def meets_min_qty(self, symbol: str, qty: float, price: float | None = None) -> bool:
        """True if qty clears the symbol's minimum order size after step-rounding.

        When `price` is given, also checks Bybit's minimum order VALUE -- an order
        can satisfy minOrderQty and still be rejected with ErrCode 110094 for
        falling under the notional floor.
        """
        info = self.get_instrument_info(symbol)
        step, min_qty = info["qty_step"], info["min_qty"]
        if step is None or min_qty is None:
            return qty > 0
        snapped = (Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_DOWN) * step
        if snapped < min_qty:
            return False
        if price is not None:
            notional = snapped * Decimal(str(price))
            if notional < info.get("min_notional", Decimal("5")):
                logger.info("%s: notional %.4f below minimum %s", symbol, notional,
                            info.get("min_notional"))
                return False
        return True

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        resp = self.client.get_kline(
            category=self.cfg.category,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        rows = resp["result"]["list"]
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df = df.astype({
            "timestamp": "int64", "open": "float64", "high": "float64",
            "low": "float64", "close": "float64", "volume": "float64",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index()  # Bybit returns newest-first
        return df

    def get_top_symbols(self, limit: int = 30) -> list[str]:
        """Fetch top N USDT linear perpetual symbols ranked by 24h turnover/volume."""
        try:
            resp = self.client.get_tickers(category=self.cfg.category)
            tickers = resp.get("result", {}).get("list", [])
            # Filter for USDT linear contracts, exclude USDC / Inverse
            usdt_tickers = [
                t for t in tickers
                if t.get("symbol", "").endswith("USDT") and not t.get("symbol", "").startswith("USDC")
            ]
            # Sort descending by 24h turnover (turnover24h)
            usdt_tickers.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
            symbols = [t["symbol"] for t in usdt_tickers[:limit]]
            return symbols if symbols else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "AVAXUSDT"]
        except Exception as e:
            logger.warning("Failed to fetch top tickers from exchange: %s", e)
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "AVAXUSDT"]

    def get_equity(self, coin: str = "USDT") -> float:
        try:
            resp = self.client.get_wallet_balance(accountType="UNIFIED", coin=coin)
            return float(resp["result"]["list"][0]["totalEquity"])
        except Exception as e:
            logger.warning("Could not fetch equity from exchange (%s): %s", coin, e)
            return 0.0

    def get_open_position(self, symbol: str) -> dict | None:
        try:
            resp = self.client.get_positions(category=self.cfg.category, symbol=symbol)
            positions = resp.get("result", {}).get("list", [])
            for p in positions:
                if float(p.get("size", 0)) > 0:
                    return p
            return None
        except Exception as e:
            logger.warning("Could not fetch open position from exchange (%s): %s", symbol, e)
            return None

    def get_all_open_positions(self) -> list[dict]:
        """Return every open position on the account, regardless of symbol.

        Needed because the quant scanner can hold positions in any of its
        top-N scanned symbols (not a fixed list) — per-symbol lookups can't
        be used to enumerate what's actually open.
        """
        try:
            resp = self.client.get_positions(category=self.cfg.category, settleCoin="USDT")
            positions = resp.get("result", {}).get("list", [])
            return [p for p in positions if float(p.get("size", 0)) > 0]
        except Exception as e:
            logger.warning("Could not fetch all open positions: %s", e)
            return []

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.client.set_leverage(
                category=self.cfg.category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
        except Exception as e:
            # Bybit errors if leverage is already set to this value — safe to ignore
            logger.info("set_leverage note: %s", e)

    def place_market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
        """side: 'Buy' or 'Sell'"""
        qty_str = self.format_qty(symbol, qty)
        logger.info("Placing %s market order: %s qty=%s reduce_only=%s", side, symbol, qty_str, reduce_only)
        resp = self.client.place_order(
            category=self.cfg.category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=qty_str,
            reduceOnly=reduce_only,
        )
        return resp

    def place_limit_order(self, symbol: str, side: str, qty: float, price: float, reduce_only: bool = False) -> dict:
        """Place a Limit order at a specific price."""
        qty_str = self.format_qty(symbol, qty)
        price_str = self.format_price(symbol, price)
        logger.info("Placing %s limit order: %s qty=%s price=%s reduce_only=%s",
                    side, symbol, qty_str, price_str, reduce_only)
        resp = self.client.place_order(
            category=self.cfg.category,
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=qty_str,
            price=price_str,
            reduceOnly=reduce_only,
        )
        return resp

    def place_stop_order(self, symbol: str, side: str, qty: float, trigger_price: float) -> dict:
        """Conditional market order that triggers at trigger_price, closing the position.

        This is always a protective stop, so the trigger direction is fixed by which
        side is closing: a Sell closes a long and sits below price (fires on a
        Falling move); a Buy closes a short and sits above price (fires on a Rising
        move). Bybit's triggerDirection: 1 = Rising, 2 = Falling.
        """
        qty_str = self.format_qty(symbol, qty)
        trigger_str = self.format_price(symbol, trigger_price)
        logger.info("Placing %s stop order: %s qty=%s trigger=%s", side, symbol, qty_str, trigger_str)
        resp = self.client.place_order(
            category=self.cfg.category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=qty_str,
            triggerPrice=trigger_str,
            reduceOnly=True,
            triggerDirection=2 if side == "Sell" else 1,
        )
        return resp

    def get_all_open_orders(self, symbol: str) -> list[dict]:
        """Return every open order for a symbol -- regular (limit/market) and conditional/stop.

        Deduped by orderId: the unfiltered query already includes stop orders for
        the linear category on Bybit, so combining it with the StopOrder-filtered
        query would otherwise double-count them.
        """
        orders = {}
        try:
            resp = self.client.get_open_orders(category=self.cfg.category, symbol=symbol)
            for o in resp.get("result", {}).get("list", []):
                orders[o.get("orderId")] = o
        except Exception as e:
            logger.warning("get_all_open_orders failed: %s", e)
        for o in self.get_open_stop_orders(symbol):
            orders[o.get("orderId")] = o
        return list(orders.values())

    def get_open_stop_orders(self, symbol: str) -> list[dict]:
        """Return all open conditional/stop orders for a symbol."""
        try:
            resp = self.client.get_open_orders(
                category=self.cfg.category,
                symbol=symbol,
                orderFilter="StopOrder",
            )
            return resp.get("result", {}).get("list", [])
        except Exception as e:
            logger.warning("get_open_stop_orders failed: %s", e)
            return []

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order by orderId. Returns True on success."""
        try:
            self.client.cancel_order(
                category=self.cfg.category,
                symbol=symbol,
                orderId=order_id,
            )
            return True
        except Exception as e:
            logger.warning("cancel_order(%s) failed: %s", order_id, e)
            return False

    def cancel_all_stops(self, symbol: str):
        """Cancel every open stop/conditional order for the symbol."""
        orders = self.get_open_stop_orders(symbol)
        for order in orders:
            oid = order.get("orderId", "")
            if oid:
                self.cancel_order(symbol, oid)

    def wait_for_next_candle_close(self, interval_min: int):
        """Sleeps until just after the next candle close boundary."""
        interval_sec = interval_min * 60
        now = time.time()
        next_close = ((now // interval_sec) + 1) * interval_sec
        sleep_for = next_close - now + 2  # +2s buffer for exchange data to settle
        time.sleep(max(sleep_for, 0))

    def cancel_all_orders(self, symbol: str):
        """Cancel all active and conditional orders instantly."""
        try:
            self.client.cancel_all_orders(category=self.cfg.category, symbol=symbol)
            logger.info("Cancelled all orders for %s", symbol)
        except Exception as e:
            logger.warning("Failed to cancel all orders for %s: %s", symbol, e)

    def close_all_positions(self, symbol: str):
        """Aggressively close any open position with a Market order."""
        pos = self.get_open_position(symbol)
        if pos and float(pos.get("size", 0)) > 0:
            side = "Sell" if pos["side"] == "Buy" else "Buy"
            qty = pos["size"]
            try:
                self.client.place_order(
                    category=self.cfg.category,
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=qty,
                    reduceOnly=True,
                )
                logger.warning("EMERGENCY CLOSE: %s %s %s @ Market", side, qty, symbol)
            except Exception as e:
                logger.error("EMERGENCY CLOSE FAILED for %s: %s", symbol, e)
