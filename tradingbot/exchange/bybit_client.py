"""
Thin wrapper around pybit's unified trading HTTP client.

Keeping this separate from strategy/risk means swapping exchanges later
(or adding a second one) only requires a new file like this — not a rewrite.
"""
import logging
import time
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
        logger.info("Placing %s market order: %s qty=%.6f reduce_only=%s", side, symbol, qty, reduce_only)
        resp = self.client.place_order(
            category=self.cfg.category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=reduce_only,
        )
        return resp

    def place_limit_order(self, symbol: str, side: str, qty: float, price: float, reduce_only: bool = False) -> dict:
        """Place a Limit order at a specific price."""
        logger.info("Placing %s limit order: %s qty=%.6f price=%.2f reduce_only=%s", side, symbol, qty, price, reduce_only)
        resp = self.client.place_order(
            category=self.cfg.category,
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty),
            price=str(round(price, 2)),
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
        resp = self.client.place_order(
            category=self.cfg.category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            triggerPrice=str(round(trigger_price, 2)),
            reduceOnly=True,
            triggerDirection=2 if side == "Sell" else 1,
        )
        return resp

    def get_all_open_orders(self, symbol: str) -> list[dict]:
        """Return every open order for a symbol -- regular (limit/market) and conditional/stop."""
        orders = []
        try:
            resp = self.client.get_open_orders(category=self.cfg.category, symbol=symbol)
            orders.extend(resp.get("result", {}).get("list", []))
        except Exception as e:
            logger.warning("get_all_open_orders failed: %s", e)
        orders.extend(self.get_open_stop_orders(symbol))
        return orders

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
