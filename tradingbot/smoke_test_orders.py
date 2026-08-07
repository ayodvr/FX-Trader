"""
One-off smoke test for the live order-placement path used by live/run.py and
live/run_quant_scanner.py -- verifies the exchange integration works against
Bybit TESTNET without waiting for a real strategy signal.

Run once, manually:
    python3 -m smoke_test_orders

Places a real (tiny) market order and stop order on testnet, verifies them,
then closes everything. Safe: hard-aborts if BYBIT_TESTNET is not true, and
every code path cleans up the position before exiting.
"""
import sys
import time

from config import CONFIG
from exchange.bybit_client import BybitExchange

SYMBOL = "BTCUSDT"
QTY = 0.001  # minimal BTCUSDT perp size -- adjust if the exchange rejects it as below minimum


def _assert_safe():
    if not CONFIG.exchange.testnet:
        print("ABORT: BYBIT_TESTNET is not true. Refusing to run against mainnet.")
        sys.exit(1)
    print(f"Confirmed testnet=True. symbol={SYMBOL} qty={QTY}\n")


def _print_position(exchange, label):
    pos = exchange.get_open_position(SYMBOL)
    if pos and float(pos.get("size", 0)) > 0:
        print(f"[{label}] Position OPEN: side={pos.get('side')} size={pos.get('size')} avgPrice={pos.get('avgPrice')}")
    else:
        print(f"[{label}] Position FLAT")
    return pos


def test_happy_path(exchange):
    print("=== TEST 1: happy path (market entry -> valid stop order) ===")
    price = exchange.get_klines(SYMBOL, "15", limit=2).iloc[-1]["close"]
    print(f"Current price: {price}")

    try:
        exchange.place_market_order(SYMBOL, "Buy", QTY)
        time.sleep(2)
        _print_position(exchange, "after entry")

        stop_price = round(price * 0.99, 1)  # 1% below -- valid protective stop for a Buy
        resp = exchange.place_stop_order(SYMBOL, "Sell", QTY, stop_price)
        order_id = resp.get("result", {}).get("orderId") if resp else None
        if order_id:
            print(f"PASS: stop order placed successfully, orderId={order_id}")
        else:
            print(f"FAIL: stop order response had no orderId: {resp}")
    finally:
        exchange.cancel_all_stops(SYMBOL)
        exchange.close_all_positions(SYMBOL)
        time.sleep(2)
        _print_position(exchange, "after cleanup")
    print()


def test_failsafe_path(exchange):
    print("=== TEST 2: fail-safe path (entry -> invalid stop order -> emergency close) ===")
    try:
        exchange.place_market_order(SYMBOL, "Buy", QTY)
        time.sleep(2)
        _print_position(exchange, "after entry")

        try:
            # trigger_price=0 is always invalid, regardless of current market price --
            # guaranteed to be rejected by the exchange, forcing the same exception
            # path the real fail-safe in live/run.py handles.
            exchange.place_stop_order(SYMBOL, "Sell", QTY, 0)
            print("UNEXPECTED: invalid stop order was accepted instead of rejected")
        except Exception as e:
            print(f"Stop order failed as expected: {e}")
            print("Invoking fail-safe: close_all_positions(...)")
            exchange.close_all_positions(SYMBOL)
    finally:
        exchange.cancel_all_stops(SYMBOL)
        exchange.close_all_positions(SYMBOL)
        time.sleep(2)
        pos = _print_position(exchange, "after fail-safe")
        print("PASS: position is flat after fail-safe close" if pos is None
              else "FAIL: position is still open after fail-safe close!")


if __name__ == "__main__":
    _assert_safe()
    exchange = BybitExchange(CONFIG.exchange)
    test_happy_path(exchange)
    test_failsafe_path(exchange)
    print("Done. Review PASS/FAIL lines above.")
