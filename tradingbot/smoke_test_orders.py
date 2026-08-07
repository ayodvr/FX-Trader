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
# 0.002 so that Test 3's 50% scale-out (0.001) still clears Bybit's minimum
# contract size for BTCUSDT -- confirmed via Tests 1/2 that 0.001 itself is
# the minimum, so half of that would be rejected.
QTY = 0.002


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


def test_full_lifecycle(exchange):
    """
    Mirrors live/run.py's full entry -> scale-out limit -> trailing-stop-replace
    -> exit sequence, without waiting for a real strategy signal.
    """
    print("=== TEST 3: full lifecycle (entry -> scale-out limit -> trail replace -> exit) ===")
    try:
        price = exchange.get_klines(SYMBOL, "15", limit=2).iloc[-1]["close"]
        exchange.place_market_order(SYMBOL, "Buy", QTY)
        time.sleep(2)
        _print_position(exchange, "after entry")

        stop_price = round(price * 0.99, 1)
        stop_resp = exchange.place_stop_order(SYMBOL, "Sell", QTY, stop_price)
        stop_id = stop_resp.get("result", {}).get("orderId") if stop_resp else None
        print(f"PASS: initial stop placed, orderId={stop_id}" if stop_id else f"FAIL: initial stop: {stop_resp}")

        tp_price = round(price * 1.02, 1)  # 2% above -- rests without filling immediately
        scale_qty = round(QTY * 0.5, 6)
        limit_resp = exchange.place_limit_order(SYMBOL, "Sell", scale_qty, tp_price, reduce_only=True)
        limit_id = limit_resp.get("result", {}).get("orderId") if limit_resp else None
        print(f"PASS: scale-out limit order placed, orderId={limit_id}" if limit_id
              else f"FAIL: scale-out limit order: {limit_resp}")

        # Trailing-stop replace: cancel the old stop, place a new one closer to price --
        # same cancel-then-place cycle live/run.py runs each time the trail ratchets.
        if stop_id:
            cancelled = exchange.cancel_order(SYMBOL, stop_id)
            print(f"PASS: old stop cancelled ({stop_id})" if cancelled
                  else f"FAIL: old stop cancel returned False ({stop_id})")
            time.sleep(1)  # let the cancellation propagate before re-querying
        new_stop_price = round(price * 0.995, 1)
        new_stop_resp = exchange.place_stop_order(SYMBOL, "Sell", QTY, new_stop_price)
        new_stop_id = new_stop_resp.get("result", {}).get("orderId") if new_stop_resp else None
        print(f"PASS: trail-replace stop placed, orderId={new_stop_id}" if new_stop_id
              else f"FAIL: trail-replace stop: {new_stop_resp}")

        open_orders = exchange.get_all_open_orders(SYMBOL)
        print(f"Open orders before exit: {len(open_orders)} (expect 2 -- the replaced stop + the scale-out limit)")
        for o in open_orders:
            print(f"  - orderId={o.get('orderId')} type={o.get('orderType')} "
                  f"triggerPrice={o.get('triggerPrice')} price={o.get('price')}")
        if stop_id in [o.get("orderId") for o in open_orders]:
            print(f"FAIL: old stop {stop_id} is still open -- cancel did not actually remove it")

        # Exit sequence, matching the fixed live/run.py: cancel everything resting
        # (stop AND scale-out limit) before the market close, so nothing is orphaned.
        exchange.cancel_all_orders(SYMBOL)
        exchange.place_market_order(SYMBOL, "Sell", QTY, reduce_only=True)
        time.sleep(2)
    finally:
        exchange.cancel_all_orders(SYMBOL)
        exchange.close_all_positions(SYMBOL)
        time.sleep(2)
        pos = _print_position(exchange, "after exit")
        remaining = exchange.get_all_open_orders(SYMBOL)
        print(f"Open orders after exit: {len(remaining)} (expect 0)")
        if pos is None and len(remaining) == 0:
            print("PASS: flat with no orphaned orders after full lifecycle")
        else:
            print("FAIL: leftover position or orphaned order(s) after exit!")


if __name__ == "__main__":
    _assert_safe()
    exchange = BybitExchange(CONFIG.exchange)
    test_happy_path(exchange)
    test_failsafe_path(exchange)
    test_full_lifecycle(exchange)
    print("Done. Review PASS/FAIL lines above.")
