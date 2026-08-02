"""
Fetch historical OHLCV from Bybit and save to CSV for backtesting.
Paginates backward since Bybit caps each request at 1000 candles.

Includes:
  - Exponential-backoff retry on transient HTTP / rate-limit errors
  - Progress reporting (candles fetched, date range covered)
  - Graceful handling of server errors mid-pagination

Usage:
    python fetch_history.py --symbol BTCUSDT --interval 60 --days 365
    python fetch_history.py --symbol ETHUSDT --interval 15 --days 180 --out data/ETH.csv
"""
import argparse
import time
import sys
from pathlib import Path

import pandas as pd
from pybit.unified_trading import HTTP


MAX_RETRIES = 5
BASE_BACKOFF = 1.5   # seconds (doubles each retry)


def _fetch_page(client, symbol: str, interval: str, end_ms: int) -> list:
    """Fetch one page of klines with exponential-backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                end=end_ms,
                limit=1000,
            )
            return resp["result"]["list"]
        except Exception as e:
            wait = BASE_BACKOFF ** attempt
            print(f"  [warn] page fetch failed (attempt {attempt + 1}/{MAX_RETRIES}): {e} — retrying in {wait:.1f}s")
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed to fetch page after {MAX_RETRIES} attempts: {e}") from e
            time.sleep(wait)
    return []   # unreachable


def fetch_history(symbol: str, interval: str, days: int, testnet: bool = False) -> pd.DataFrame:
    # Use bytick domain to avoid ISP DNS blocks common with api.bybit.com
    client = HTTP(testnet=testnet, domain="bytick")
    interval_ms  = int(interval) * 60 * 1000
    end_ms       = int(time.time() * 1000)
    start_ms     = end_ms - days * 24 * 60 * 60 * 1000
    expected_candles = days * 24 * 60 // int(interval)

    print(f"Fetching {symbol} {interval}m candles for {days} days (~{expected_candles:,} candles expected)...")

    all_rows = []
    cursor_end = end_ms
    pages = 0

    while cursor_end > start_ms:
        rows = _fetch_page(client, symbol, interval, cursor_end)
        if not rows:
            break

        all_rows.extend(rows)
        pages += 1
        oldest_ts = int(rows[-1][0])
        newest_ts = int(rows[0][0])

        # Progress indicator every 5 pages
        if pages % 5 == 0 or oldest_ts <= start_ms:
            from datetime import datetime, timezone
            oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            newest_dt = datetime.fromtimestamp(newest_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  [{pages:>3} pages | {len(all_rows):>6,} candles]  {oldest_dt} -> {newest_dt}")

        cursor_end = oldest_ts - interval_ms
        time.sleep(0.12)   # ~8 req/s — well within Bybit's limit

    if not all_rows:
        raise ValueError(f"No data returned for {symbol} {interval}m. "
                         "Check that the symbol and interval are valid.")

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df = df.astype({
        "timestamp": "int64", "open": "float64", "high": "float64",
        "low": "float64", "close": "float64", "volume": "float64",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (df
          .drop_duplicates(subset="timestamp")
          .set_index("timestamp")
          .sort_index())
    df = df[df.index >= pd.to_datetime(start_ms, unit="ms", utc=True)]

    print(f"\nDone. {len(df):,} candles from {df.index[0].date()} to {df.index[-1].date()}")
    return df[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Bybit OHLCV history for backtesting")
    parser.add_argument("--symbol",   default="BTCUSDT")
    parser.add_argument("--interval", default="60",  help="Candle interval in minutes")
    parser.add_argument("--days",     type=int, default=365)
    parser.add_argument("--out",      default=None,  help="Output CSV path (default: data/<SYMBOL>_<INTERVAL>m.csv)")
    parser.add_argument("--testnet",  action="store_true")
    args = parser.parse_args()

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = args.out or str(out_dir / f"{args.symbol}_{args.interval}m.csv")

    try:
        df = fetch_history(args.symbol, args.interval, args.days, testnet=args.testnet)
        df.to_csv(out_path)
        print(f"Saved -> {out_path}")
    except (RuntimeError, ValueError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
