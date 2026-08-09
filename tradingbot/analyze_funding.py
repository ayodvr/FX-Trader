"""
Funding-rate harvesting: is the delta-neutral carry trade viable on Bybit?

The trade
---------
Perpetual futures have no expiry, so they are tethered to spot by a funding
payment every 8 hours. When the perp trades above spot (crowded longs), funding
is positive and longs pay shorts. Holding **long spot + short perp** in equal
size leaves you with no directional exposure -- if price doubles, the spot leg
gains what the perp leg loses -- while the short perp collects funding every 8h.

That is the entire edge, and it is structural rather than predictive: you are
paid for taking the other side of leveraged longs, not for forecasting anything.
No signal, no entry timing, no stop-loss.

What this script measures
-------------------------
Real historical funding from Bybit (public endpoint, no API key needed), then:
  - annualised gross yield per symbol
  - how often funding goes negative, and how bad the negative stretches get
  - net yield after the round-trip cost of opening and closing BOTH legs
  - the break-even holding period -- below it, fees exceed funding collected

What it deliberately does not claim
-----------------------------------
Delta-neutral is not risk-free. Not modelled here: liquidation of the short leg
if the position is under-margined during a spike, spot/perp basis moving against
you at entry or exit, borrow costs if the spot leg is margined rather than owned
outright, and exchange/counterparty risk. Treat the output as an upper bound.

Usage:
    python analyze_funding.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 365
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

PERIODS_PER_DAY = 3        # funding settles every 8h
PERIODS_PER_YEAR = 365 * PERIODS_PER_DAY


def fetch_funding_history(symbol: str, days: int, testnet: bool = False) -> pd.DataFrame:
    """Page backwards through Bybit's funding history (200 rows per request)."""
    client = HTTP(testnet=testnet, domain="bytick")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    rows, cursor = [], end_ms

    while cursor > start_ms:
        try:
            resp = client.get_funding_rate_history(
                category="linear", symbol=symbol, endTime=cursor, limit=200
            )
            page = resp.get("result", {}).get("list", [])
        except Exception as e:
            print(f"  [warn] {symbol}: {e}")
            break
        if not page:
            break
        rows.extend(page)
        oldest = int(page[-1]["fundingRateTimestamp"])
        if oldest <= start_ms or len(page) < 200:
            break
        cursor = oldest - 1
        time.sleep(0.12)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    df = (df[["time", "funding_rate"]]
          .drop_duplicates(subset="time")
          .set_index("time")
          .sort_index())
    return df[df.index >= pd.to_datetime(start_ms, unit="ms", utc=True)]


def analyse(symbol: str, df: pd.DataFrame, round_trip_cost: float) -> dict:
    r = df["funding_rate"]
    n = len(r)
    mean = r.mean()
    gross_annual = mean * PERIODS_PER_YEAR * 100
    neg_share = (r < 0).mean() * 100

    # Cumulative funding, and the worst peak-to-trough stretch of it. This is
    # the real risk of the carry trade: funding flipping negative and staying
    # there while you keep paying to hold the position.
    cum = r.cumsum()
    worst_dd = (cum - cum.cummax()).min() * 100

    # Longest unbroken run of negative funding
    longest_neg, run = 0, 0
    for v in r:
        run = run + 1 if v < 0 else 0
        longest_neg = max(longest_neg, run)

    # Break-even: how many 8h periods of average funding pay for the round trip
    breakeven_periods = (round_trip_cost / mean) if mean > 0 else float("inf")

    return {
        "symbol": symbol,
        "periods": n,
        "days": n / PERIODS_PER_DAY,
        "mean_8h_%": mean * 100,
        "gross_annual_%": gross_annual,
        "neg_share_%": neg_share,
        "longest_neg_run": longest_neg,
        "worst_cum_dd_%": worst_dd,
        "breakeven_days": breakeven_periods / PERIODS_PER_DAY,
        "median_8h_%": r.median() * 100,
        "p05_8h_%": r.quantile(0.05) * 100,
        "p95_8h_%": r.quantile(0.95) * 100,
    }


def net_yield(df: pd.DataFrame, hold_days: float, round_trip_cost: float) -> float:
    """Annualised net yield if the position is opened and closed every hold_days."""
    mean = df["funding_rate"].mean()
    per_cycle = mean * PERIODS_PER_DAY * hold_days - round_trip_cost
    return (per_cycle / hold_days) * 365 * 100


def main():
    p = argparse.ArgumentParser(description="Analyse Bybit funding rates for carry viability")
    p.add_argument("--symbols", nargs="+",
                   default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"])
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--spot-fee", type=float, default=0.001,
                   help="Spot taker fee per side (Bybit spot ~0.1%%)")
    p.add_argument("--perp-fee", type=float, default=0.00055,
                   help="Perp taker fee per side")
    p.add_argument("--slippage", type=float, default=0.0005, help="Slippage per leg per side")
    p.add_argument("--out", default="funding_analysis.csv")
    args = p.parse_args()

    # Open and close, two legs each: (spot + perp + slippage on both) x 2
    round_trip = 2 * (args.spot_fee + args.perp_fee + 2 * args.slippage)
    print(f"Round-trip cost of the pair (open+close, both legs): {round_trip*100:.3f}%")
    print(f"Fetching {args.days}d of funding history...\n")

    results, frames = [], {}
    for sym in args.symbols:
        df = fetch_funding_history(sym, args.days)
        if df.empty:
            print(f"  {sym}: no data")
            continue
        frames[sym] = df
        results.append(analyse(sym, df, round_trip))
        print(f"  {sym}: {len(df)} funding periods "
              f"({df.index[0].date()} to {df.index[-1].date()})")

    if not results:
        print("\n[ERROR] No funding data retrieved.", file=sys.stderr)
        sys.exit(1)

    res = pd.DataFrame(results)
    print(f"\n{'='*104}")
    print("FUNDING RATE SUMMARY")
    print(f"{'='*104}")
    print(f"{'symbol':<11}{'days':>7}{'mean 8h':>10}{'gross/yr':>11}{'median 8h':>11}"
          f"{'neg %':>8}{'worst run':>11}{'cum dd':>9}{'b/e days':>10}")
    for _, r in res.iterrows():
        print(f"{r['symbol']:<11}{r['days']:>7.0f}{r['mean_8h_%']:>9.4f}%{r['gross_annual_%']:>10.2f}%"
              f"{r['median_8h_%']:>10.4f}%{r['neg_share_%']:>7.1f}%{r['longest_neg_run']:>11.0f}"
              f"{r['worst_cum_dd_%']:>8.2f}%{r['breakeven_days']:>10.1f}")

    print(f"\n{'='*104}")
    print("NET ANNUALISED YIELD BY HOLDING PERIOD  (after opening and closing both legs)")
    print(f"{'='*104}")
    holds = [7, 14, 30, 90, 180, 365]
    print(f"{'symbol':<11}" + "".join(f"{f'{h}d':>13}" for h in holds))
    for sym, df in frames.items():
        print(f"{sym:<11}" + "".join(f"{net_yield(df, h, round_trip):>12.2f}%" for h in holds))

    print("\nInterpretation")
    print("  - gross/yr is what the funding stream pays before any costs.")
    print("  - neg % and worst run show how often, and for how long, you PAY instead.")
    print("  - b/e days is how long you must hold before funding covers the round trip;")
    print("    anything shorter loses money no matter what funding does.")
    print("  - Not modelled: short-leg liquidation risk, basis moves at entry/exit,")
    print("    spot borrow costs, exchange risk. These numbers are an upper bound.")

    res.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
