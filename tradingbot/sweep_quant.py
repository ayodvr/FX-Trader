"""
Parameter sweep for the Quant Scanner strategy, with walk-forward validation.

Answers one question: does ANY configuration of this strategy clear real
transaction costs out-of-sample?

Every combination is run twice -- once on the first 70% of the data (in-sample)
and once on the held-out final 30% (out-of-sample). A configuration that only
looks good in-sample is curve fit, and the out-of-sample column is the one that
matters. Sweeping parameters and then reporting the best in-sample result is how
strategies get talked into production; the split is here to make that harder.

Usage:
    python sweep_quant.py --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT
    python sweep_quant.py --symbols BTCUSDT --sort-by oos_return --top 20
    python sweep_quant.py --symbols BTCUSDT --zero-cost   # isolate raw edge
"""
import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import CONFIG
from backtest.run_quant import btc_regime_series, load_or_fetch, run_quant_backtest

# Grid. Kept deliberately small -- a wider grid on 180 days of data finds
# noise, not edge.
GRID = {
    "min_volume_spike":    [1.0, 1.5, 2.0],
    "quant_stop_atr_mult": [1.0, 1.5, 2.5],
    "quant_tp1_r":         [1.5, 2.5],
    "quant_tp2_r":         [3.0, 5.0],
    "max_hold_hours":      [3.0, 12.0],
}


def _metrics(eq_df, trades_df, starting_equity: float) -> dict:
    final = eq_df["equity"].iloc[-1] if len(eq_df) else starting_equity
    ret = (final / starting_equity - 1) * 100
    running_max = eq_df["equity"].cummax()
    dd = ((eq_df["equity"] - running_max) / running_max).min() * 100 if len(eq_df) else 0.0

    exits = trades_df[trades_df["action"] == "EXIT"] if len(trades_df) else pd.DataFrame()
    n = len(exits)
    if n == 0:
        return {"return_%": ret, "max_dd_%": dd, "pf": 0.0, "n_trades": 0, "win_%": 0.0}
    pnl = exits["pnl"]
    gw, gl = pnl[pnl > 0].sum(), abs(pnl[pnl <= 0].sum())
    return {
        "return_%":  ret,
        "max_dd_%":  dd,
        "pf":        (gw / gl) if gl > 0 else float("inf"),
        "n_trades":  n,
        "win_%":     (pnl > 0).mean() * 100,
    }


def split_data(symbol_data: dict, frac: float = 0.7):
    """Split every symbol at the same wall-clock time, not per-symbol row count."""
    all_idx = sorted(set().union(*(df.index for df in symbol_data.values())))
    cut = all_idx[int(len(all_idx) * frac)]
    is_data  = {s: df[df.index <= cut] for s, df in symbol_data.items()}
    oos_data = {s: df[df.index > cut]  for s, df in symbol_data.items()}
    return is_data, oos_data, cut


def main():
    p = argparse.ArgumentParser(description="Sweep Quant Scanner parameters with walk-forward split")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--fee", type=float, default=0.00055)
    p.add_argument("--funding", type=float, default=0.0001)
    p.add_argument("--zero-cost", action="store_true", help="Run with no fees/slippage/funding")
    p.add_argument("--sort-by", default="oos_return_%",
                   choices=["oos_return_%", "oos_pf", "is_return_%", "is_pf"])
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="sweep_quant_results.csv")
    args = p.parse_args()

    fee      = 0.0 if args.zero_cost else args.fee
    slippage = 0.0 if args.zero_cost else args.slippage
    funding  = 0.0 if args.zero_cost else args.funding

    data_dir = Path(args.data_dir)
    symbol_data = {}
    for sym in args.symbols:
        df = load_or_fetch(sym, "15", args.days, data_dir, allow_fetch=True)
        if df is not None and len(df) >= 60:
            symbol_data[sym] = df
    if not symbol_data:
        print("[ERROR] No usable symbol data.", file=sys.stderr)
        sys.exit(1)

    btc = load_or_fetch("BTCUSDT", "60", args.days, data_dir, allow_fetch=True)
    if btc is None:
        print("[ERROR] BTCUSDT 60m data is required for the regime filter.", file=sys.stderr)
        sys.exit(1)
    regime = btc_regime_series(btc)

    is_data, oos_data, cut = split_data(symbol_data, 0.7)
    print(f"Symbols: {', '.join(symbol_data)}")
    print(f"Walk-forward split at {cut}  (70% in-sample / 30% out-of-sample)")
    print(f"Costs: fee {fee*100:.4f}% | slippage {slippage*100:.3f}% | funding {funding*100:.4f}%/8h"
          + ("   [ZERO-COST MODE]" if args.zero_cost else ""))

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"Testing {len(combos)} combinations...\n")

    rows = []
    for n, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        max_hold = params.pop("max_hold_hours")
        cfg = replace(CONFIG.strategy, **params)

        try:
            run_kwargs = dict(
                starting_equity=args.equity, fee_rate=fee, slippage_pct=slippage,
                funding_rate_8h=funding, max_hold_hours=max_hold, cfg=cfg, quiet=True,
            )
            is_eq, is_tr   = run_quant_backtest(is_data, regime, **run_kwargs)
            oos_eq, oos_tr = run_quant_backtest(oos_data, regime, **run_kwargs)
        except Exception as e:
            print(f"  [{n}/{len(combos)}] failed: {e}")
            continue

        m_is, m_oos = _metrics(is_eq, is_tr, args.equity), _metrics(oos_eq, oos_tr, args.equity)
        rows.append({
            **params, "max_hold_hours": max_hold,
            "is_return_%":  round(m_is["return_%"], 2),
            "is_pf":        round(m_is["pf"], 3),
            "is_trades":    m_is["n_trades"],
            "oos_return_%": round(m_oos["return_%"], 2),
            "oos_pf":       round(m_oos["pf"], 3),
            "oos_max_dd_%": round(m_oos["max_dd_%"], 2),
            "oos_trades":   m_oos["n_trades"],
            "oos_win_%":    round(m_oos["win_%"], 1),
        })
        print(f"  [{n}/{len(combos)}] vol>={params['min_volume_spike']} "
              f"stop={params['quant_stop_atr_mult']} tp1={params['quant_tp1_r']}R "
              f"tp2={params['quant_tp2_r']}R hold={max_hold}h  ->  "
              f"IS {m_is['return_%']:+7.2f}% (pf {m_is['pf']:.2f}) | "
              f"OOS {m_oos['return_%']:+7.2f}% (pf {m_oos['pf']:.2f}, {m_oos['n_trades']} trades)")

    if not rows:
        print("\nNo results.")
        return

    df = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*100}")
    print(f"TOP {args.top} BY {args.sort_by}")
    print(f"{'='*100}")
    cols = ["min_volume_spike", "quant_stop_atr_mult", "quant_tp1_r", "quant_tp2_r",
            "max_hold_hours", "is_return_%", "is_pf", "oos_return_%", "oos_pf",
            "oos_max_dd_%", "oos_trades"]
    print(df[cols].head(args.top).to_string(index=False))

    profitable = df[(df["oos_return_%"] > 0) & (df["oos_trades"] >= 20)]
    print(f"\n{len(profitable)} of {len(df)} configurations were profitable out-of-sample "
          f"with at least 20 trades.")
    if len(profitable) == 0:
        print("No configuration survived out-of-sample. The strategy does not have an\n"
              "edge that clears these costs -- this is a strategy problem, not a tuning one.")
    else:
        best = profitable.iloc[0]
        print(f"Best out-of-sample: {best['oos_return_%']:+.2f}% (pf {best['oos_pf']:.2f}) "
              f"vs in-sample {best['is_return_%']:+.2f}%")
        print("Check the two columns agree before trusting it -- a large gap means curve fit.")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
