"""
Parameter sweep: tests multiple EMA combinations on one or more CSV files.

Usage:
    python sweep.py --csv data/BTCUSDT_15m.csv data/BTCUSDT_1h.csv
    python sweep.py --csv data/BTCUSDT_1h.csv --sort-by sharpe

For each (csv, fast_ema, slow_ema) combination it runs a full backtest and prints
a ranked results table. Default sort: Sharpe ratio (most risk-adjusted).
"""
import argparse

import pandas as pd

from config import CONFIG
from backtest.run import run_backtest


# ── Sweep grid ─────────────────────────────────────────────────────────────────
EMA_COMBOS = [
    (21,  55),   # original (baseline)
    (30,  90),
    (50, 100),
    (50, 150),
    (50, 200),
    (75, 200),
    (100, 200),
]
# ───────────────────────────────────────────────────────────────────────────────



def _calc_metrics(df, equity):
    eq_df, trades_df = run_backtest(df, starting_equity=equity)

    final_equity = eq_df["equity"].iloc[-1] if len(eq_df) else equity
    total_return = (final_equity / equity - 1) * 100

    running_max = eq_df["equity"].cummax()
    drawdown    = (eq_df["equity"] - running_max) / running_max
    max_dd      = drawdown.min() * 100 if len(drawdown) else 0.0

    exits = trades_df[trades_df["action"] == "EXIT"] if len(trades_df) else trades_df
    has_pnl = "pnl" in exits.columns and len(exits) > 0
    n_trades = len(exits)
    win_rate = (exits["pnl"] > 0).mean() * 100 if has_pnl else 0.0

    wins   = exits.loc[exits["pnl"] > 0, "pnl"] if has_pnl else pd.Series([], dtype=float)
    losses = exits.loc[exits["pnl"] <= 0, "pnl"] if has_pnl else pd.Series([], dtype=float)
    avg_win  = wins.mean()  if len(wins)   else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    gross_win  = wins.sum()        if len(wins)   else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    if has_pnl and n_trades >= 2:
        pnl_series  = exits["pnl"]
        avg_dur_h   = exits["duration_h"].dropna().mean() \
                      if "duration_h" in exits.columns else None
        tpy         = 8760 / avg_dur_h if avg_dur_h and avg_dur_h > 0 else n_trades
        std_pnl     = pnl_series.std(ddof=1)
        sharpe      = (pnl_series.mean() / std_pnl * (tpy ** 0.5)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    calmar = total_return / abs(max_dd) if max_dd != 0 else float("inf")

    return {
        "return_%":      round(total_return, 2),
        "max_dd_%":      round(max_dd, 2),
        "sharpe":        round(sharpe, 3),
        "calmar":        round(calmar, 3),
        "profit_factor": round(profit_factor, 3),
        "n_trades":      n_trades,
        "win_rate_%":    round(win_rate, 1),
    }

def sweep_one_csv(csv_path: str, equity: float = 10_000.0) -> list[dict]:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 100:
        print(f"Not enough data in {csv_path}")
        return []

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train (IS):  {len(train_df)} candles ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"Test  (OOS): {len(test_df)} candles ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    results = []

    for fast, slow in EMA_COMBOS:
        CONFIG.strategy.fast_ema = fast
        CONFIG.strategy.slow_ema = slow

        train_stats = _calc_metrics(train_df, equity)
        test_stats = _calc_metrics(test_df, equity)

        results.append({
            "csv":           csv_path.split("\\")[-1].split("/")[-1],
            "fast_ema":      fast,
            "slow_ema":      slow,
            # In-Sample (IS)
            "IS_return":     train_stats["return_%"],
            "IS_sharpe":     train_stats["sharpe"],
            "IS_calmar":     train_stats["calmar"],
            "IS_pf":         train_stats["profit_factor"],
            "IS_trades":     train_stats["n_trades"],
            # Out-Of-Sample (OOS)
            "OOS_return":    test_stats["return_%"],
            "OOS_sharpe":    test_stats["sharpe"],
            "OOS_calmar":    test_stats["calmar"],
            "OOS_pf":        test_stats["profit_factor"],
            "OOS_trades":    test_stats["n_trades"],
        })
        print(f"  EMA({fast}/{slow}) -> IS: ret={train_stats['return_%']:.1f}% shp={train_stats['sharpe']:.2f} | "
              f"OOS: ret={test_stats['return_%']:.1f}% shp={test_stats['sharpe']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV files to test")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument(
        "--sort-by", dest="sort_by", default="IS_sharpe",
        help="Column to rank results by (default: IS_sharpe)",
    )
    args = parser.parse_args()

    sort_col = args.sort_by

    all_results = []
    for csv_path in args.csv:
        print(f"\n{'='*64}")
        print(f"Sweeping: {csv_path}")
        print(f"{'='*64}")
        all_results.extend(sweep_one_csv(csv_path, equity=args.equity))

    results_df = pd.DataFrame(all_results).sort_values(sort_col, ascending=False)

    print(f"\n{'='*64}")
    print(f"RANKED RESULTS (by {args.sort_by})")
    print(f"{'='*64}")
    print(results_df.to_string(index=False))

    results_df.to_csv("sweep_results.csv", index=False)
    print("\nSaved: sweep_results.csv")


if __name__ == "__main__":
    main()
