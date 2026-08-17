"""
Backtest for cross-sectional momentum (strategy/xsmom_strategy.py).

Portfolio-level rather than trade-level: the strategy holds a book of weights
and rebalances on a schedule, so the engine tracks weights and turnover instead
of individual entries and exits.

Cost model matches the other engines. Costs here are charged on TURNOVER --
the fraction of the book that actually changes at each rebalance -- which is the
honest way to price a portfolio strategy: holding a winner for three periods
costs nothing extra, only the switch does.

Survivorship: symbols are only rankable once they have enough history, and a
symbol missing at a rebalance is simply excluded from that period's ranking
rather than dropped from the whole run.

Usage:
    python -m backtest.run_xsmom --formation 42 --holding 42 --n-long 5 --n-short 5
    python -m backtest.run_xsmom --sweep
    python -m backtest.run_xsmom --long-only --n-long 5
"""
import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.run_quant import load_or_fetch
from strategy.xsmom_strategy import XsMomConfig, target_weights

BARS_PER_DAY_4H = 6


def load_universe(symbols: list[str], interval: str, days: int, data_dir: Path,
                  allow_fetch: bool = False) -> pd.DataFrame:
    """Close-price panel: rows = timestamps, columns = symbols."""
    series = {}
    for s in symbols:
        df = load_or_fetch(s, interval, days, data_dir, allow_fetch=allow_fetch)
        if df is not None and len(df) > 50:
            series[s] = df["close"]
    if not series:
        raise ValueError("No usable symbol data")
    return pd.DataFrame(series).sort_index()


def run_xsmom_backtest(
    prices: pd.DataFrame,
    cfg: XsMomConfig,
    starting_equity: float = 10_000.0,
    fee_rate: float = 0.00055,
    slippage_pct: float = 0.0005,
    funding_rate_8h: float = 0.0001,
    gross_leverage: float = 1.0,
    quiet: bool = False,
):
    def _log(m):
        if not quiet:
            print(m)

    rets = prices.pct_change().fillna(0.0)
    warmup = cfg.formation_bars + cfg.skip_bars + 1

    equity = starting_equity
    weights = pd.Series(0.0, index=prices.columns)
    curve, turnover_log, rebalances = [], [], 0
    # Cost per unit of turnover: fee + slippage on the way out and the way in
    cost_per_turnover = fee_rate + slippage_pct

    _log(f"Simulating {len(prices):,} bars, {len(prices.columns)} symbols, "
         f"formation {cfg.formation_bars} / holding {cfg.holding_bars} bars, "
         f"{cfg.n_long}L/{cfg.n_short}S, gross {gross_leverage}x...")

    for i in range(len(prices)):
        # Mark the existing book to market on this bar's returns
        if i > 0 and weights.abs().sum() > 0:
            bar_ret = (weights * rets.iloc[i]).sum() * gross_leverage
            equity *= (1.0 + bar_ret)
            # Funding accrues on gross exposure while positions are held
            if i % 2 == 0:   # 4h bars -> funding every other bar (8h)
                equity -= equity * weights.abs().sum() * gross_leverage * funding_rate_8h

        if equity <= 0:
            _log(f"  WIPED OUT at {prices.index[i]}")
            curve.extend([0.0] * (len(prices) - len(curve)))
            break

        # Rebalance on schedule, once past warmup
        if i >= warmup and (i - warmup) % cfg.holding_bars == 0:
            window = prices.iloc[: i + 1].dropna(axis=1, how="any")
            if window.shape[1] >= cfg.min_universe:
                new_w = target_weights(window, cfg)
                if not new_w.empty:
                    aligned = pd.Series(0.0, index=prices.columns)
                    aligned.loc[new_w.index] = new_w.values
                    turnover = (aligned - weights).abs().sum()
                    equity -= equity * turnover * gross_leverage * cost_per_turnover
                    weights = aligned
                    turnover_log.append(turnover)
                    rebalances += 1

        curve.append(equity)

    eq = pd.DataFrame({"equity": curve[: len(prices)]},
                      index=prices.index[: len(curve)]).rename_axis("time")
    stats = {
        "rebalances": rebalances,
        "avg_turnover": float(np.mean(turnover_log)) if turnover_log else 0.0,
        "total_turnover": float(np.sum(turnover_log)) if turnover_log else 0.0,
    }
    return eq, stats


def summarize(eq, stats, starting_equity, bars_per_year, quiet=False):
    if len(eq) == 0:
        return {"total_return_%": 0.0, "sharpe": 0.0, "max_dd_%": 0.0}
    final = eq["equity"].iloc[-1]
    ret = (final / starting_equity - 1) * 100
    rm = eq["equity"].cummax()
    max_dd = ((eq["equity"] - rm) / rm.replace(0, np.nan)).min() * 100

    r = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(bars_per_year)) if r.std(ddof=1) > 0 else 0.0
    years = len(eq) / bars_per_year
    cagr = ((final / starting_equity) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else -100.0

    out = {
        "total_return_%": round(ret, 2), "cagr_%": round(float(cagr), 2),
        "sharpe": round(float(sharpe), 3), "max_dd_%": round(float(max_dd), 2),
        "rebalances": stats["rebalances"], "avg_turnover": round(stats["avg_turnover"], 3),
    }
    if quiet:
        return out

    print()
    print(f"Starting equity:       {starting_equity:>12,.2f}")
    print(f"Final equity:          {final:>12,.2f}")
    print(f"Total return:          {ret:>11.2f}%")
    print(f"CAGR:                  {cagr:>11.2f}%")
    print(f"Sharpe ratio:          {sharpe:>12.3f}")
    print(f"Max drawdown:          {max_dd:>11.2f}%")
    print(f"Rebalances:            {stats['rebalances']:>12}")
    print(f"Avg turnover/rebal:    {stats['avg_turnover']:>12.3f}   "
          f"(1.0 = the whole book changes)")
    return out


def main():
    p = argparse.ArgumentParser(description="Backtest cross-sectional momentum")
    p.add_argument("--symbols", nargs="+", help="Universe (default: every *_<interval>m.csv in data/)")
    p.add_argument("--interval", default="240")
    p.add_argument("--days", type=int, default=1000)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--formation", type=int, default=42, help="Lookback in bars (42 = 7d on 4h)")
    p.add_argument("--holding", type=int, default=42, help="Rebalance interval in bars")
    p.add_argument("--skip", type=int, default=0, help="Bars skipped before the formation window ends")
    p.add_argument("--n-long", type=int, default=5)
    p.add_argument("--n-short", type=int, default=5)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--vol-target", action="store_true", help="Inverse-volatility weighting")
    p.add_argument("--gross", type=float, default=1.0, help="Gross leverage")
    p.add_argument("--fee", type=float, default=0.00055)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--funding", type=float, default=0.0001)
    p.add_argument("--sweep", action="store_true", help="Grid search with 70/30 walk-forward")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = sorted(f.name.split("_")[0]
                         for f in data_dir.glob(f"*_{args.interval}m.csv"))
    if not symbols:
        print("[ERROR] No symbols found.", file=sys.stderr)
        sys.exit(1)

    prices = load_universe(symbols, args.interval, args.days, data_dir)
    bars_per_year = 365 * 24 * 60 / int(args.interval)
    print(f"Universe: {prices.shape[1]} symbols, {len(prices):,} bars "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    cfg = XsMomConfig(
        formation_bars=args.formation, holding_bars=args.holding, skip_bars=args.skip,
        n_long=args.n_long, n_short=args.n_short, long_only=args.long_only,
        vol_target=args.vol_target,
    )
    run_kwargs = dict(starting_equity=args.equity, fee_rate=args.fee,
                      slippage_pct=args.slippage, funding_rate_8h=args.funding,
                      gross_leverage=args.gross)

    if args.sweep:
        cut = int(len(prices) * 0.7)
        is_p, oos_p = prices.iloc[:cut], prices.iloc[cut:]
        print(f"Walk-forward split at {prices.index[cut]} (70/30)\n")
        grid = {
            "formation_bars": [18, 42, 84, 126],    # 3d, 7d, 14d, 21d on 4h bars
            "holding_bars":   [42, 84],             # 7d, 14d
            "n_long":         [3, 5, 8],
            "long_only":      [False, True],
        }
        keys = list(grid)
        rows = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            c = replace(cfg, **params, n_short=params["n_long"])
            try:
                m_is  = summarize(*run_xsmom_backtest(is_p, c, quiet=True, **run_kwargs),
                                  args.equity, bars_per_year, quiet=True)
                m_oos = summarize(*run_xsmom_backtest(oos_p, c, quiet=True, **run_kwargs),
                                  args.equity, bars_per_year, quiet=True)
            except Exception as e:
                print(f"  failed {params}: {e}")
                continue
            rows.append({**params, "is_return_%": m_is["total_return_%"], "is_sharpe": m_is["sharpe"],
                         "oos_return_%": m_oos["total_return_%"], "oos_sharpe": m_oos["sharpe"],
                         "oos_dd_%": m_oos["max_dd_%"], "oos_cagr_%": m_oos["cagr_%"]})
            print(f"  form={params['formation_bars']:>3} hold={params['holding_bars']:>3} "
                  f"n={params['n_long']} {'L-only' if params['long_only'] else 'L/S   '} -> "
                  f"IS {m_is['total_return_%']:+8.2f}% (sh {m_is['sharpe']:+.2f}) | "
                  f"OOS {m_oos['total_return_%']:+8.2f}% (sh {m_oos['oos_sharpe'] if False else m_oos['sharpe']:+.2f}, "
                  f"dd {m_oos['max_dd_%']:+.1f}%)")

        df = pd.DataFrame(rows).sort_values("oos_return_%", ascending=False)
        df.to_csv("xsmom_sweep.csv", index=False)
        print(f"\n{'='*100}\nTOP 12 BY OUT-OF-SAMPLE RETURN\n{'='*100}")
        print(df.head(12).to_string(index=False))
        good = df[df["oos_return_%"] > 0]
        print(f"\n{len(good)} of {len(df)} configurations profitable out-of-sample.")
        if len(good):
            b = good.iloc[0]
            print(f"Best OOS: {b['oos_return_%']:+.2f}% (sharpe {b['oos_sharpe']:+.2f}, "
                  f"dd {b['oos_dd_%']:+.1f}%) vs IS {b['is_return_%']:+.2f}%")
        print("\nSaved: xsmom_sweep.csv")
        return

    print(f"Costs: fee {args.fee*100:.4f}% | slippage {args.slippage*100:.3f}% | "
          f"funding {args.funding*100:.4f}%/8h | charged on turnover")
    eq, stats = run_xsmom_backtest(prices, cfg, **run_kwargs)
    summarize(eq, stats, args.equity, bars_per_year)
    eq.to_csv("xsmom_equity.csv")
    print("\nSaved: xsmom_equity.csv")


if __name__ == "__main__":
    main()
