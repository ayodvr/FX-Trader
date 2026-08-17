"""
Cross-sectional momentum — rank the universe, hold the winners, short the losers.

This is the one strategy family in this repo that comes from published research
rather than from the repo's own history. Cross-sectional momentum is a
documented cryptocurrency anomaly: rank every coin by its return over a
formation window, go long the top slice and short the bottom slice, rebalance on
a fixed schedule. Reported figures in the literature sit around 1.2-1.7% average
weekly return with Sharpe ratios near 1.1-1.5.

Why it is worth testing here specifically, given everything that failed before:

  - It is RELATIVE, not directional. The signal is "which coin outperformed
    which", so it does not need to forecast whether the market goes up. That
    sidesteps the problem that beat every previous strategy.
  - Turnover is low by construction. Weekly rebalancing across a fixed universe
    means a handful of trades per week, not thousands. Transaction costs are the
    thing that killed the scanner and the breakout system; this design attacks
    that directly.
  - Long and short legs are held simultaneously, so a broad market move largely
    cancels between them.

Known caveats from the same literature, which the backtest must respect:
  - Signal decay is faster in crypto than in equities, so the formation and
    holding windows matter more.
  - Costs on small illiquid tokens can be punitive; the universe should stay
    liquid.
  - Momentum crashes: the strategy is short the worst performers, which is
    exactly the basket that rips hardest on a market-wide reversal.

Pure logic, same contract as the other strategy modules: prices in, target
weights out. No exchange or state awareness.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class XsMomConfig:
    formation_bars: int = 42        # lookback for the ranking signal (in bars)
    holding_bars: int = 42          # rebalance interval (in bars)
    n_long: int = 5                 # coins in the long leg
    n_short: int = 5                # coins in the short leg (0 = long-only)
    skip_bars: int = 0              # skip most-recent bars to avoid short-term reversal
    min_universe: int = 10          # refuse to rank a universe smaller than this
    vol_target: bool = False        # scale each leg by inverse volatility
    vol_lookback: int = 42
    long_only: bool = False


def compute_momentum(prices: pd.DataFrame, cfg: XsMomConfig) -> pd.Series:
    """
    Formation-period return per symbol, as of the last row of `prices`.

    With skip_bars > 0 the most recent bars are excluded from the measurement --
    the standard "12-1" construction from the equity literature, which avoids
    contaminating a momentum signal with short-term reversal.
    """
    end = len(prices) - cfg.skip_bars
    start = end - cfg.formation_bars
    if start < 0 or end <= start:
        return pd.Series(dtype=float)

    window = prices.iloc[start:end]
    first, last = window.iloc[0], window.iloc[-1]
    mom = (last / first) - 1.0
    return mom.replace([np.inf, -np.inf], np.nan).dropna()


def inverse_vol_weights(returns: pd.DataFrame, symbols: list[str], lookback: int) -> pd.Series:
    """Weights proportional to 1/volatility, normalised to sum to 1."""
    if not symbols:
        return pd.Series(dtype=float)
    vol = returns[symbols].tail(lookback).std()
    vol = vol.replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty or inv.sum() == 0:
        return pd.Series(1.0 / len(symbols), index=symbols)
    return inv / inv.sum()


def target_weights(prices: pd.DataFrame, cfg: XsMomConfig) -> pd.Series:
    """
    Target portfolio weights as of the last row of `prices`.

    Positive = long, negative = short. Each leg sums to 1 (or -1), so the book is
    roughly dollar-neutral when both legs are active. Returns an empty Series if
    the universe is too small to rank meaningfully.
    """
    mom = compute_momentum(prices, cfg)
    if len(mom) < cfg.min_universe:
        return pd.Series(dtype=float)

    ranked = mom.sort_values(ascending=False)
    n_long = min(cfg.n_long, len(ranked) // 2)
    n_short = 0 if cfg.long_only else min(cfg.n_short, len(ranked) // 2)
    if n_long <= 0:
        return pd.Series(dtype=float)

    longs = list(ranked.index[:n_long])
    shorts = list(ranked.index[-n_short:]) if n_short > 0 else []

    weights = pd.Series(0.0, index=mom.index)
    if cfg.vol_target:
        rets = prices.pct_change()
        lw = inverse_vol_weights(rets, longs, cfg.vol_lookback)
        weights.loc[lw.index] = lw.values
        if shorts:
            sw = inverse_vol_weights(rets, shorts, cfg.vol_lookback)
            weights.loc[sw.index] = -sw.values
    else:
        weights.loc[longs] = 1.0 / len(longs)
        if shorts:
            weights.loc[shorts] = -1.0 / len(shorts)

    return weights[weights != 0.0]
