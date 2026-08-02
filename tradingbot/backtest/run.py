"""
Backtest engine. Reuses strategy.trend_strategy and risk.risk_manager verbatim —
the same modules live.py calls — so a strategy that looks good here behaves the
same way live (mechanically; live still has slippage/latency the backtest cannot fully capture).

Realism models included:
  - **Slippage**: market orders fill at price * (1 ± slippage_pct).
    Default 0.10% per fill — adjust downward for highly liquid symbols.
  - **Funding rate**: perpetual funding is charged every 8 h (00:00/08:00/16:00 UTC).
    Default 0.01% per 8 h period on notional position value.
    Bybit publishes historical funding rates; check yours and tune accordingly.

Usage:
    python -m backtest.run --csv data/BTCUSDT_60m.csv
    python -m backtest.run --csv data/BTCUSDT_60m.csv --slippage 0.0005 --funding 0.0001
or fetch fresh data from Bybit directly (see fetch_history.py).
"""
import argparse
import pandas as pd

from config import CONFIG
from strategy.trend_strategy import generate_signal, Signal
from risk.risk_manager import RiskManager

# Funding is charged at these UTC hours every day
_FUNDING_HOURS = {0, 8, 16}


def run_backtest(
    df: pd.DataFrame,
    starting_equity: float = 10_000.0,
    fee_rate: float = 0.00055,
    slippage_pct: float = 0.001,     # 0.10% per market-order fill
    funding_rate_8h: float = 0.0001, # 0.01% every 8 h while in position
):
    cfg = CONFIG.strategy
    risk_cfg = CONFIG.risk
    risk = RiskManager(risk_cfg)

    # ── Precompute ALL indicators once (O(n) instead of O(n²)) ──────────────────────────────────
    from strategy.trend_strategy import compute_indicators
    ind = compute_indicators(df, cfg)

    equity = starting_equity
    position   = 0          # 1 long, -1 short, 0 flat
    qty        = 0.0
    entry_price = 0.0
    entry_time  = None      # timestamp of the ENTER candle (for duration calc)
    trail = None
    trades = []
    equity_curve = []

    min_lookback = max(cfg.slow_ema, cfg.atr_period) + 5

    for i in range(min_lookback, len(ind)):
        last  = ind.iloc[i]
        prev  = ind.iloc[i - 1]
        price = last["close"]
        ts    = ind.index[i]

        fast, slow, atr = last["fast_ema"], last["slow_ema"], last["atr"]
        adx = last.get("adx", 999)   # 999 = no filter if column missing

        crossed_up   = prev["fast_ema"] <= prev["slow_ema"] and fast > slow
        crossed_down = prev["fast_ema"] >= prev["slow_ema"] and fast < slow
        adx_ok = adx >= cfg.min_adx if getattr(cfg, "min_adx", 0) > 0 else True
        long_only = getattr(cfg, "long_only", False)

        # ── Funding rate deduction (charged at 00:00, 08:00, 16:00 UTC) ─────
        if position != 0 and hasattr(ts, "hour") and ts.hour in _FUNDING_HOURS:
            # Check we haven't already charged this candle's funding this session
            # (only once per boundary hour — the candle timestamp IS the boundary)
            funding_cost = qty * price * funding_rate_8h
            equity -= funding_cost

        # ── Entry ────────────────────────────────────────────────────────────
        if position == 0:
            if crossed_up and adx_ok:
                stop = price - atr * cfg.atr_stop_mult
                sizing = risk.size_position(equity, price, stop, open_positions=0, now=ts)
                if sizing.approved:
                    position = 1
                    qty = sizing.qty
                    fill_price = price * (1 + slippage_pct)  # long: pay slightly more
                    entry_price = fill_price
                    entry_time  = ts
                    trail = stop
                    fee = qty * fill_price * fee_rate
                    equity -= fee
                    trades.append({"time": ts, "action": "ENTER", "side": "long",
                                   "price": fill_price, "qty": qty, "fee": fee})
            elif crossed_down and adx_ok and not long_only:
                stop = price + atr * cfg.atr_stop_mult
                sizing = risk.size_position(equity, price, stop, open_positions=0, now=ts)
                if sizing.approved:
                    position = -1
                    qty = sizing.qty
                    fill_price = price * (1 - slippage_pct)  # short: sell slightly less
                    entry_price = fill_price
                    entry_time  = ts
                    trail = stop
                    fee = qty * fill_price * fee_rate
                    equity -= fee
                    trades.append({"time": ts, "action": "ENTER", "side": "short",
                                   "price": fill_price, "qty": qty, "fee": fee})

        # ── In position ──────────────────────────────────────────────────────
        elif position == 1:
            new_trail = price - atr * cfg.atr_trail_mult
            trail = max(trail or new_trail, new_trail)
            if crossed_down or price <= trail:
                fill_price = price * (1 - slippage_pct)  # exit long: receive slightly less
                pnl = (fill_price - entry_price) * qty
                fee = qty * fill_price * fee_rate
                equity += pnl - fee
                risk.record_realized_pnl(pnl - fee, equity, now=ts)
                dur_h = (ts - entry_time).total_seconds() / 3600 if entry_time else None
                trades.append({"time": ts, "action": "EXIT", "price": fill_price,
                               "qty": qty, "pnl": pnl - fee, "duration_h": dur_h})
                position, qty, entry_price, entry_time, trail = 0, 0.0, 0.0, None, None

        elif position == -1:
            new_trail = price + atr * cfg.atr_trail_mult
            trail = min(trail or new_trail, new_trail)
            if crossed_up or price >= trail:
                fill_price = price * (1 + slippage_pct)  # exit short: buy back slightly more
                pnl = (entry_price - fill_price) * qty
                fee = qty * fill_price * fee_rate
                equity += pnl - fee
                risk.record_realized_pnl(pnl - fee, equity, now=ts)
                dur_h = (ts - entry_time).total_seconds() / 3600 if entry_time else None
                trades.append({"time": ts, "action": "EXIT", "price": fill_price,
                               "qty": qty, "pnl": pnl - fee, "duration_h": dur_h})
                position, qty, entry_price, entry_time, trail = 0, 0.0, 0.0, None, None

        equity_curve.append({"time": ts, "equity": equity})

    eq_df = pd.DataFrame(equity_curve).set_index("time")
    trades_df = pd.DataFrame(trades)
    return eq_df, trades_df


def summarize(eq_df: pd.DataFrame, trades_df: pd.DataFrame, starting_equity: float) -> dict:
    """Print and return a dict of backtest performance metrics."""
    import numpy as np

    final_equity = eq_df["equity"].iloc[-1] if len(eq_df) else starting_equity
    total_return = (final_equity / starting_equity - 1) * 100
    running_max  = eq_df["equity"].cummax()
    drawdown     = (eq_df["equity"] - running_max) / running_max
    max_dd       = drawdown.min() * 100 if len(drawdown) else 0.0

    exits = trades_df[trades_df["action"] == "EXIT"] if len(trades_df) else trades_df
    # Guard: EXIT rows may not exist or may not have a pnl column yet
    has_pnl = "pnl" in exits.columns and len(exits) > 0
    n_trades  = len(exits)
    win_rate  = (exits["pnl"] > 0).mean() * 100 if has_pnl else 0.0

    wins  = exits.loc[exits["pnl"] > 0, "pnl"] if has_pnl else pd.Series([], dtype=float)
    losses = exits.loc[exits["pnl"] <= 0, "pnl"] if has_pnl else pd.Series([], dtype=float)
    avg_win  = wins.mean()   if len(wins)   else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    gross_win  = wins.sum()         if len(wins)   else 0.0
    gross_loss = abs(losses.sum())  if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualised, using per-trade PnL as return stream)
    if has_pnl and n_trades >= 2:
        pnl_series = exits["pnl"]
        # Annualise: assume each trade represents (avg_duration / 8760) of a year
        avg_dur_h  = exits["duration_h"].dropna().mean() if "duration_h" in exits.columns else None
        trades_per_year = 8760 / avg_dur_h if avg_dur_h and avg_dur_h > 0 else n_trades
        mean_pnl = pnl_series.mean()
        std_pnl  = pnl_series.std(ddof=1)
        sharpe = (mean_pnl / std_pnl * (trades_per_year ** 0.5)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0
        avg_dur_h = None

    # Max consecutive losses
    max_consec_loss = 0
    cur = 0
    for pnl in (exits["pnl"] if has_pnl else []):
        if pnl <= 0:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0

    calmar = total_return / abs(max_dd) if max_dd != 0 else float("inf")
    avg_dur_display = f"{avg_dur_h:.1f}h" if avg_dur_h is not None else "N/A"

    print(f"Starting equity:       {starting_equity:>12,.2f}")
    print(f"Final equity:          {final_equity:>12,.2f}")
    print(f"Total return:          {total_return:>11.2f}%")
    print(f"Max drawdown:          {max_dd:>11.2f}%")
    print(f"Sharpe ratio:          {sharpe:>12.3f}")
    print(f"Calmar ratio:          {calmar:>12.3f}")
    print(f"Profit factor:         {profit_factor:>12.3f}")
    print(f"Number of trades:      {n_trades:>12}")
    print(f"Win rate:              {win_rate:>11.1f}%")
    print(f"Avg win:               {avg_win:>12.2f}")
    print(f"Avg loss:              {avg_loss:>12.2f}")
    print(f"Max consec. losses:    {max_consec_loss:>12}")
    print(f"Avg trade duration:    {avg_dur_display:>12}")

    return {
        "total_return_%":    round(total_return, 3),
        "max_dd_%":          round(max_dd, 3),
        "sharpe":            round(sharpe, 4),
        "calmar":            round(calmar, 4),
        "profit_factor":     round(profit_factor, 4),
        "n_trades":          n_trades,
        "win_rate_%":        round(win_rate, 2),
        "avg_win_$":         round(avg_win, 4),
        "avg_loss_$":        round(avg_loss, 4),
        "max_consec_loss":   max_consec_loss,
        "avg_duration_h":    round(avg_dur_h, 2) if avg_dur_h is not None else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      required=True, help="CSV with columns: timestamp,open,high,low,close,volume")
    parser.add_argument("--equity",   type=float, default=10_000.0)
    parser.add_argument("--slippage", type=float, default=0.001,    help="Market order slippage fraction (default 0.001 = 0.1%)")
    parser.add_argument("--funding",  type=float, default=0.0001,   help="Funding rate per 8 h (default 0.0001 = 0.01%)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    print(f"Slippage model: {args.slippage*100:.3f}% per fill | "
          f"Funding rate:   {args.funding*100:.4f}% per 8h")
    eq_df, trades_df = run_backtest(
        df,
        starting_equity=args.equity,
        slippage_pct=args.slippage,
        funding_rate_8h=args.funding,
    )
    summarize(eq_df, trades_df, args.equity)
    trades_df.to_csv("backtest_trades.csv", index=False)
    eq_df.to_csv("backtest_equity_curve.csv")
    print("\nSaved: backtest_trades.csv, backtest_equity_curve.csv")
