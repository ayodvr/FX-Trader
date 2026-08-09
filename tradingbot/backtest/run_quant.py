"""
Portfolio backtest for the Quant Scanner strategy (strategy/quant_strategy.py).

This is the counterpart to backtest/run.py, which backtests the *other* strategy
(strategy/trend_strategy.py, used by live/run.py). The scanner is a different
animal and needs a different engine:

  - Multi-symbol: it watches the top-N by volume and can hold several positions
    at once, capped by ScannerConfig.max_active_trades.
  - Staged exits: TP1 scales out 50% and moves the stop to breakeven, TP2 closes
    the runner, plus a hard max-hold timeout.
  - Regime filter: entries are gated on BTC's 1h trend.

Fidelity notes (what this does and does not claim):
  - Intra-candle fills. The live scanner places real exchange stop orders and
    polls every 30s, so stops and targets fire *within* a candle, not only at
    its close. Hits are therefore tested against each candle's high/low.
  - Worst-case ordering. If a candle's range covers both the stop and a target,
    the stop is taken. Without tick data there is no way to know which came
    first, so the pessimistic branch is assumed.
  - BTC regime uses the last fully-closed 1h candle before each bar. Live reads
    the still-forming candle; using it here would be lookahead.
  - Slippage and fees are charged on every fill, funding every 8h on open
    notional. Defaults are the same as backtest/run.py.
  - Signal rules come from strategy.quant_strategy.evaluate_quant_signal --
    the same code the live scanner runs, not a reimplementation.

Usage:
    python -m backtest.run_quant --symbols BTCUSDT ETHUSDT SOLUSDT --days 180
    python -m backtest.run_quant --symbols-from-file symbols.txt --equity 10000
    python -m backtest.run_quant --symbols BTCUSDT --data-dir data --no-fetch
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from risk.risk_manager import RiskManager
from strategy.quant_strategy import (
    Signal,
    compute_quant_indicators,
    evaluate_quant_signal,
)

_FUNDING_HOURS = {0, 8, 16}
_TIMEFRAME_MIN = 15


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_or_fetch(symbol: str, interval: str, days: int, data_dir: Path,
                  allow_fetch: bool = True) -> pd.DataFrame | None:
    """Load <data_dir>/<SYMBOL>_<interval>m.csv, fetching it from Bybit if absent."""
    path = data_dir / f"{symbol}_{interval}m.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    if not allow_fetch:
        print(f"  [skip] {symbol}: {path} not found and --no-fetch is set")
        return None

    try:
        from fetch_history import fetch_history
        data_dir.mkdir(exist_ok=True)
        df = fetch_history(symbol, interval, days)
        df.to_csv(path)
        return df
    except Exception as e:
        print(f"  [skip] {symbol}: could not fetch history ({e})")
        return None


def btc_regime_series(df_btc_1h: pd.DataFrame) -> pd.Series:
    """
    Precompute BTC's regime (+1/-1/0) per 1h candle, mirroring
    strategy.quant_strategy.evaluate_btc_regime but vectorised.
    """
    fast = df_btc_1h["close"].ewm(span=20, adjust=False).mean()
    slow = df_btc_1h["close"].ewm(span=50, adjust=False).mean()
    regime = pd.Series(0, index=df_btc_1h.index, dtype=int)
    regime[fast > slow * 1.001] = 1
    regime[fast < slow * 0.999] = -1
    # Fewer than 30 candles of history -> evaluate_btc_regime returns 0
    regime.iloc[:30] = 0
    return regime


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

def run_quant_backtest(
    symbol_data: dict[str, pd.DataFrame],
    regime: pd.Series,
    starting_equity: float = 10_000.0,
    fee_rate: float = 0.00055,
    slippage_pct: float = 0.001,
    funding_rate_8h: float = 0.0001,
    max_active_trades: int | None = None,
    max_hold_hours: float | None = None,
    cfg=None,
    risk_cfg=None,
    quiet: bool = False,
):
    cfg = cfg if cfg is not None else CONFIG.strategy
    risk = RiskManager(risk_cfg if risk_cfg is not None else CONFIG.risk)
    max_active = max_active_trades if max_active_trades is not None else CONFIG.scanner.max_active_trades
    max_hold   = max_hold_hours   if max_hold_hours   is not None else CONFIG.scanner.max_hold_hours

    def _log(msg):
        if not quiet:
            print(msg)

    # Precompute indicators once per symbol -- the expensive part, done O(n) not O(n^2)
    _log("Computing indicators...")
    ind = {}
    for sym, df in symbol_data.items():
        if len(df) < 60:
            _log(f"  [skip] {sym}: only {len(df)} candles")
            continue
        ind[sym] = compute_quant_indicators(df, cfg)
    if not ind:
        raise ValueError("No symbols had enough data to backtest")

    # Unified, ordered timeline across every symbol
    timeline = pd.DatetimeIndex(sorted(set().union(*(df.index for df in ind.values()))))
    # Regime lookup: last CLOSED 1h candle at or before each bar
    regime_arr = regime.reindex(timeline, method="ffill").fillna(0).astype(int).to_numpy()

    # Reindex every symbol onto the shared timeline and drop to numpy. Pandas
    # .loc/.get_loc per bar dominates runtime otherwise -- this is what makes
    # the engine fast enough to sweep parameters with.
    _COLS = ("close", "high", "low", "fast_ema", "slow_ema", "trend_ema", "atr", "rsi", "rvol")
    arr: dict[str, dict] = {}
    first_valid: dict[str, int] = {}
    for sym, df_ind in ind.items():
        r = df_ind.reindex(timeline)
        cols = {c: r[c].to_numpy(dtype=float) for c in _COLS}
        cols["valid"] = r["close"].notna().to_numpy()
        arr[sym] = cols
        vidx = np.flatnonzero(cols["valid"])
        first_valid[sym] = int(vidx[0]) if len(vidx) else len(timeline)

    def row_at(sym: str, i: int) -> dict:
        a = arr[sym]
        return {c: a[c][i] for c in _COLS}

    hours   = timeline.hour.to_numpy()
    minutes = timeline.minute.to_numpy()

    equity = starting_equity
    open_trades: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve = np.empty(len(timeline), dtype=float)
    skipped_max_active = 0

    warmup = 60  # candles before a symbol is eligible, so EWMs have settled

    _log(f"Simulating {len(timeline):,} bars across {len(ind)} symbols "
         f"(max {max_active} concurrent, {max_hold}h max hold)...")

    for i in range(len(timeline)):
        ts = timeline[i]
        btc_regime = int(regime_arr[i])

        # ── Funding on open notional ─────────────────────────────────────────
        if hours[i] in _FUNDING_HOURS and minutes[i] == 0:
            for sym, tr in open_trades.items():
                if arr[sym]["valid"][i]:
                    equity -= tr["qty_open"] * arr[sym]["close"][i] * funding_rate_8h

        # ── Manage open positions ────────────────────────────────────────────
        for sym in list(open_trades.keys()):
            a = arr[sym]
            if not a["valid"][i]:
                continue
            tr   = open_trades[sym]
            high, low, close = a["high"][i], a["low"][i], a["close"][i]
            is_long = tr["side"] == "Buy"
            factor  = 1.0 if is_long else -1.0

            def _close(exit_price: float, qty: float, reason: str, slip_against: bool = True):
                """Realise `qty` at `exit_price`, charging slippage and fees."""
                nonlocal equity
                if slip_against:
                    fill = exit_price * (1 - slippage_pct) if is_long else exit_price * (1 + slippage_pct)
                else:
                    fill = exit_price
                pnl = (fill - tr["entry_price"]) * qty * factor
                fee = qty * fill * fee_rate
                equity += pnl - fee
                risk.record_realized_pnl(pnl - fee, equity, now=ts)
                trades.append({
                    "time": ts, "symbol": sym, "side": tr["side"], "action": "EXIT",
                    "reason": reason, "price": fill, "qty": qty, "pnl": pnl - fee,
                    "fee": fee,
                    "duration_h": (ts - tr["entry_time"]).total_seconds() / 3600,
                })

            # 1. Stop loss -- checked first: if the candle covers both stop and
            #    target we cannot know which hit first, so assume the stop.
            stop_hit = (low <= tr["stop_price"]) if is_long else (high >= tr["stop_price"])
            if stop_hit:
                _close(tr["stop_price"], tr["qty_open"], "SL" if not tr["tp1_hit"] else "BE")
                del open_trades[sym]
                continue

            # 2. TP1 -- scale out half, move stop to breakeven
            if not tr["tp1_hit"] and tr["tp1"] is not None:
                tp1_hit = (high >= tr["tp1"]) if is_long else (low <= tr["tp1"])
                if tp1_hit:
                    half = tr["qty_open"] * 0.5
                    _close(tr["tp1"], half, "TP1")
                    tr["qty_open"] -= half
                    tr["tp1_hit"] = True
                    tr["stop_price"] = tr["entry_price"]  # breakeven

            # 3. TP2 -- close the runner
            if tr["tp2"] is not None:
                tp2_hit = (high >= tr["tp2"]) if is_long else (low <= tr["tp2"])
                if tp2_hit:
                    _close(tr["tp2"], tr["qty_open"], "TP2")
                    del open_trades[sym]
                    continue

            # 4. Max-hold timeout
            if (ts - tr["entry_time"]).total_seconds() / 3600 >= max_hold:
                _close(close, tr["qty_open"], "TIMEOUT")
                del open_trades[sym]
                continue

            # 5. Strategy flip
            if i >= 1 and a["valid"][i - 1]:
                out = evaluate_quant_signal(
                    row_at(sym, i), row_at(sym, i - 1), cfg,
                    current_position=1 if is_long else -1,
                    current_trail=None,
                    btc_regime=btc_regime,
                )
                if out.signal == Signal.FLAT:
                    _close(close, tr["qty_open"], "SIGNAL")
                    del open_trades[sym]

        # ── Look for entries ─────────────────────────────────────────────────
        if risk.kill_switch_active(now=ts):
            equity_curve[i] = equity
            continue

        for sym in arr:
            a = arr[sym]
            if sym in open_trades or not a["valid"][i] or not a["valid"][i - 1]:
                continue
            if i - first_valid[sym] < warmup:
                continue
            if len(open_trades) >= max_active:
                skipped_max_active += 1
                break

            out = evaluate_quant_signal(
                row_at(sym, i), row_at(sym, i - 1), cfg,
                current_position=0, btc_regime=btc_regime,
            )
            if out.signal not in (Signal.LONG, Signal.SHORT):
                continue

            close = a["close"][i]
            # Scanner passes open_positions=0 -- it enforces its own concurrency
            # cap via max_active_trades rather than RiskConfig.max_open_positions.
            sizing = risk.size_position(equity, close, out.stop_price, open_positions=0, now=ts)
            if not sizing.approved:
                continue

            is_long = out.signal == Signal.LONG
            fill = close * (1 + slippage_pct) if is_long else close * (1 - slippage_pct)
            fee  = sizing.qty * fill * fee_rate
            equity -= fee
            open_trades[sym] = {
                "side": "Buy" if is_long else "Sell",
                "entry_price": fill,
                "entry_time": ts,
                "qty_open": sizing.qty,
                "qty_orig": sizing.qty,
                "stop_price": out.stop_price,
                "tp1": out.tp1_price,
                "tp2": out.tp2_price,
                "tp1_hit": False,
            }
            trades.append({
                "time": ts, "symbol": sym, "side": open_trades[sym]["side"],
                "action": "ENTER", "reason": "", "price": fill, "qty": sizing.qty,
                "pnl": None, "fee": fee, "duration_h": None,
            })

        equity_curve[i] = equity

    eq_df = pd.DataFrame({"equity": equity_curve}, index=timeline).rename_axis("time")
    trades_df = pd.DataFrame(trades)
    if skipped_max_active:
        _log(f"  ({skipped_max_active:,} signals skipped: max concurrent trades reached)")
    return eq_df, trades_df


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def summarize(eq_df: pd.DataFrame, trades_df: pd.DataFrame, starting_equity: float) -> dict:
    final_equity = eq_df["equity"].iloc[-1] if len(eq_df) else starting_equity
    total_return = (final_equity / starting_equity - 1) * 100

    running_max = eq_df["equity"].cummax()
    drawdown    = (eq_df["equity"] - running_max) / running_max
    max_dd      = drawdown.min() * 100 if len(drawdown) else 0.0

    exits = trades_df[trades_df["action"] == "EXIT"] if len(trades_df) else pd.DataFrame()
    n_trades = len(exits)
    if n_trades == 0:
        print("\nNo trades were taken.")
        return {"n_trades": 0, "total_return_%": round(total_return, 3)}

    pnl      = exits["pnl"]
    wins     = pnl[pnl > 0]
    losses   = pnl[pnl <= 0]
    win_rate = len(wins) / n_trades * 100
    gross_win, gross_loss = wins.sum(), abs(losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe from the equity curve, annualised for 15m bars
    rets = eq_df["equity"].pct_change().dropna()
    bars_per_year = 365 * 24 * 60 / _TIMEFRAME_MIN
    sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(bars_per_year)) if rets.std(ddof=1) > 0 else 0.0
    calmar = total_return / abs(max_dd) if max_dd != 0 else float("inf")

    max_consec, cur = 0, 0
    for p in pnl:
        cur = cur + 1 if p <= 0 else 0
        max_consec = max(max_consec, cur)

    print()
    print(f"Starting equity:       {starting_equity:>12,.2f}")
    print(f"Final equity:          {final_equity:>12,.2f}")
    print(f"Total return:          {total_return:>11.2f}%")
    print(f"Max drawdown:          {max_dd:>11.2f}%")
    print(f"Sharpe ratio:          {sharpe:>12.3f}")
    print(f"Calmar ratio:          {calmar:>12.3f}")
    print(f"Profit factor:         {profit_factor:>12.3f}")
    print(f"Closed legs:           {n_trades:>12}")
    print(f"Win rate:              {win_rate:>11.1f}%")
    print(f"Avg win:               {wins.mean() if len(wins) else 0:>12.2f}")
    print(f"Avg loss:              {losses.mean() if len(losses) else 0:>12.2f}")
    print(f"Max consec. losses:    {max_consec:>12}")
    print(f"Avg leg duration:      {exits['duration_h'].mean():>11.1f}h")
    print(f"Total fees paid:       {trades_df['fee'].sum():>12,.2f}")

    print("\nExit reasons:")
    for reason, grp in exits.groupby("reason"):
        print(f"  {reason:<9} {len(grp):>5} legs   net {grp['pnl'].sum():>10,.2f}   "
              f"avg {grp['pnl'].mean():>8,.2f}")

    print("\nPer-symbol net PnL:")
    by_sym = exits.groupby("symbol")["pnl"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    for sym, row in by_sym.iterrows():
        print(f"  {sym:<14} {int(row['count']):>4} legs   {row['sum']:>10,.2f}")

    return {
        "total_return_%": round(total_return, 3),
        "max_dd_%":       round(max_dd, 3),
        "sharpe":         round(float(sharpe), 4),
        "calmar":         round(float(calmar), 4),
        "profit_factor":  round(float(profit_factor), 4),
        "n_trades":       n_trades,
        "win_rate_%":     round(win_rate, 2),
        "max_consec_loss": max_consec,
    }


def main():
    p = argparse.ArgumentParser(description="Backtest the Quant Scanner strategy")
    p.add_argument("--symbols", nargs="+", help="Symbols to include, e.g. BTCUSDT ETHUSDT")
    p.add_argument("--symbols-from-file", help="File with one symbol per line")
    p.add_argument("--top", type=int, help="Use the current top-N by 24h volume from Bybit")
    p.add_argument("--days", type=int, default=180, help="Days of history (default 180)")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--slippage", type=float, default=0.001, help="Per-fill slippage (default 0.001 = 0.1%%)")
    p.add_argument("--funding", type=float, default=0.0001, help="Funding per 8h (default 0.0001 = 0.01%%)")
    p.add_argument("--fee", type=float, default=0.00055, help="Taker fee (default 0.00055)")
    p.add_argument("--max-active", type=int, help="Override max concurrent trades")
    p.add_argument("--max-hold", type=float, help="Override max hold hours")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-fetch", action="store_true", help="Only use CSVs already on disk")
    p.add_argument("--out-prefix", default="quant_backtest")
    args = p.parse_args()

    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.symbols_from_file:
        symbols = [ln.strip() for ln in Path(args.symbols_from_file).read_text().splitlines() if ln.strip()]
    elif args.top:
        from exchange.bybit_client import BybitExchange
        symbols = BybitExchange(CONFIG.exchange).get_top_symbols(limit=args.top)
        print(f"Top {args.top} by volume: {', '.join(symbols)}")
    else:
        p.error("Provide --symbols, --symbols-from-file, or --top")

    data_dir = Path(args.data_dir)
    print(f"Loading {len(symbols)} symbols ({args.days} days of 15m candles)...")
    symbol_data = {}
    for sym in symbols:
        df = load_or_fetch(sym, str(_TIMEFRAME_MIN), args.days, data_dir, allow_fetch=not args.no_fetch)
        if df is not None and len(df) >= 60:
            symbol_data[sym] = df
    if not symbol_data:
        print("[ERROR] No usable symbol data.", file=sys.stderr)
        sys.exit(1)

    btc = load_or_fetch("BTCUSDT", "60", args.days, data_dir, allow_fetch=not args.no_fetch)
    if btc is None:
        print("[ERROR] BTCUSDT 60m data is required for the regime filter.", file=sys.stderr)
        sys.exit(1)
    regime = btc_regime_series(btc)

    # Echo the exact parameters -- results are meaningless without them, and
    # these come from .env, which differs between machines.
    s, r, sc = CONFIG.strategy, CONFIG.risk, CONFIG.scanner
    print(f"\nStrategy: EMA {s.fast_ema}/{s.slow_ema} | atr_period {s.atr_period} | "
          f"min_adx {s.min_adx} | min_volume_spike {s.min_volume_spike} | long_only {s.long_only}")
    print(f"Risk:     {r.account_risk_per_trade*100:.3f}%/trade | max_pos {r.max_position_pct*100:.0f}% | "
          f"{r.leverage}x | min_stop {r.min_stop_pct*100:.2f}% | daily_loss_cap {r.max_daily_loss_pct*100:.1f}%")
    print(f"Scanner:  max_active {args.max_active or sc.max_active_trades} | "
          f"max_hold {args.max_hold or sc.max_hold_hours}h")
    print(f"Costs:    slippage {args.slippage*100:.3f}%/fill | fee {args.fee*100:.4f}% | "
          f"funding {args.funding*100:.4f}%/8h")

    eq_df, trades_df = run_quant_backtest(
        symbol_data, regime,
        starting_equity=args.equity,
        fee_rate=args.fee,
        slippage_pct=args.slippage,
        funding_rate_8h=args.funding,
        max_active_trades=args.max_active,
        max_hold_hours=args.max_hold,
    )
    summarize(eq_df, trades_df, args.equity)

    trades_path = f"{args.out_prefix}_trades.csv"
    equity_path = f"{args.out_prefix}_equity_curve.csv"
    trades_df.to_csv(trades_path, index=False)
    eq_df.to_csv(equity_path)
    print(f"\nSaved: {trades_path}, {equity_path}")


if __name__ == "__main__":
    main()
