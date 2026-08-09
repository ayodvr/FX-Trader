"""
Backtest for the Donchian breakout trend strategy (strategy/breakout_strategy.py).

Same cost model and fidelity rules as backtest/run_quant.py so the two are
directly comparable: intra-candle stop fills against high/low, worst-case
ordering when a bar spans both stop and target, per-fill slippage, taker fees,
and 8-hourly funding on open notional.

Usage:
    python -m backtest.run_breakout --symbols BTCUSDT ETHUSDT SOLUSDT --interval 240
    python -m backtest.run_breakout --symbols BTCUSDT --interval 240 --channel 20 --atr-exit 3.0
"""
import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from backtest.run_quant import load_or_fetch
from risk.risk_manager import RiskManager
from strategy.breakout_strategy import (
    BreakoutConfig,
    Signal,
    compute_breakout_indicators,
    evaluate_breakout,
)

_FUNDING_HOURS = {0, 8, 16}


def run_breakout_backtest(
    symbol_data: dict[str, pd.DataFrame],
    cfg: BreakoutConfig,
    starting_equity: float = 10_000.0,
    fee_rate: float = 0.00055,
    slippage_pct: float = 0.001,
    funding_rate_8h: float = 0.0001,
    max_concurrent: int = 3,
    risk_cfg=None,
    quiet: bool = False,
    entry_mode: str = "taker",
    maker_fee: float = 0.0002,
    maker_offset_pct: float = 0.0,
    maker_timeout_bars: int = 3,
):
    """
    entry_mode:
      "taker" -- market entry on the signal bar's close, paying fee + slippage.
      "maker" -- rest a post-only limit at close*(1 -/+ maker_offset_pct) and fill
                 only if a later bar within maker_timeout_bars actually trades
                 through it. Cheaper per fill, but a breakout that never looks
                 back is simply missed -- which is the cost this mode exists to
                 measure, since those runaway moves are where the profit is.
    Exits stay taker in both modes: a trailing stop is a stop-market order and
    cannot be posted passively.
    """
    risk = RiskManager(risk_cfg if risk_cfg is not None else CONFIG.risk)

    def _log(m):
        if not quiet:
            print(m)

    ind = {}
    warm = max(cfg.channel, cfg.trend_ma, cfg.atr_period) + 5
    for sym, df in symbol_data.items():
        if len(df) < warm + 10:
            _log(f"  [skip] {sym}: only {len(df)} candles")
            continue
        ind[sym] = compute_breakout_indicators(df, cfg)
    if not ind:
        raise ValueError("No symbols had enough data")

    timeline = pd.DatetimeIndex(sorted(set().union(*(d.index for d in ind.values()))))
    _COLS = ("close", "high", "low", "atr", "upper", "lower",
             "exit_lower", "exit_upper", "trend_ma")
    arr, first_valid = {}, {}
    for sym, d in ind.items():
        r = d.reindex(timeline)
        cols = {c: r[c].to_numpy(dtype=float) for c in _COLS}
        cols["valid"] = r["close"].notna().to_numpy()
        arr[sym] = cols
        v = np.flatnonzero(cols["valid"])
        first_valid[sym] = int(v[0]) if len(v) else len(timeline)

    def row_at(sym, i):
        a = arr[sym]
        return {c: a[c][i] for c in _COLS}

    hours, minutes = timeline.hour.to_numpy(), timeline.minute.to_numpy()
    equity = starting_equity
    open_trades: dict[str, dict] = {}
    pending: dict[str, dict] = {}     # resting post-only entries, maker mode only
    trades: list[dict] = []
    curve = np.empty(len(timeline), dtype=float)
    missed_fills = 0

    _log(f"Simulating {len(timeline):,} bars across {len(ind)} symbols "
         f"(channel {cfg.channel}, exit {cfg.atr_exit_mult} ATR, max {max_concurrent} concurrent)...")

    for i in range(len(timeline)):
        ts = timeline[i]

        if hours[i] in _FUNDING_HOURS and minutes[i] == 0:
            for sym, tr in open_trades.items():
                if arr[sym]["valid"][i]:
                    equity -= tr["qty"] * arr[sym]["close"][i] * funding_rate_8h

        # ── Manage open positions ────────────────────────────────────────────
        for sym in list(open_trades):
            a = arr[sym]
            if not a["valid"][i]:
                continue
            tr = open_trades[sym]
            high, low, close = a["high"][i], a["low"][i], a["close"][i]
            is_long = tr["side"] == "Buy"
            factor = 1.0 if is_long else -1.0

            # Track the extreme the chandelier stop hangs from
            tr["extreme"] = max(tr["extreme"], high) if is_long else min(tr["extreme"], low)

            def _close(px, reason):
                nonlocal equity
                fill = px * (1 - slippage_pct) if is_long else px * (1 + slippage_pct)
                pnl = (fill - tr["entry_price"]) * tr["qty"] * factor
                fee = tr["qty"] * fill * fee_rate
                equity += pnl - fee
                risk.record_realized_pnl(pnl - fee, equity, now=ts)
                trades.append({
                    "time": ts, "symbol": sym, "side": tr["side"], "action": "EXIT",
                    "reason": reason, "price": fill, "qty": tr["qty"], "pnl": pnl - fee,
                    "fee": fee, "duration_h": (ts - tr["entry_time"]).total_seconds() / 3600,
                    "r_multiple": ((fill - tr["entry_price"]) * factor) / tr["risk_per_unit"]
                                  if tr["risk_per_unit"] > 0 else 0.0,
                })

            # Hard stop first -- worst case when a bar spans stop and trail
            stop_hit = (low <= tr["stop_price"]) if is_long else (high >= tr["stop_price"])
            if stop_hit:
                _close(tr["stop_price"], "STOP")
                del open_trades[sym]
                continue

            out = evaluate_breakout(row_at(sym, i), cfg,
                                    current_position=1 if is_long else -1,
                                    current_trail=tr["trail"],
                                    extreme_since_entry=tr["extreme"])
            if out.signal == Signal.FLAT:
                _close(close, "TRAIL")
                del open_trades[sym]
                continue
            if out.trail_price is not None:
                tr["trail"] = out.trail_price
                # The trail becomes the working stop once it passes the initial one
                tr["stop_price"] = (max(tr["stop_price"], out.trail_price) if is_long
                                    else min(tr["stop_price"], out.trail_price))

        # ── Entries ──────────────────────────────────────────────────────────
        if risk.kill_switch_active(now=ts):
            curve[i] = equity
            continue

        for sym in arr:
            a = arr[sym]
            if sym in open_trades or not a["valid"][i] or i - first_valid[sym] < warm:
                continue
            if len(open_trades) >= max_concurrent:
                break

            out = evaluate_breakout(row_at(sym, i), cfg, current_position=0)
            if out.signal not in (Signal.LONG, Signal.SHORT):
                continue

            close = a["close"][i]
            sizing = risk.size_position(equity, close, out.stop_price, open_positions=0, now=ts)
            if not sizing.approved:
                continue

            is_long = out.signal == Signal.LONG

            if entry_mode == "maker":
                # Post-only: a buy must rest at or below market, a sell at or
                # above it, otherwise the exchange rejects it for crossing.
                limit = (close * (1 - maker_offset_pct) if is_long
                         else close * (1 + maker_offset_pct))
                pending[sym] = {
                    "is_long": is_long, "limit": limit, "qty": sizing.qty,
                    "stop_price": out.stop_price, "expires": i + maker_timeout_bars,
                }
                continue

            fill = close * (1 + slippage_pct) if is_long else close * (1 - slippage_pct)
            fee = sizing.qty * fill * fee_rate
            equity -= fee
            open_trades[sym] = {
                "side": "Buy" if is_long else "Sell",
                "entry_price": fill, "entry_time": ts, "qty": sizing.qty,
                "stop_price": out.stop_price, "trail": None,
                "extreme": a["high"][i] if is_long else a["low"][i],
                "risk_per_unit": abs(fill - out.stop_price),
            }
            trades.append({
                "time": ts, "symbol": sym, "side": open_trades[sym]["side"],
                "action": "ENTER", "reason": "", "price": fill, "qty": sizing.qty,
                "pnl": None, "fee": fee, "duration_h": None, "r_multiple": None,
            })

        # ── Resting post-only entries: fill, expire, or keep waiting ─────────
        for sym in list(pending):
            po = pending[sym]
            a = arr[sym]
            if not a["valid"][i]:
                continue
            if sym in open_trades:
                del pending[sym]
                continue

            # Filled only if this bar actually traded through the resting price.
            hit = (a["low"][i] <= po["limit"]) if po["is_long"] else (a["high"][i] >= po["limit"])
            if hit:
                fill = po["limit"]          # maker: no slippage, we set the price
                fee = po["qty"] * fill * maker_fee
                equity -= fee
                is_long = po["is_long"]
                trades.append({
                    "time": ts, "symbol": sym, "side": "Buy" if is_long else "Sell",
                    "action": "ENTER", "reason": "maker", "price": fill,
                    "qty": po["qty"], "pnl": None, "fee": fee,
                    "duration_h": None, "r_multiple": None,
                })
                del pending[sym]

                # The bar that filled us kept trading after the fill. If it also
                # reached the stop, we were stopped inside that same bar -- not
                # granted a free look at the next one. Without this the deeper
                # the resting offset, the more the model quietly filled on
                # down-moves and then ignored the rest of the move.
                same_bar_stop = ((a["low"][i] <= po["stop_price"]) if is_long
                                 else (a["high"][i] >= po["stop_price"]))
                if same_bar_stop:
                    px = po["stop_price"]
                    exit_fill = px * (1 - slippage_pct) if is_long else px * (1 + slippage_pct)
                    factor = 1.0 if is_long else -1.0
                    pnl = (exit_fill - fill) * po["qty"] * factor
                    exit_fee = po["qty"] * exit_fill * fee_rate
                    equity += pnl - exit_fee
                    risk.record_realized_pnl(pnl - exit_fee, equity, now=ts)
                    rpu = abs(fill - po["stop_price"])
                    trades.append({
                        "time": ts, "symbol": sym, "side": "Buy" if is_long else "Sell",
                        "action": "EXIT", "reason": "STOP", "price": exit_fill,
                        "qty": po["qty"], "pnl": pnl - exit_fee, "fee": exit_fee,
                        "duration_h": 0.0,
                        "r_multiple": ((exit_fill - fill) * factor) / rpu if rpu > 0 else 0.0,
                    })
                    continue

                open_trades[sym] = {
                    "side": "Buy" if is_long else "Sell",
                    "entry_price": fill, "entry_time": ts, "qty": po["qty"],
                    "stop_price": po["stop_price"], "trail": None,
                    "extreme": a["high"][i] if is_long else a["low"][i],
                    "risk_per_unit": abs(fill - po["stop_price"]),
                }
            elif i >= po["expires"]:
                missed_fills += 1
                del pending[sym]

        curve[i] = equity

    if missed_fills and not quiet:
        filled = len([t for t in trades if t["action"] == "ENTER"])
        _log(f"  maker entries: {filled} filled, {missed_fills} missed "
             f"({missed_fills / (filled + missed_fills) * 100:.1f}% of signals never filled)")

    return (pd.DataFrame({"equity": curve}, index=timeline).rename_axis("time"),
            pd.DataFrame(trades))


def summarize(eq_df, trades_df, starting_equity, bars_per_year: float, quiet=False):
    final = eq_df["equity"].iloc[-1] if len(eq_df) else starting_equity
    ret = (final / starting_equity - 1) * 100
    rm = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - rm) / rm).min() * 100 if len(eq_df) else 0.0

    exits = trades_df[trades_df["action"] == "EXIT"] if len(trades_df) else pd.DataFrame()
    n = len(exits)
    if n == 0:
        if not quiet:
            print("\nNo trades taken.")
        return {"n_trades": 0, "total_return_%": round(ret, 2), "pf": 0.0}

    pnl = exits["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else float("inf")
    rets = eq_df["equity"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(bars_per_year)) if rets.std(ddof=1) > 0 else 0.0

    out = {
        "total_return_%": round(ret, 2), "max_dd_%": round(max_dd, 2),
        "sharpe": round(float(sharpe), 3), "pf": round(float(pf), 3),
        "n_trades": n, "win_%": round((pnl > 0).mean() * 100, 1),
    }
    if quiet:
        return out

    print()
    print(f"Starting equity:       {starting_equity:>12,.2f}")
    print(f"Final equity:          {final:>12,.2f}")
    print(f"Total return:          {ret:>11.2f}%")
    print(f"Max drawdown:          {max_dd:>11.2f}%")
    print(f"Sharpe ratio:          {sharpe:>12.3f}")
    print(f"Profit factor:         {pf:>12.3f}")
    print(f"Trades:                {n:>12}")
    print(f"Win rate:              {(pnl > 0).mean()*100:>11.1f}%")
    print(f"Avg win:               {wins.mean() if len(wins) else 0:>12.2f}")
    print(f"Avg loss:              {losses.mean() if len(losses) else 0:>12.2f}")
    print(f"Largest win:           {pnl.max():>12.2f}")
    print(f"Avg duration:          {exits['duration_h'].mean()/24:>11.1f}d")
    print(f"Total fees:            {trades_df['fee'].sum():>12,.2f}")
    if "r_multiple" in exits:
        r = exits["r_multiple"].dropna()
        print(f"Avg R multiple:        {r.mean():>12.2f}   (best {r.max():.1f}R)")
        print(f"Expectancy:            {r.mean():>12.2f}R per trade")

    print("\nExit reasons:")
    for reason, g in exits.groupby("reason"):
        print(f"  {reason:<7} {len(g):>4}   net {g['pnl'].sum():>10,.2f}   avg {g['pnl'].mean():>8,.2f}")
    print("\nPer-symbol:")
    for sym, g in exits.groupby("symbol"):
        print(f"  {sym:<12} {len(g):>4} trades   {g['pnl'].sum():>10,.2f}")
    return out


def main():
    p = argparse.ArgumentParser(description="Backtest the Donchian breakout strategy")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--interval", default="240", help="Candle interval in minutes (default 240 = 4h)")
    p.add_argument("--days", type=int, default=1000)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--channel", type=int, default=20)
    p.add_argument("--exit-channel", type=int, default=10)
    p.add_argument("--atr-stop", type=float, default=3.0)
    p.add_argument("--atr-exit", type=float, default=3.0)
    p.add_argument("--trend-ma", type=int, default=50)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--fee", type=float, default=0.00055)
    p.add_argument("--funding", type=float, default=0.0001)
    p.add_argument("--entry-mode", choices=["taker", "maker", "both"], default="taker",
                   help="taker = market entry; maker = post-only limit; both = compare")
    p.add_argument("--maker-fee", type=float, default=0.0002)
    p.add_argument("--maker-offset", type=float, default=0.0,
                   help="Rest this fraction away from close (0 = at the close)")
    p.add_argument("--maker-timeout", type=int, default=3, help="Bars before an unfilled entry is cancelled")
    p.add_argument("--sweep", action="store_true", help="Grid search with 70/30 walk-forward split")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-prefix", default="breakout_backtest")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    symbol_data = {}
    for s in args.symbols:
        df = load_or_fetch(s, args.interval, args.days, data_dir, allow_fetch=True)
        if df is not None:
            symbol_data[s] = df
    if not symbol_data:
        print("[ERROR] No usable data.", file=sys.stderr)
        sys.exit(1)

    bars_per_year = 365 * 24 * 60 / int(args.interval)
    cfg = BreakoutConfig(
        channel=args.channel, exit_channel=args.exit_channel,
        atr_stop_mult=args.atr_stop, atr_exit_mult=args.atr_exit,
        trend_ma=args.trend_ma, long_only=args.long_only,
    )
    run_kwargs = dict(starting_equity=args.equity, fee_rate=args.fee,
                      slippage_pct=args.slippage, funding_rate_8h=args.funding,
                      max_concurrent=args.max_concurrent,
                      maker_fee=args.maker_fee, maker_offset_pct=args.maker_offset,
                      maker_timeout_bars=args.maker_timeout)

    if args.entry_mode == "both":
        print(f"Comparing execution modes | taker: fee {args.fee*100:.4f}% + slip "
              f"{args.slippage*100:.3f}%  vs  maker: fee {args.maker_fee*100:.4f}%, no slip, "
              f"offset {args.maker_offset*100:.3f}%, {args.maker_timeout}-bar timeout\n")
        results = {}
        for mode in ("taker", "maker"):
            eq, tr = run_breakout_backtest(symbol_data, cfg, entry_mode=mode, **run_kwargs)
            results[mode] = summarize(eq, tr, args.equity, bars_per_year, quiet=True)
            n_ent = len(tr[tr["action"] == "ENTER"]) if len(tr) else 0
            m = results[mode]
            print(f"  {mode:<6} return {m['total_return_%']:+8.2f}%  pf {m['pf']:.3f}  "
                  f"dd {m.get('max_dd_%', 0):+7.2f}%  entries {n_ent:>4}  "
                  f"trades {m['n_trades']:>4}  win {m['win_%']:.1f}%")
        d_ret = results["maker"]["total_return_%"] - results["taker"]["total_return_%"]
        d_pf  = results["maker"]["pf"] - results["taker"]["pf"]
        print(f"\n  maker - taker:  return {d_ret:+.2f} pts   profit factor {d_pf:+.3f}")
        return

    if args.sweep:
        grid = {
            "channel":       [20, 30, 55],
            "atr_exit_mult": [2.5, 3.0, 4.0],
            "trend_ma":      [0, 50, 100],
            "exit_channel":  [0, 10],
        }
        all_idx = sorted(set().union(*(d.index for d in symbol_data.values())))
        cut = all_idx[int(len(all_idx) * 0.7)]
        is_d  = {s: d[d.index <= cut] for s, d in symbol_data.items()}
        oos_d = {s: d[d.index > cut]  for s, d in symbol_data.items()}
        print(f"Walk-forward split at {cut} (70/30)\n")

        keys = list(grid)
        rows = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            c = replace(cfg, **params)
            try:
                m_is  = summarize(*run_breakout_backtest(is_d, c, quiet=True, **run_kwargs),
                                  args.equity, bars_per_year, quiet=True)
                m_oos = summarize(*run_breakout_backtest(oos_d, c, quiet=True, **run_kwargs),
                                  args.equity, bars_per_year, quiet=True)
            except Exception as e:
                print(f"  failed {params}: {e}")
                continue
            rows.append({**params,
                         "is_return_%": m_is["total_return_%"], "is_pf": m_is["pf"],
                         "is_trades": m_is["n_trades"],
                         "oos_return_%": m_oos["total_return_%"], "oos_pf": m_oos["pf"],
                         "oos_dd_%": m_oos.get("max_dd_%", 0), "oos_trades": m_oos["n_trades"]})
            print(f"  ch={params['channel']:<3} exit={params['atr_exit_mult']:<4} "
                  f"ma={params['trend_ma']:<4} xch={params['exit_channel']:<3} -> "
                  f"IS {m_is['total_return_%']:+8.2f}% (pf {m_is['pf']:.2f}) | "
                  f"OOS {m_oos['total_return_%']:+8.2f}% (pf {m_oos['pf']:.2f}, {m_oos['n_trades']}t)")

        df = pd.DataFrame(rows).sort_values("oos_return_%", ascending=False)
        df.to_csv(f"{args.out_prefix}_sweep.csv", index=False)
        print(f"\n{'='*95}\nTOP 12 BY OUT-OF-SAMPLE RETURN\n{'='*95}")
        print(df.head(12).to_string(index=False))
        good = df[(df["oos_return_%"] > 0) & (df["oos_trades"] >= 10)]
        print(f"\n{len(good)} of {len(df)} configurations profitable out-of-sample "
              f"with >=10 trades.")
        print(f"Saved: {args.out_prefix}_sweep.csv")
        return

    print(f"Costs: fee {args.fee*100:.4f}% | slippage {args.slippage*100:.3f}% | "
          f"funding {args.funding*100:.4f}%/8h")
    eq, tr = run_breakout_backtest(symbol_data, cfg, entry_mode=args.entry_mode, **run_kwargs)
    summarize(eq, tr, args.equity, bars_per_year)
    tr.to_csv(f"{args.out_prefix}_trades.csv", index=False)
    eq.to_csv(f"{args.out_prefix}_equity.csv")
    print(f"\nSaved: {args.out_prefix}_trades.csv, {args.out_prefix}_equity.csv")


if __name__ == "__main__":
    main()
