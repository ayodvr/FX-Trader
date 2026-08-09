"""
Backtest for the DCA / martingale grid bot (strategy/dca_strategy.py).

The point of this engine is the part the marketing screenshots leave out.
A DCA bot's win rate and daily PnL are not informative -- they are high and
smooth by construction. What decides whether the strategy is solvent is:

  - liquidation, modelled explicitly per position under isolated margin, since
    with leverage the ladder can run out of margin before price ever comes back
  - how deep the account drawdown goes while deals sit underwater
  - whether the worst single deal exceeds the sum of every winning deal

Fill ordering inside a bar is deliberately adverse: safety orders and
liquidation are evaluated against the bar's extreme BEFORE the take-profit is
considered. Without that, a bar that swept down through the ladder and back up
would be recorded as a clean win, which flatters exactly the scenario that
matters.

Usage:
    python -m backtest.run_dca --symbols BTCUSDT ETHUSDT --interval 60
    python -m backtest.run_dca --symbols BTCUSDT --leverage 10 --max-safety 5 --stop-loss 0
    python -m backtest.run_dca --symbols BTCUSDT --compare-leverage
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.run_quant import load_or_fetch
from strategy.dca_strategy import (
    DcaConfig,
    build_ladder,
    liquidation_price,
    stop_loss_price,
    take_profit_price,
)

_FUNDING_HOURS = {0, 8, 16}


def run_dca_backtest(
    symbol_data: dict[str, pd.DataFrame],
    cfg: DcaConfig,
    starting_equity: float = 10_000.0,
    fee_rate: float = 0.00055,
    slippage_pct: float = 0.0005,
    funding_rate_8h: float = 0.0001,
    max_concurrent_deals: int = 3,
    quiet: bool = False,
):
    def _log(m):
        if not quiet:
            print(m)

    data = {s: d for s, d in symbol_data.items() if len(d) > 50}
    if not data:
        raise ValueError("No usable symbol data")

    timeline = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    arr = {}
    for sym, d in data.items():
        r = d.reindex(timeline)
        arr[sym] = {c: r[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")}
        arr[sym]["valid"] = r["close"].notna().to_numpy()

    hours, minutes = timeline.hour.to_numpy(), timeline.minute.to_numpy()

    cash = starting_equity
    deals: dict[str, dict] = {}
    closed: list[dict] = []
    curve = np.empty(len(timeline), dtype=float)
    liquidations = 0
    peak_margin = 0.0
    is_long = cfg.direction != "short"

    _log(f"Simulating {len(timeline):,} bars across {len(data)} symbols "
         f"({cfg.max_safety_orders} safety orders, {cfg.leverage}x, "
         f"TP {cfg.take_profit_pct*100:.2f}%, "
         f"SL {'off' if cfg.stop_loss_pct <= 0 else f'{cfg.stop_loss_pct*100:.1f}%'})...")

    def _equity(i: float) -> float:
        """Cash plus unrealised PnL across every open deal."""
        unreal = 0.0
        for sym, dl in deals.items():
            if arr[sym]["valid"][i]:
                px = arr[sym]["close"][i]
                d = 1.0 if dl["is_long"] else -1.0
                unreal += (px - dl["avg"]) * dl["qty"] * d
        return cash + unreal

    for i in range(len(timeline)):
        ts = timeline[i]

        if hours[i] in _FUNDING_HOURS and minutes[i] == 0:
            for sym, dl in deals.items():
                if arr[sym]["valid"][i]:
                    cash -= dl["qty"] * arr[sym]["close"][i] * funding_rate_8h

        # ── Manage open deals ────────────────────────────────────────────────
        for sym in list(deals):
            a = arr[sym]
            if not a["valid"][i]:
                continue
            dl = deals[sym]
            high, low, close = a["high"][i], a["low"][i], a["close"][i]
            lng = dl["is_long"]
            d = 1.0 if lng else -1.0
            adverse = low if lng else high

            # 1. Safety orders fill on the adverse extreme, cheapest first.
            while dl["next_step"] < len(dl["ladder"]):
                step = dl["ladder"][dl["next_step"]]
                reached = (adverse <= step.price) if lng else (adverse >= step.price)
                if not reached:
                    break
                fill = step.price
                qty = step.notional / fill
                fee = step.notional * fee_rate
                cash -= fee
                dl["fills"].append((fill, qty))
                tot_q = dl["qty"] + qty
                dl["avg"] = (dl["avg"] * dl["qty"] + fill * qty) / tot_q
                dl["qty"] = tot_q
                dl["notional"] = dl["avg"] * dl["qty"]
                dl["margin"] = dl["notional"] / cfg.leverage
                dl["fees"] += fee
                dl["next_step"] += 1
                dl["safety_used"] += 1

            peak_margin = max(peak_margin, sum(x["margin"] for x in deals.values()))

            # 2. Liquidation -- checked before take-profit, because a bar that
            #    blew through the margin and recovered is not a win.
            liq = liquidation_price(dl["avg"], cfg.leverage, lng)
            hit_liq = (low <= liq) if lng else (high >= liq)
            if hit_liq:
                loss = -dl["margin"]        # isolated margin: the posted margin is gone
                cash += loss
                liquidations += 1
                closed.append({
                    "time": ts, "symbol": sym, "reason": "LIQUIDATED",
                    "pnl": loss - dl["fees"], "safety_used": dl["safety_used"],
                    "duration_h": (ts - dl["opened"]).total_seconds() / 3600,
                    "max_adverse_%": abs(liq / dl["fills"][0][0] - 1) * 100,
                    "notional": dl["notional"],
                })
                del deals[sym]
                continue

            # 3. Optional stop loss
            sl = stop_loss_price(dl["avg"], cfg, lng)
            if sl is not None:
                hit_sl = (low <= sl) if lng else (high >= sl)
                if hit_sl:
                    exit_px = sl * (1 - slippage_pct) if lng else sl * (1 + slippage_pct)
                    pnl = (exit_px - dl["avg"]) * dl["qty"] * d
                    fee = dl["qty"] * exit_px * fee_rate
                    cash += pnl - fee
                    closed.append({
                        "time": ts, "symbol": sym, "reason": "STOP",
                        "pnl": pnl - fee - dl["fees"], "safety_used": dl["safety_used"],
                        "duration_h": (ts - dl["opened"]).total_seconds() / 3600,
                        "max_adverse_%": abs(sl / dl["fills"][0][0] - 1) * 100,
                        "notional": dl["notional"],
                    })
                    del deals[sym]
                    continue

            # 4. Take profit off the running average entry
            tp = take_profit_price(dl["avg"], cfg, lng)
            hit_tp = (high >= tp) if lng else (low <= tp)
            if hit_tp:
                exit_px = tp * (1 - slippage_pct) if lng else tp * (1 + slippage_pct)
                pnl = (exit_px - dl["avg"]) * dl["qty"] * d
                fee = dl["qty"] * exit_px * fee_rate
                cash += pnl - fee
                closed.append({
                    "time": ts, "symbol": sym, "reason": "TP",
                    "pnl": pnl - fee - dl["fees"], "safety_used": dl["safety_used"],
                    "duration_h": (ts - dl["opened"]).total_seconds() / 3600,
                    "max_adverse_%": dl["ladder"][max(dl["next_step"] - 1, 0)].deviation * 100,
                    "notional": dl["notional"],
                })
                del deals[sym]

        eq = _equity(i)
        if eq <= 0:
            _log(f"  ACCOUNT WIPED OUT at {ts}")
            curve[i:] = 0.0
            break

        # ── Open new deals: no signal, just refill any free slot ─────────────
        for sym in arr:
            if sym in deals or not arr[sym]["valid"][i]:
                continue
            if len(deals) >= max_concurrent_deals:
                break
            px = arr[sym]["close"][i]
            ladder = build_ladder(px, eq, cfg, is_long)
            total_notional = sum(s.notional for s in ladder)
            # Refuse a deal whose full ladder could not be margined
            if total_notional / cfg.leverage > eq - sum(x["margin"] for x in deals.values()):
                continue

            base = ladder[0]
            fill = px * (1 + slippage_pct) if is_long else px * (1 - slippage_pct)
            qty = base.notional / fill
            fee = base.notional * fee_rate
            cash -= fee
            deals[sym] = {
                "is_long": is_long, "opened": ts, "ladder": ladder, "next_step": 1,
                "fills": [(fill, qty)], "avg": fill, "qty": qty,
                "notional": base.notional, "margin": base.notional / cfg.leverage,
                "fees": fee, "safety_used": 0,
            }

        curve[i] = _equity(i)

    eq_df = pd.DataFrame({"equity": curve}, index=timeline).rename_axis("time")
    return eq_df, pd.DataFrame(closed), {"liquidations": liquidations, "peak_margin": peak_margin}


def summarize(eq_df, deals_df, stats, starting_equity, quiet=False):
    final = eq_df["equity"].iloc[-1] if len(eq_df) else starting_equity
    ret = (final / starting_equity - 1) * 100
    rm = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - rm) / rm.replace(0, np.nan)).min() * 100 if len(eq_df) else 0.0

    n = len(deals_df)
    if n == 0:
        if not quiet:
            print("\nNo deals closed.")
        return {"n_deals": 0, "total_return_%": round(ret, 2), "win_%": 0.0,
                "max_dd_%": round(float(max_dd), 2), "liquidations": stats["liquidations"]}

    pnl = deals_df["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    win_rate = len(wins) / n * 100
    out = {
        "n_deals": n, "total_return_%": round(ret, 2), "win_%": round(win_rate, 1),
        "max_dd_%": round(float(max_dd), 2), "liquidations": stats["liquidations"],
        "worst_deal": round(float(pnl.min()), 2),
        "gross_wins": round(float(wins.sum()), 2),
    }
    if quiet:
        return out

    print()
    print(f"Starting equity:       {starting_equity:>12,.2f}")
    print(f"Final equity:          {final:>12,.2f}")
    print(f"Total return:          {ret:>11.2f}%")
    print(f"Max drawdown:          {max_dd:>11.2f}%")
    print(f"Deals closed:          {n:>12}")
    print(f"WIN RATE:              {win_rate:>11.1f}%   <-- the number the screenshots show")
    print(f"Avg win:               {wins.mean() if len(wins) else 0:>12.2f}")
    print(f"Avg loss:              {losses.mean() if len(losses) else 0:>12.2f}")
    print(f"Worst single deal:     {pnl.min():>12.2f}")
    print(f"Sum of ALL wins:       {wins.sum():>12.2f}")
    print(f"Sum of ALL losses:     {losses.sum():>12.2f}")
    print(f"Liquidations:          {stats['liquidations']:>12}")
    print(f"Peak margin committed: {stats['peak_margin']:>12,.2f}")
    print(f"Avg deal duration:     {deals_df['duration_h'].mean():>11.1f}h")
    print(f"Longest deal:          {deals_df['duration_h'].max()/24:>11.1f}d")

    if len(losses) and len(wins):
        print(f"\n  Worst deal is {abs(pnl.min()) / wins.mean():>.0f}x the average win, "
              f"and {abs(pnl.min()) / wins.sum() * 100:.1f}% of everything the wins earned.")

    print("\nOutcome breakdown:")
    for reason, g in deals_df.groupby("reason"):
        print(f"  {reason:<11} {len(g):>5} deals   net {g['pnl'].sum():>12,.2f}   "
              f"avg {g['pnl'].mean():>9,.2f}")

    print("\nSafety orders used per deal:")
    for used, g in deals_df.groupby("safety_used"):
        print(f"  {int(used)} used: {len(g):>5} deals   net {g['pnl'].sum():>12,.2f}")
    return out


def main():
    p = argparse.ArgumentParser(description="Backtest a DCA / martingale grid bot")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--interval", default="60")
    p.add_argument("--days", type=int, default=1000)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--direction", choices=["long", "short"], default="long")
    p.add_argument("--base-pct", type=float, default=0.02)
    p.add_argument("--safety-pct", type=float, default=0.02)
    p.add_argument("--max-safety", type=int, default=5)
    p.add_argument("--deviation", type=float, default=0.01)
    p.add_argument("--step-scale", type=float, default=1.5)
    p.add_argument("--volume-scale", type=float, default=1.5)
    p.add_argument("--take-profit", type=float, default=0.01)
    p.add_argument("--stop-loss", type=float, default=0.0, help="0 = no stop loss (the usual default)")
    p.add_argument("--leverage", type=int, default=10)
    p.add_argument("--max-deals", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.00055)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--funding", type=float, default=0.0001)
    p.add_argument("--compare-leverage", action="store_true",
                   help="Run 1x/3x/5x/10x/20x side by side")
    p.add_argument("--data-dir", default="data")
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

    cfg = DcaConfig(
        direction=args.direction, base_order_pct=args.base_pct,
        safety_order_pct=args.safety_pct, max_safety_orders=args.max_safety,
        price_deviation_pct=args.deviation, step_scale=args.step_scale,
        volume_scale=args.volume_scale, take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss, leverage=args.leverage,
    )
    run_kwargs = dict(starting_equity=args.equity, fee_rate=args.fee,
                      slippage_pct=args.slippage, funding_rate_8h=args.funding,
                      max_concurrent_deals=args.max_deals)

    if args.compare_leverage:
        print(f"{'lev':>5}{'return':>12}{'max dd':>11}{'win %':>9}{'deals':>8}"
              f"{'liquidations':>14}{'worst deal':>13}")
        for lev in (1, 3, 5, 10, 20):
            eq, dl, st = run_dca_backtest(symbol_data, replace(cfg, leverage=lev),
                                          quiet=True, **run_kwargs)
            m = summarize(eq, dl, st, args.equity, quiet=True)
            print(f"{lev:>4}x{m['total_return_%']:>11.2f}%{m['max_dd_%']:>10.2f}%"
                  f"{m['win_%']:>8.1f}%{m['n_deals']:>8}{m['liquidations']:>14}"
                  f"{m.get('worst_deal', 0):>13,.2f}")
        return

    eq, dl, st = run_dca_backtest(symbol_data, cfg, **run_kwargs)
    summarize(eq, dl, st, args.equity)
    dl.to_csv("dca_backtest_deals.csv", index=False)
    eq.to_csv("dca_backtest_equity.csv")
    print("\nSaved: dca_backtest_deals.csv, dca_backtest_equity.csv")


if __name__ == "__main__":
    main()
