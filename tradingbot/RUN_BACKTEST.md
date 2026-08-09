# Running backtests

There are **two strategies in this repo, and each has its own backtest engine.**
Running the wrong one gives you numbers for a strategy you are not trading.

| Strategy | Lives in | Run by | Backtest with |
|---|---|---|---|
| Trend (EMA crossover, single symbol) | `strategy/trend_strategy.py` | `live/run.py` | `backtest/run.py` |
| Quant Scanner (multi-factor, top-N symbols) | `strategy/quant_strategy.py` | `live/run_quant_scanner.py` | `backtest/run_quant.py` |

---

## Quant Scanner backtest

Portfolio backtest — multiple symbols at once, staged TP1/TP2 exits, max-hold
timeout, and the BTC regime filter, matching what the live scanner does.

```bash
cd tradingbot
pip install -r requirements.txt

# Fetches any missing history automatically
python -m backtest.run_quant --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT --days 180

# Or use whatever is currently top-N by 24h volume
python -m backtest.run_quant --top 10 --days 180
```

Useful flags:

```bash
--slippage 0.0005     # per-fill slippage (default 0.001 = 0.1%)
--fee 0.00055         # taker fee (Bybit default)
--funding 0.0001      # per 8h on open notional
--max-active 2        # override concurrent trade cap
--max-hold 3.0        # override max hold hours
--no-fetch            # only use CSVs already in data/
```

Saves `quant_backtest_trades.csv` and `quant_backtest_equity_curve.csv`, and prints
a breakdown by exit reason (SL / BE / TP1 / TP2 / TIMEOUT / SIGNAL) and by symbol.

**Strategy parameters come from `.env`**, so the same command gives different
results on different machines. The run echoes every parameter it used — check
that line before comparing two results.

### Isolating whether an edge is real

The most useful diagnostic is running with costs switched off. If a strategy is
only profitable at zero fees, it does not have enough edge to trade:

```bash
python -m backtest.run_quant --symbols BTCUSDT --fee 0 --slippage 0 --funding 0
```

Compare against a fee-only run (`--slippage 0`) and a full-cost run. A strategy
that survives the first but not the second is losing to transaction costs, not
to the market — and no amount of parameter tuning fixes that.

---

## Trend strategy backtest

Single symbol, single position:

```bash
python fetch_history.py --symbol BTCUSDT --interval 15 --days 365
python -m backtest.run --csv data/BTCUSDT_15m.csv --equity 10000
```

Saves `backtest_trades.csv`, `backtest_equity_curve.csv`.

---

## What to actually look at (don't just check if it's profitable)

1. **Profit factor.** Gross wins / gross losses. Below 1.0 the strategy loses
   money. Around 1.0–1.1 it is inside the noise band and will not survive costs.
2. **Max drawdown vs total return.** +40% return with -35% drawdown is a much
   worse bet than +20% with -10% — you have to survive the drawdown both
   financially and psychologically.
3. **Trade count.** Under ~30–50 trades and the result is noise, not edge.
4. **Average win vs average loss.** If average loss exceeds average win, the win
   rate has to be correspondingly high just to break even. Staged exits that take
   partial profit early but keep full-size stops are structurally exposed to this.
5. **Exit-reason breakdown.** If almost all the PnL comes from one exit type, the
   others are probably costing you money.
6. **Per-symbol PnL.** If one symbol carries the whole result, that is curve fit,
   not edge.
7. **Different time windows.** Split into quarters — profitable only in one
   trending stretch is a red flag.
