# Running the real backtest (on your machine)

My sandbox has no network access, so this step has to run on your machine. It's quick:

```bash
cd tradingbot
pip install -r requirements.txt

# Pull 1 year of 15-minute BTCUSDT history from Bybit
python fetch_history.py --symbol BTCUSDT --interval 15 --days 365

# Run the backtest using that data
python -m backtest.run --csv data/BTCUSDT_15m.csv --equity 10000
```

This prints total return, max drawdown, trade count, and win rate, and saves two
CSVs (`backtest_trades.csv`, `backtest_equity_curve.csv`) you can inspect or chart.

## What to actually look at (don't just check if it's profitable)

1. **Max drawdown vs total return.** A strategy with +40% return but -35% max
   drawdown is a much worse bet than +20% return with -10% drawdown — you're the
   one who has to survive the drawdown psychologically and financially.
2. **Trade count.** Too few trades (under ~30-50) over a year means the result
   isn't statistically meaningful — you're looking at noise, not edge.
3. **Win rate vs average win/loss size.** Trend-following strategies often have
   win rates well under 50% but stay profitable because winners are bigger than
   losers (the trailing stop lets winners run). Don't reject the strategy just
   because win rate looks low — check the equity curve trend instead.
4. **Performance across different time windows.** Split the year into quarters
   and check if the strategy was only profitable in one trending stretch (red
   flag) vs reasonably consistent across regimes.
5. **Try other symbols/timeframes** (ETHUSDT, different EMA periods in
   `config.py`) to see if the edge is robust or curve-fit to one specific setup.

Send me the printed output (or the CSVs) and I'll help you interpret the results
and decide what to tune next.
