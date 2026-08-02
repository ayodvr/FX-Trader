# Bybit Trend-Following Bot

EMA-crossover trend bot with ATR-based stops and trailing stops, built for Bybit USDT perpetuals.
The exact same strategy code runs in backtest and live trading — eliminating the
"backtest says X, live does Y" class of bugs.

---

## ⚠️ Before you do anything else

1. **Run a backtest on your target symbol/timeframe first.** Check Sharpe ratio and max drawdown,
   not just total return. A high return with a 40% drawdown is not a viable strategy.
2. **Always start with `DRY_RUN=true` and `BYBIT_TESTNET=true`.** Watch it run for at least
   a few days before touching either flag — and never flip both at once.
3. **Past backtest performance does not guarantee future results.** Crypto trend strategies can
   have long losing streaks. Review `max_daily_loss_pct` in `config.py` and make sure you are
   comfortable losing what it allows.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then edit `.env`:
```
BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here
BYBIT_TESTNET=true        # keep true until you have validated the bot
DRY_RUN=true              # keep true until you trust the logic

# Optional — Telegram alerts
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

The bot loads `.env` automatically on startup. You can also set these as shell
environment variables instead.

---

## Quickstart

### Step 1 — Fetch historical data

```bash
python fetch_history.py --symbol BTCUSDT --interval 60 --days 365
```

Saves `data/BTCUSDT_60m.csv`. Try multiple symbols and timeframes — results vary a lot.

### Step 2 — Backtest

```bash
python -m backtest.run --csv data/BTCUSDT_60m.csv --equity 10000
```

Prints a full performance report and saves `backtest_trades.csv` / `backtest_equity_curve.csv`.

**Sample output:**
```
Starting equity:            10,000.00
Final equity:               13,842.17
Total return:                   38.42%
Max drawdown:                  -12.31%
Sharpe ratio:                    1.143
Calmar ratio:                    3.120
Profit factor:                   1.872
Number of trades:                   47
Win rate:                       57.4%
Avg win:                        284.11
Avg loss:                      -138.22
Max consec. losses:                  4
Avg trade duration:             28.3h
```

Aim for: **Sharpe > 0.8**, **max drawdown < 25%**, **profit factor > 1.3**.

### Step 3 — Sweep parameter space

```bash
python sweep.py --csv data/BTCUSDT_60m.csv --sort-by sharpe
```

Tests EMA(21/55), (50/150), (100/200) and more combinations. Results are ranked by Sharpe ratio
by default (use `--sort-by return`, `calmar`, or `pf` to change). Includes a 70/30
**walk-forward split** — the "out-of-sample" return is what matters, not the in-sample figure.
Saves `sweep_results.csv`.

### Step 4 — Paper-trade (dry run on testnet)

```bash
python -m live.run
```

With `DRY_RUN=true`, the bot fetches real market data and logs/alerts what it *would* do,
but places no orders. Run for a meaningful stretch of time and compare decisions to what
you would expect.

### Step 5 — Go live

Only after step 4. Set `DRY_RUN=false` in your `.env`. Start with small size —
`account_risk_per_trade` and `max_position_pct` in `config.py` control this directly.

### Multi-symbol portfolio

```bash
python run_portfolio.py
```

Starts BTCUSDT, ETHUSDT, and SOLUSDT bots in parallel with independent risk managers.
Open the dashboard (`python dashboard.py` → http://localhost:8080) to monitor all three.

---

## Project structure

```
config.py                    # all tunable parameters, no secrets hardcoded
.env.example                 # copy to .env and fill in your API keys
strategy/trend_strategy.py   # pure signal logic — shared by backtest and live
risk/risk_manager.py         # position sizing + daily loss kill switch
exchange/bybit_client.py     # pybit wrapper: klines, orders, positions, equity
backtest/run.py              # backtest engine (uses same strategy + risk modules)
live/run.py                  # live trading loop
live/alerts.py               # Telegram alerting
dashboard.py                 # web UI: http://localhost:8080
fetch_history.py             # downloads historical OHLCV from Bybit for backtesting
sweep.py                     # parameter sweep with walk-forward out-of-sample validation
run_portfolio.py             # launches multiple live bots in parallel
tests/                       # 39+ unit and smoke tests (pytest)
```

---

## Key design decisions

- **Risk is enforced independently of strategy.** The strategy can only suggest direction;
  `RiskManager` decides size and can veto trades (kill switch, max positions, max notional).
  A bug in the strategy cannot bypass risk limits.
- **Live position state is read from the exchange every cycle**, not trusted from memory —
  avoids drift after a restart, missed fill, or manual intervention.
- **Stops are placed as real conditional orders on the exchange**, so a stop still fires
  even if the bot process dies.
- **Trailing stops are synced to the exchange** every time they ratchet by ≥0.05% — the
  exchange-side stop always reflects the current trail level, not just the initial entry stop.

---

## Known limitations

- **Market orders only** — no smarter execution (TWAP / limit-order-with-fallback). Fine
  for swing-style 15m+ timeframes; potentially problematic for very short timeframes.
- **Backtest slippage is modelled at 0.1%** per market order fill. Real slippage depends on
  symbol liquidity and order size. Tune `SLIPPAGE_PCT` in `backtest/run.py` if needed.
- **Funding rates are modelled at 0.01% per 8h** while a position is open. Real funding
  varies; check the actual historical funding on Bybit for your symbol.
- **Single-symbol, single-position** per bot instance — no intra-portfolio hedging.

---

## Running tests

```bash
python -m pytest tests/ -v
```
