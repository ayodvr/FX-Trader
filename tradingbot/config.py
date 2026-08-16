"""
Central configuration for the trading bot.
Secrets (API keys) are loaded from environment variables, never hardcoded.
A .env file in the same directory is loaded automatically if present.
Copy .env.example to .env and fill in your values.
"""
import os
from dataclasses import dataclass, field

# Load .env file if present (python-dotenv); no-op if the file doesn't exist
try:
    from pathlib import Path
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to shell env vars


@dataclass
class ExchangeConfig:
    api_key: str = os.getenv("BYBIT_API_KEY", "")
    api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    testnet: bool = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    category: str = "linear"  # USDT perpetuals


@dataclass
class StrategyConfig:
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    timeframe: str = os.getenv("TIMEFRAME", "15")  # minutes (Bybit kline interval) — 15m intraday candles
    fast_ema: int = int(os.getenv("FAST_EMA", "9"))
    slow_ema: int = int(os.getenv("SLOW_EMA", "21"))
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
    atr_stop_mult: float = float(os.getenv("ATR_STOP_MULT", "2.0"))      # stop-loss = entry -/+ ATR * mult
    atr_trail_mult: float = float(os.getenv("ATR_TRAIL_MULT", "3.0"))    # trailing stop distance
    min_adx: float = float(os.getenv("MIN_ADX", "10.0"))                 # filter choppy markets
    # 1.5 = require 50% above-average volume — real smart-money spike filter.
    # Set to 1.0 to disable (not recommended for live trading).
    min_volume_spike: float = float(os.getenv("MIN_VOLUME_SPIKE", "1.5"))
    long_only: bool = os.getenv("LONG_ONLY", "false").lower() == "true"   # allow both Long and Short trades

    # ── Quant Scanner exit structure (strategy/quant_strategy.py only) ────────
    # Defaults reproduce the original hardcoded values. Exposed because the
    # ratio between them decides the strategy's payoff profile: the stop risks
    # full size while TP1 takes profit on half, so tp1_r must clear costs by a
    # wide enough margin to pay for the losses that stop out at full size.
    quant_stop_atr_mult: float = float(os.getenv("QUANT_STOP_ATR_MULT", "1.5"))
    quant_tp1_r: float = float(os.getenv("QUANT_TP1_R", "1.5"))   # first target, in R
    quant_tp2_r: float = float(os.getenv("QUANT_TP2_R", "3.0"))   # runner target, in R


@dataclass
class RiskConfig:
    account_risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.0075")) # 0.75% of equity risked per trade
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.25"))       # never put more than 25% of equity in one position (notional)
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS", "0.03"))       # kill switch: stop trading for the day after -3%
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "1"))          # this bot trades one symbol/position at a time
    leverage: int = int(os.getenv("LEVERAGE", "5"))                              # 5X — safe default for live altcoin futures
    min_stop_pct: float = float(os.getenv("MIN_STOP_PCT", "0.003"))              # reject trades where stop < 0.3% from entry (too tight at leverage)
    # Reject absurdly WIDE stops too. A stop further away than the liquidation
    # distance can never fire -- the position is liquidated first -- so it is not
    # a stop at all. At 5x, liquidation is roughly 20% adverse; a bad ATR (from a
    # data spike) once produced a 57% stop on a live signal, which is unusable.
    max_stop_pct: float = float(os.getenv("MAX_STOP_PCT", "0.15"))


@dataclass
class AlertConfig:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))


@dataclass
class ScannerConfig:
    """Settings specific to the Quant Scanner daemon."""
    top_symbols_count: int = int(os.getenv("TOP_SYMBOLS_COUNT", "30"))     # number of top-volume symbols to scan
    max_hold_hours: float = float(os.getenv("MAX_HOLD_HOURS", "3.0"))      # stagnant-trade auto-exit after this many hours
    max_active_trades: int = int(os.getenv("MAX_ACTIVE_TRADES", "2"))      # cap simultaneous open positions (reduced from 3 for live safety)


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    poll_interval_sec: int = 30     # check every 30 seconds for closed 15m candles
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"  # paper-trade mode, no real orders


CONFIG = BotConfig()
