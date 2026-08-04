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
    min_volume_spike: float = float(os.getenv("MIN_VOLUME_SPIKE", "1.0"))# 1.0 = off / disabled
    long_only: bool = os.getenv("LONG_ONLY", "false").lower() == "true"   # allow both Long and Short trades


@dataclass
class RiskConfig:
    account_risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.0075")) # 0.75% of equity risked per trade
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.25"))       # never put more than 25% of equity in one position (notional)
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS", "0.03"))       # kill switch: stop trading for the day after -3%
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "1"))          # this bot trades one symbol/position at a time
    leverage: int = int(os.getenv("LEVERAGE", "5"))                              # default 5X leverage for altcoin futures


@dataclass
class AlertConfig:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    poll_interval_sec: int = 30     # check every 30 seconds for closed 15m candles
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"  # paper-trade mode, no real orders


CONFIG = BotConfig()
