"""
Trend-following strategy: EMA crossover for direction + ATR for volatility-adjusted stops.

Design principle: this module is PURE LOGIC. It takes a DataFrame of OHLCV candles
and returns signals. It knows nothing about exchanges, order execution, or state.
This is what lets the exact same code run in backtest.py and live.py — eliminating
the "backtest says X, live does Y" class of bugs.
"""
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from config import StrategyConfig


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    HOLD = "hold"  # no change to current position


@dataclass
class StrategyOutput:
    signal: Signal
    stop_price: float | None = None       # initial stop-loss level if entering
    trail_price: float | None = None      # current trailing stop level if in position
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    atr: float = 0.0


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    df must have columns: open, high, low, close, volume, indexed by time, ascending.
    Adds: fast_ema, slow_ema, atr, adx, trend (1/-1/0)
    """
    out = df.copy()
    out["fast_ema"] = out["close"].ewm(span=cfg.fast_ema, adjust=False).mean()
    out["slow_ema"] = out["close"].ewm(span=cfg.slow_ema, adjust=False).mean()

    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = true_range.ewm(span=cfg.atr_period, adjust=False).mean()

    # ADX — measures trend strength (direction-agnostic); below ~20 = choppy/ranging
    up_move   = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = true_range.ewm(span=cfg.atr_period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=out.index).ewm(span=cfg.atr_period, adjust=False).mean() / atr14
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(span=cfg.atr_period, adjust=False).mean() / atr14
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    out["adx"] = dx.ewm(span=cfg.atr_period, adjust=False).mean()

    out["trend"] = np.where(out["fast_ema"] > out["slow_ema"], 1,
                     np.where(out["fast_ema"] < out["slow_ema"], -1, 0))
    
    # Volume Spike Filter
    vol_ma = out["volume"].rolling(window=20).mean()
    out["vol_spike"] = (out["volume"] / vol_ma.replace(0, np.nan)).fillna(0)

    return out


def generate_signal(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    current_position: int,   # 1 = long, -1 = short, 0 = flat
    current_trail: float | None = None,
    df_daily: pd.DataFrame | None = None,
) -> StrategyOutput:
    """
    Called once per closed candle. Returns the signal for the *next* action.

    Entry: EMA crossover (fast crosses slow) when flat.
    Exit: trend flips against current position, OR price hits the trailing ATR stop.
    """
    ind = compute_indicators(df, cfg)
    last = ind.iloc[-1]
    prev = ind.iloc[-2]

    fast, slow, atr, close = last["fast_ema"], last["slow_ema"], last["atr"], last["close"]
    crossed_up = prev["fast_ema"] <= prev["slow_ema"] and fast > slow
    crossed_down = prev["fast_ema"] >= prev["slow_ema"] and fast < slow

    # --- Flat: look for entry (only when trend is strong enough) ---
    if current_position == 0:
        adx_ok = last["adx"] >= cfg.min_adx if cfg.min_adx > 0 else True
        vol_ok = last["vol_spike"] >= cfg.min_volume_spike if cfg.min_volume_spike > 0 else True
        
        daily_trend = 0
        if df_daily is not None and not df_daily.empty:
            daily_ind = compute_indicators(df_daily, cfg)
            daily_trend = daily_ind.iloc[-1]["trend"]
            
        if crossed_up and adx_ok and vol_ok and daily_trend >= 0:
            stop = close - atr * cfg.atr_stop_mult
            return StrategyOutput(Signal.LONG, stop_price=stop, fast_ema=fast, slow_ema=slow, atr=atr)
        if crossed_down and adx_ok and vol_ok and not cfg.long_only and daily_trend <= 0:
            stop = close + atr * cfg.atr_stop_mult
            return StrategyOutput(Signal.SHORT, stop_price=stop, fast_ema=fast, slow_ema=slow, atr=atr)
        return StrategyOutput(Signal.HOLD, fast_ema=fast, slow_ema=slow, atr=atr)

    # --- In position: trend flip or trailing stop ---
    if current_position == 1:
        new_trail = close - atr * cfg.atr_trail_mult
        trail = max(current_trail or new_trail, new_trail)  # only ratchets up
        if crossed_down or close <= trail:
            return StrategyOutput(Signal.FLAT, fast_ema=fast, slow_ema=slow, atr=atr)
        return StrategyOutput(Signal.HOLD, trail_price=trail, fast_ema=fast, slow_ema=slow, atr=atr)

    if current_position == -1:
        new_trail = close + atr * cfg.atr_trail_mult
        trail = min(current_trail or new_trail, new_trail)  # only ratchets down
        if crossed_up or close >= trail:
            return StrategyOutput(Signal.FLAT, fast_ema=fast, slow_ema=slow, atr=atr)
        return StrategyOutput(Signal.HOLD, trail_price=trail, fast_ema=fast, slow_ema=slow, atr=atr)

    return StrategyOutput(Signal.HOLD, fast_ema=fast, slow_ema=slow, atr=atr)
