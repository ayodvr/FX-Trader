"""
Institutional Multi-Factor Quantitative Strategy

Pillar 1: Top-Down Market Regime (BTC Trend Alignment)
Pillar 2: Multi-Factor Confluence (EMA 9/21/50 + RSI + MACD + Relative Volume RVOL)
Pillar 3: Dual Take-Profit (TP1 @ 1.5R with Auto-Breakeven, TP2 @ 3.0R runner)
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from config import StrategyConfig


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    HOLD = "hold"


@dataclass
class QuantOutput:
    signal: Signal
    stop_price: float | None = None        # initial stop-loss price
    tp1_price: float | None = None         # 1.5R target (scale out 50%, move SL to breakeven)
    tp2_price: float | None = None         # 3.0R target (runner exit)
    trail_price: float | None = None       # current ATR trailing-stop level while in position
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    trend_ema: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    rvol: float = 1.0
    btc_regime: int = 0                    # +1 bullish, -1 bearish, 0 neutral


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_quant_indicators(df: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    out = df.copy()

    fast_period  = getattr(cfg, "fast_ema", 9)  if cfg else 9
    slow_period  = getattr(cfg, "slow_ema", 21) if cfg else 21
    trend_period = 50
    atr_period   = getattr(cfg, "atr_period", 14) if cfg else 14

    out["fast_ema"]  = out["close"].ewm(span=fast_period, adjust=False).mean()
    out["slow_ema"]  = out["close"].ewm(span=slow_period, adjust=False).mean()
    out["trend_ema"] = out["close"].ewm(span=trend_period, adjust=False).mean()

    # ATR
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = true_range.ewm(span=atr_period, adjust=False).mean()

    # RSI
    out["rsi"] = compute_rsi(out["close"], period=14)

    # MACD (12, 26, 9)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Relative Volume (RVOL) = current volume / 20-candle moving average
    vol_ma = out["volume"].rolling(window=20).mean().replace(0, np.nan)
    out["rvol"] = (out["volume"] / vol_ma).fillna(1.0)

    return out


def evaluate_btc_regime(df_btc_1h: pd.DataFrame | None) -> int:
    """
    Returns:
       +1 = Bullish BTC Regime (1h Fast EMA > Slow EMA)
       -1 = Bearish BTC Regime (1h Fast EMA < Slow EMA)
        0 = Neutral / Ranging
    """
    if df_btc_1h is None or len(df_btc_1h) < 30:
        return 0
    
    fast = df_btc_1h["close"].ewm(span=20, adjust=False).mean()
    slow = df_btc_1h["close"].ewm(span=50, adjust=False).mean()
    
    last_fast = fast.iloc[-1]
    last_slow = slow.iloc[-1]
    
    if last_fast > last_slow * 1.001:
        return 1
    elif last_fast < last_slow * 0.999:
        return -1
    return 0


def generate_quant_signal(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    current_position: int = 0,
    current_trail: float | None = None,
    df_btc_1h: pd.DataFrame | None = None,
) -> QuantOutput:
    """
    Evaluates 3-pillar confluence:
    1. BTC Regime Alignment
    2. EMA Stack + RSI + RVOL Confluence
    3. Calculates Stop-Loss (1.5x ATR), TP1 (1.5R), and TP2 (3.0R)
    """
    ind = compute_quant_indicators(df, cfg)
    return evaluate_quant_signal(
        ind.iloc[-1], ind.iloc[-2], cfg,
        current_position=current_position,
        current_trail=current_trail,
        btc_regime=evaluate_btc_regime(df_btc_1h),
    )


def evaluate_quant_signal(
    last,
    prev,
    cfg: StrategyConfig,
    current_position: int = 0,
    current_trail: float | None = None,
    btc_regime: int = 0,
) -> QuantOutput:
    """
    The signal rules themselves, evaluated from two already-computed indicator rows.

    Split out from generate_quant_signal so the backtest can compute indicators
    once per symbol and then step through bars, instead of recomputing every EWM
    on every bar (which is O(n^2) and far too slow across 30 symbols). Live code
    goes through generate_quant_signal above; both share this one implementation,
    so the rules cannot drift apart between backtest and live.
    """
    fast, slow, trend = last["fast_ema"], last["slow_ema"], last["trend_ema"]
    atr, close, rsi, rvol = last["atr"], last["close"], last["rsi"], last["rvol"]

    crossed_up   = prev["fast_ema"] <= prev["slow_ema"] and fast > slow
    crossed_down = prev["fast_ema"] >= prev["slow_ema"] and fast < slow

    min_rvol = float(getattr(cfg, "min_volume_spike", 1.5))
    # In a neutral BTC regime (btc_regime == 0) we are trading against the broader
    # trend, so require a stronger volume spike to confirm conviction.
    # In a directional regime (+1/-1), the configured min_rvol applies directly.
    required_rvol = max(min_rvol, 1.15) if btc_regime == 0 else min_rvol

    # ── Flat: Look for 3-Point Confluence Entry ────────────────────────────────
    if current_position == 0:
        # LONG Confluence Requirements:
        #  - Fast > Slow EMA crossover
        #  - Price above Trend EMA (50)
        #  - RSI in momentum expansion range (40 - 72)
        #  - Relative Volume (RVOL) >= required_rvol
        #  - BTC regime not bearish (btc_regime >= 0)
        long_confluence = (
            crossed_up and
            close > trend and
            (40.0 <= rsi <= 72.0) and
            rvol >= required_rvol and
            btc_regime >= 0
        )

        # SHORT Confluence Requirements:
        #  - Fast < Slow EMA crossover
        #  - Price below Trend EMA (50)
        #  - RSI in momentum drop range (28 - 60)
        #  - Relative Volume (RVOL) >= required_rvol
        #  - BTC regime not bullish (btc_regime <= 0)
        short_confluence = (
            crossed_down and
            close < trend and
            (28.0 <= rsi <= 60.0) and
            rvol >= required_rvol and
            not cfg.long_only and
            btc_regime <= 0
        )

        stop_mult = float(getattr(cfg, "quant_stop_atr_mult", 1.5))
        tp1_r     = float(getattr(cfg, "quant_tp1_r", 1.5))
        tp2_r     = float(getattr(cfg, "quant_tp2_r", 3.0))

        if long_confluence:
            risk_dist = atr * stop_mult
            stop = close - risk_dist
            tp1  = close + (risk_dist * tp1_r)
            tp2  = close + (risk_dist * tp2_r)
            return QuantOutput(
                Signal.LONG, stop_price=stop, tp1_price=tp1, tp2_price=tp2,
                fast_ema=fast, slow_ema=slow, trend_ema=trend,
                atr=atr, rsi=rsi, rvol=rvol, btc_regime=btc_regime,
            )

        if short_confluence:
            risk_dist = atr * stop_mult
            stop = close + risk_dist
            tp1  = close - (risk_dist * tp1_r)
            tp2  = close - (risk_dist * tp2_r)
            return QuantOutput(
                Signal.SHORT, stop_price=stop, tp1_price=tp1, tp2_price=tp2,
                fast_ema=fast, slow_ema=slow, trend_ema=trend,
                atr=atr, rsi=rsi, rvol=rvol, btc_regime=btc_regime,
            )

        return QuantOutput(
            Signal.HOLD, fast_ema=fast, slow_ema=slow, trend_ema=trend,
            atr=atr, rsi=rsi, rvol=rvol, btc_regime=btc_regime,
        )

    # ── In Position: Trailing stop & Trend reversal exit ─────────────────────
    if current_position == 1:
        new_trail = close - atr * getattr(cfg, "atr_trail_mult", 2.5)
        trail = max(current_trail or new_trail, new_trail)
        if crossed_down or close <= trail:
            return QuantOutput(Signal.FLAT, fast_ema=fast, slow_ema=slow, atr=atr, btc_regime=btc_regime)
        return QuantOutput(Signal.HOLD, trail_price=trail, fast_ema=fast, slow_ema=slow, atr=atr, btc_regime=btc_regime)

    if current_position == -1:
        new_trail = close + atr * getattr(cfg, "atr_trail_mult", 2.5)
        trail = min(current_trail or new_trail, new_trail)
        if crossed_up or close >= trail:
            return QuantOutput(Signal.FLAT, fast_ema=fast, slow_ema=slow, atr=atr, btc_regime=btc_regime)
        return QuantOutput(Signal.HOLD, trail_price=trail, fast_ema=fast, slow_ema=slow, atr=atr, btc_regime=btc_regime)

    return QuantOutput(Signal.HOLD, fast_ema=fast, slow_ema=slow, atr=atr, btc_regime=btc_regime)
