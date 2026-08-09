"""
Donchian breakout trend-following — designed around transaction costs, not against them.

Why this exists: the quant scanner's signal has a real edge (profit factor ~1.3
out-of-sample with costs off) but pays it all back in fees and slippage, because
it captures ~1% moves against a ~0.31% round trip. No parameter change fixes
that; the trade has to be bigger relative to the cost of making it.

So this strategy inverts the scanner's design:

  scanner                        this
  ─────────────────────────────  ────────────────────────────────────────
  15m candles                    4h/daily candles
  TP1 at 1.5R, half off          no profit target at all
  3-hour max hold                no time limit — winners run for weeks
  stop 1.5 ATR                   wider stop, sized so noise cannot hit it
  full-size loss, half-size win  full size both ways

That last row is the important one. Trend systems win maybe 35-45% of the time
and live on rare enormous winners, so capping the upside while leaving the
downside at full size inverts the only edge they have.

Entry:  close breaks the highest high / lowest low of the last `channel` bars.
Regime: only long above the long-term MA, only short below it — breakouts
        against the primary trend are where this family of systems bleeds.
Exit:   chandelier trailing stop, `atr_exit_mult` ATR from the highest high
        (or lowest low) reached since entry. It only ratchets in your favour.

Pure logic, same contract as the other strategy modules: OHLCV in, signal out,
no exchange or state awareness.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    HOLD = "hold"


@dataclass
class BreakoutConfig:
    channel: int = 20              # breakout lookback, in bars
    exit_channel: int = 10         # opposite-channel exit (0 disables)
    atr_period: int = 14
    atr_stop_mult: float = 3.0     # initial stop distance
    atr_exit_mult: float = 3.0     # chandelier trail distance
    trend_ma: int = 50             # regime filter; 0 disables
    long_only: bool = False


@dataclass
class BreakoutOutput:
    signal: Signal
    stop_price: float | None = None
    trail_price: float | None = None
    atr: float = 0.0
    upper: float = 0.0
    lower: float = 0.0
    trend_ma: float = 0.0


def compute_breakout_indicators(df: pd.DataFrame, cfg: BreakoutConfig) -> pd.DataFrame:
    out = df.copy()

    high_low   = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close  = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = true_range.ewm(span=cfg.atr_period, adjust=False).mean()

    # Channels are shifted by one bar so the current bar's own high/low cannot
    # form the level it is being tested against -- otherwise every bar trivially
    # "breaks out" of a channel it just helped define.
    out["upper"] = out["high"].rolling(cfg.channel).max().shift(1)
    out["lower"] = out["low"].rolling(cfg.channel).min().shift(1)

    if cfg.exit_channel > 0:
        out["exit_lower"] = out["low"].rolling(cfg.exit_channel).min().shift(1)
        out["exit_upper"] = out["high"].rolling(cfg.exit_channel).max().shift(1)
    else:
        out["exit_lower"] = np.nan
        out["exit_upper"] = np.nan

    out["trend_ma"] = (out["close"].rolling(cfg.trend_ma).mean()
                       if cfg.trend_ma > 0 else pd.Series(0.0, index=out.index))
    return out


def evaluate_breakout(
    last,
    cfg: BreakoutConfig,
    current_position: int = 0,
    current_trail: float | None = None,
    extreme_since_entry: float | None = None,
) -> BreakoutOutput:
    """
    Decide from one already-computed indicator row.

    `extreme_since_entry` is the highest high (long) or lowest low (short) seen
    since entry — the anchor the chandelier stop hangs from. Passing None falls
    back to the current close, which is what a freshly opened position uses.
    """
    close = last["close"]
    atr   = last["atr"]
    upper, lower = last["upper"], last["lower"]
    ma    = last["trend_ma"]

    if not np.isfinite(atr) or atr <= 0:
        return BreakoutOutput(Signal.HOLD, atr=0.0)

    base = BreakoutOutput(Signal.HOLD, atr=atr, upper=upper, lower=lower, trend_ma=ma)

    # ── Flat: look for a breakout in the direction of the primary trend ──────
    if current_position == 0:
        regime_ok_long  = (cfg.trend_ma == 0) or (np.isfinite(ma) and close > ma)
        regime_ok_short = (cfg.trend_ma == 0) or (np.isfinite(ma) and close < ma)

        if np.isfinite(upper) and close > upper and regime_ok_long:
            return BreakoutOutput(Signal.LONG, stop_price=close - atr * cfg.atr_stop_mult,
                                  atr=atr, upper=upper, lower=lower, trend_ma=ma)
        if (np.isfinite(lower) and close < lower and regime_ok_short and not cfg.long_only):
            return BreakoutOutput(Signal.SHORT, stop_price=close + atr * cfg.atr_stop_mult,
                                  atr=atr, upper=upper, lower=lower, trend_ma=ma)
        return base

    # ── In position: chandelier trail, plus optional opposite-channel exit ───
    if current_position == 1:
        anchor = max(extreme_since_entry, close) if extreme_since_entry is not None else close
        new_trail = anchor - atr * cfg.atr_exit_mult
        trail = max(current_trail, new_trail) if current_trail is not None else new_trail
        hit_channel = np.isfinite(last["exit_lower"]) and close < last["exit_lower"]
        if close <= trail or hit_channel:
            return BreakoutOutput(Signal.FLAT, atr=atr, upper=upper, lower=lower, trend_ma=ma)
        return BreakoutOutput(Signal.HOLD, trail_price=trail, atr=atr,
                              upper=upper, lower=lower, trend_ma=ma)

    if current_position == -1:
        anchor = min(extreme_since_entry, close) if extreme_since_entry is not None else close
        new_trail = anchor + atr * cfg.atr_exit_mult
        trail = min(current_trail, new_trail) if current_trail is not None else new_trail
        hit_channel = np.isfinite(last["exit_upper"]) and close > last["exit_upper"]
        if close >= trail or hit_channel:
            return BreakoutOutput(Signal.FLAT, atr=atr, upper=upper, lower=lower, trend_ma=ma)
        return BreakoutOutput(Signal.HOLD, trail_price=trail, atr=atr,
                              upper=upper, lower=lower, trend_ma=ma)

    return base


def generate_breakout_signal(
    df: pd.DataFrame,
    cfg: BreakoutConfig,
    current_position: int = 0,
    current_trail: float | None = None,
    extreme_since_entry: float | None = None,
) -> BreakoutOutput:
    """Convenience wrapper for live use: indicators + evaluation on the last bar."""
    ind = compute_breakout_indicators(df, cfg)
    return evaluate_breakout(ind.iloc[-1], cfg, current_position, current_trail, extreme_since_entry)
