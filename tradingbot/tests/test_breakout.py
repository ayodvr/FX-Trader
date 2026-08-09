"""Tests for the Donchian breakout trend strategy."""
import numpy as np
import pandas as pd
import pytest

from strategy.breakout_strategy import (
    BreakoutConfig,
    Signal,
    compute_breakout_indicators,
    evaluate_breakout,
    generate_breakout_signal,
)


def _flat_series(n: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "volume": 1000.0,
    }, index=idx)


class TestIndicators:
    def test_channels_exclude_the_current_bar(self):
        """A bar must not help define the channel it is tested against."""
        df = _flat_series(40)
        df.iloc[-1, df.columns.get_loc("high")] = 500.0
        ind = compute_breakout_indicators(df, BreakoutConfig(channel=20))
        # The spike is on the last bar, so it must not appear in that bar's upper
        assert ind["upper"].iloc[-1] < 500.0

    def test_upper_is_rolling_max_of_prior_bars(self):
        df = _flat_series(40)
        df.iloc[10, df.columns.get_loc("high")] = 200.0
        ind = compute_breakout_indicators(df, BreakoutConfig(channel=20))
        # upper[i] = max(high[i-20 .. i-1]); the spike at 10 is inside that
        # window at i=25 and has rolled out of it by i=35.
        assert ind["upper"].iloc[25] == 200.0
        assert ind["upper"].iloc[35] != 200.0


class TestEntries:
    def test_breakout_above_channel_goes_long(self):
        df = _flat_series(60)
        df.iloc[-1, df.columns.get_loc("close")] = 150.0
        df.iloc[-1, df.columns.get_loc("high")] = 150.0
        out = generate_breakout_signal(df, BreakoutConfig(channel=20, trend_ma=0))
        assert out.signal == Signal.LONG
        assert out.stop_price < 150.0

    def test_breakdown_below_channel_goes_short(self):
        df = _flat_series(60)
        df.iloc[-1, df.columns.get_loc("close")] = 50.0
        df.iloc[-1, df.columns.get_loc("low")] = 50.0
        out = generate_breakout_signal(df, BreakoutConfig(channel=20, trend_ma=0))
        assert out.signal == Signal.SHORT
        assert out.stop_price > 50.0

    def test_long_only_blocks_shorts(self):
        df = _flat_series(60)
        df.iloc[-1, df.columns.get_loc("close")] = 50.0
        df.iloc[-1, df.columns.get_loc("low")] = 50.0
        out = generate_breakout_signal(df, BreakoutConfig(channel=20, trend_ma=0, long_only=True))
        assert out.signal != Signal.SHORT

    def test_regime_filter_blocks_counter_trend_entry(self):
        """A long breakout below the trend MA must be rejected."""
        idx = pd.date_range("2026-01-01", periods=120, freq="4h", tz="UTC")
        close = np.concatenate([np.linspace(300, 100, 100), np.full(20, 100.0)])
        df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": 1000.0}, index=idx)
        # 110 clears the 20-bar channel (~101) but sits under the falling MA50
        df.iloc[-1, df.columns.get_loc("close")] = 110.0
        df.iloc[-1, df.columns.get_loc("high")] = 110.0
        cfg = BreakoutConfig(channel=20, trend_ma=50)
        ind = compute_breakout_indicators(df, cfg)
        assert ind["close"].iloc[-1] > ind["upper"].iloc[-1]      # it IS a breakout
        assert ind["close"].iloc[-1] < ind["trend_ma"].iloc[-1]   # but counter-trend
        assert evaluate_breakout(ind.iloc[-1], cfg).signal != Signal.LONG
        # ...and with the regime filter off, the same bar does trigger
        no_filter = BreakoutConfig(channel=20, trend_ma=0)
        ind2 = compute_breakout_indicators(df, no_filter)
        assert evaluate_breakout(ind2.iloc[-1], no_filter).signal == Signal.LONG


class TestTrailingExit:
    def test_trail_only_ratchets_up_for_longs(self):
        df = _flat_series(60)
        ind = compute_breakout_indicators(df, BreakoutConfig())
        cfg = BreakoutConfig(atr_exit_mult=3.0, exit_channel=0)
        high_trail = 95.0
        out = evaluate_breakout(ind.iloc[-1], cfg, current_position=1,
                                current_trail=high_trail, extreme_since_entry=100.0)
        if out.trail_price is not None:
            assert out.trail_price >= high_trail

    def test_trail_only_ratchets_down_for_shorts(self):
        df = _flat_series(60)
        ind = compute_breakout_indicators(df, BreakoutConfig())
        cfg = BreakoutConfig(atr_exit_mult=3.0, exit_channel=0)
        low_trail = 105.0
        out = evaluate_breakout(ind.iloc[-1], cfg, current_position=-1,
                                current_trail=low_trail, extreme_since_entry=100.0)
        if out.trail_price is not None:
            assert out.trail_price <= low_trail

    def test_close_through_trail_exits(self):
        df = _flat_series(60)
        ind = compute_breakout_indicators(df, BreakoutConfig())
        cfg = BreakoutConfig(atr_exit_mult=3.0, exit_channel=0)
        out = evaluate_breakout(ind.iloc[-1], cfg, current_position=1,
                                current_trail=500.0, extreme_since_entry=100.0)
        assert out.signal == Signal.FLAT

    def test_no_profit_target_exists(self):
        """The design has no take-profit: winners are only ended by the trail."""
        cfg = BreakoutConfig(exit_channel=0)
        ind = compute_breakout_indicators(_flat_series(60), cfg)
        out = evaluate_breakout(ind.iloc[-1], cfg, current_position=1,
                                current_trail=1.0, extreme_since_entry=100.0)
        assert not hasattr(out, "tp1_price")
        assert not hasattr(out, "take_profit_price")
        assert out.signal == Signal.HOLD   # a deep trail does not force an exit


class TestDegenerateInput:
    def test_zero_atr_holds_rather_than_dividing_by_zero(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="4h", tz="UTC")
        df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "volume": 1000.0}, index=idx)
        out = generate_breakout_signal(df, BreakoutConfig())
        assert out.signal == Signal.HOLD

    def test_insufficient_history_does_not_raise(self):
        df = _flat_series(5)
        out = generate_breakout_signal(df, BreakoutConfig(channel=20))
        assert out.signal in (Signal.HOLD, Signal.FLAT)
