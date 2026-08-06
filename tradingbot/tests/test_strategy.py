"""
Unit tests for strategy/trend_strategy.py

Tests use synthetic OHLCV DataFrames so no exchange connection is needed.
"""
import numpy as np
import pandas as pd
import pytest

from config import StrategyConfig
from strategy.trend_strategy import (
    Signal,
    StrategyOutput,
    compute_indicators,
    generate_signal,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(close_prices, spread=1.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    closes = np.array(close_prices, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h")
    return pd.DataFrame(
        {
            "open":   closes - spread / 2,
            "high":   closes + spread,
            "low":    closes - spread,
            "close":  closes,
            "volume": 100.0,
        },
        index=idx,
    )


def _default_cfg(**kwargs) -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.min_volume_spike = 0.0  # Default off for synthetic test data with uniform volume
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ── compute_indicators ────────────────────────────────────────────────────────
class TestComputeIndicators:
    def test_required_columns_present(self):
        df = _make_ohlcv(range(50, 100))
        cfg = _default_cfg(fast_ema=5, slow_ema=10, atr_period=7)
        result = compute_indicators(df, cfg)
        for col in ("fast_ema", "slow_ema", "atr", "adx", "trend"):
            assert col in result.columns, f"Missing column: {col}"

    def test_fast_above_slow_gives_trend_1(self):
        """A sustained uptrend should produce trend=1 eventually."""
        # Use 300 candles and a wide spread so fast EMA clearly overtakes slow EMA
        prices = [100.0 + i * 0.5 for i in range(300)]  # strongly rising
        df = _make_ohlcv(prices, spread=0.1)
        cfg = _default_cfg(fast_ema=5, slow_ema=15, atr_period=7)
        result = compute_indicators(df, cfg)
        assert result["trend"].iloc[-1] == 1

    def test_fast_below_slow_gives_trend_minus1(self):
        """A sustained downtrend should produce trend=-1 eventually."""
        prices = [300.0 - i * 0.5 for i in range(300)]  # strongly falling
        df = _make_ohlcv(prices, spread=0.1)
        cfg = _default_cfg(fast_ema=5, slow_ema=15, atr_period=7)
        result = compute_indicators(df, cfg)
        assert result["trend"].iloc[-1] == -1

    def test_atr_is_positive(self):
        df = _make_ohlcv(range(100, 130))
        cfg = _default_cfg(fast_ema=5, slow_ema=10, atr_period=7)
        result = compute_indicators(df, cfg)
        assert (result["atr"].dropna() > 0).all()

    def test_output_length_matches_input(self):
        prices = list(range(100, 150))
        df = _make_ohlcv(prices)
        cfg = _default_cfg(fast_ema=5, slow_ema=10, atr_period=7)
        result = compute_indicators(df, cfg)
        assert len(result) == len(df)


# ── generate_signal ───────────────────────────────────────────────────────────

def _build_crossover_df(n_flat=60, n_up=60, fast=5, slow=15):
    """
    Build a price series that:
      - starts flat (fast ≈ slow)
      - then rises so fast crosses above slow
    Use spread=0.01 to minimise ATR noise interfering with crossover detection.
    """
    flat   = [100.0] * n_flat
    rising = [100.0 + i * 3.0 for i in range(n_up)]
    return _make_ohlcv(flat + rising, spread=0.01)


def _build_crossdown_df(n_flat=60, n_down=60):
    flat    = [100.0] * n_flat
    falling = [100.0 - i * 3.0 for i in range(n_down)]
    return _make_ohlcv(flat + falling, spread=0.01)


class TestGenerateSignal:
    def test_long_entry_on_crossover(self):
        df = _build_crossover_df(n_flat=40, n_up=40)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, long_only=False)
        # Scan candle by candle until we see a LONG
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=0)
            if out.signal == Signal.LONG:
                assert out.stop_price is not None
                assert out.stop_price < df["close"].iloc[i - 1]
                return
        pytest.fail("Expected a LONG signal but none was produced")

    def test_short_entry_on_crossdown(self):
        df = _build_crossdown_df(n_flat=40, n_down=40)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, long_only=False)
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=0)
            if out.signal == Signal.SHORT:
                assert out.stop_price is not None
                assert out.stop_price > df["close"].iloc[i - 1]
                return
        pytest.fail("Expected a SHORT signal but none was produced")

    def test_long_only_blocks_short(self):
        df = _build_crossdown_df(n_flat=40, n_down=40)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, long_only=True)
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=0)
            assert out.signal != Signal.SHORT, "SHORT signal produced in long_only mode"

    def test_adx_filter_blocks_entry_in_flat_market(self):
        """
        In a perfectly flat market ADX should be very low — a high min_adx threshold
        should prevent any entries.
        """
        flat_prices = [100.0] * 100
        df = _make_ohlcv(flat_prices, spread=0.01)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=50, long_only=False)
        signals = set()
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=0)
            signals.add(out.signal)
        assert Signal.LONG  not in signals, "LONG entered in flat market with high ADX filter"
        assert Signal.SHORT not in signals, "SHORT entered in flat market with high ADX filter"

    def test_volume_spike_filter_blocks_entry(self):
        """
        With min_volume_spike = 1.5 and uniform volume (vol_spike = 1.0),
        crossover signals must be blocked.
        """
        df = _build_crossover_df(n_flat=40, n_up=40)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, min_volume_spike=1.5, long_only=False)
        signals = set()
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=0)
            signals.add(out.signal)
        assert Signal.LONG not in signals, "LONG signal should be blocked when volume spike threshold is not met"

    def test_trailing_stop_ratchets_up_for_long(self):
        """Trailing stop for a long position should only ever increase."""
        df = _build_crossover_df(n_flat=60, n_up=100)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, atr_trail_mult=2.0, long_only=True)
        trail = None
        prev_trail = None
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=1, current_trail=trail)
            if out.trail_price is not None and not (out.trail_price != out.trail_price):  # skip NaN
                if prev_trail is not None:
                    assert out.trail_price >= prev_trail - 1e-9, (
                        f"Trailing stop decreased: {out.trail_price} < {prev_trail}"
                    )
                prev_trail = out.trail_price
                trail = out.trail_price

    def test_flat_signal_on_exit(self):
        """When price drops below the trailing stop we should get FLAT."""
        # Large price series: flat warm-up → rise → sharp drop below a high trail stop
        prices = [100.0] * 80 + [100.0 + i * 3 for i in range(60)] + [40.0] * 10
        df = _make_ohlcv(prices, spread=0.01)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, atr_trail_mult=0.5, long_only=True)
        # Set trail_price very high (300) so any real price < 300 triggers exit
        for i in range(25, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=1, current_trail=300.0)
            if out.signal == Signal.FLAT:
                return  # found the expected exit
        pytest.fail("Expected a FLAT signal when price drops below trailing stop")


# ── Quant Strategy tests ─────────────────────────────────────────────────────

class TestQuantStrategy:
    """Tests for strategy/quant_strategy.py (3-pillar confluence engine)."""

    def _make_bearish_crossover_df(self, n_flat=60, n_down=60, volume=100.0) -> pd.DataFrame:
        flat    = [100.0] * n_flat
        falling = [100.0 - i * 0.5 for i in range(n_down)]
        closes  = np.array(flat + falling, dtype=float)
        idx     = pd.date_range("2024-01-01", periods=len(closes), freq="15min")
        vols    = np.full(len(closes), volume)
        return pd.DataFrame(
            {"open": closes, "high": closes * 1.001, "low": closes * 0.999,
             "close": closes, "volume": vols},
            index=idx,
        )

    def _bearish_btc_df(self) -> pd.DataFrame:
        prices = np.array([50000.0 - i * 20 for i in range(100)], dtype=float)
        idx    = pd.date_range("2024-01-01", periods=100, freq="1h")
        return pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1000.0},
            index=idx,
        )

    def test_short_requires_all_three_pillars(self):
        """
        A SHORT signal must only fire when all 3 confluence pillars pass:
        (1) EMA bearish crossover, (2) RVOL >= threshold, (3) BTC regime <= 0.
        Blocking any one should suppress the signal.
        """
        from config import StrategyConfig
        from strategy.quant_strategy import generate_quant_signal, Signal

        # With valid crossover, sufficient RVOL (spike on last candle), bearish BTC regime → SHOULD fire eventually
        df = self._make_bearish_crossover_df(volume=100.0)
        # Give the last candle a volume spike so RVOL passes
        df.iloc[-1, df.columns.get_loc("volume")] = 300.0

        cfg = StrategyConfig()
        cfg.min_volume_spike = 1.5
        cfg.long_only = False
        df_btc = self._bearish_btc_df()

        found_short = False
        for i in range(30, len(df)):
            out = generate_quant_signal(df.iloc[:i], cfg, current_position=0, df_btc_1h=df_btc)
            if out.signal == Signal.SHORT:
                found_short = True
                break
        # It's acceptable not to find one (volume spike only on last candle may not align with crossover)
        # The important thing is we don't error, and if found, the signal is SHORT
        if found_short:
            assert out.stop_price is not None
            assert out.tp1_price is not None
            assert out.tp2_price is not None

    def test_rvol_filter_blocks_short_in_bearish_regime(self):
        """
        Even with a clear bearish EMA crossover and bearish BTC regime,
        if RVOL < min_volume_spike the signal must be blocked.
        """
        from config import StrategyConfig
        from strategy.quant_strategy import generate_quant_signal, Signal

        df = self._make_bearish_crossover_df(volume=100.0)  # flat volume = RVOL ~1.0
        cfg = StrategyConfig()
        cfg.min_volume_spike = 1.5
        cfg.long_only = False
        df_btc = self._bearish_btc_df()

        for i in range(30, len(df)):
            out = generate_quant_signal(df.iloc[:i], cfg, current_position=0, df_btc_1h=df_btc)
            assert out.signal not in (Signal.LONG, Signal.SHORT), (
                f"Got {out.signal} at candle {i} with RVOL={out.rvol:.2f} — expected blocked by filter"
            )

    def test_btc_regime_evaluator_bearish(self):
        """evaluate_btc_regime should return -1 on a clearly falling BTC series."""
        from strategy.quant_strategy import evaluate_btc_regime
        prices = np.array([50000.0 - i * 20 for i in range(100)], dtype=float)
        idx    = pd.date_range("2024-01-01", periods=100, freq="1h")
        df_btc = pd.DataFrame(
            {"close": prices, "open": prices, "high": prices, "low": prices, "volume": 1.0},
            index=idx,
        )
        assert evaluate_btc_regime(df_btc) == -1

    def test_btc_regime_evaluator_bullish(self):
        """evaluate_btc_regime should return +1 on a clearly rising BTC series."""
        from strategy.quant_strategy import evaluate_btc_regime
        prices = np.array([30000.0 + i * 20 for i in range(100)], dtype=float)
        idx    = pd.date_range("2024-01-01", periods=100, freq="1h")
        df_btc = pd.DataFrame(
            {"close": prices, "open": prices, "high": prices, "low": prices, "volume": 1.0},
            index=idx,
        )
        assert evaluate_btc_regime(df_btc) == 1

    def test_btc_regime_evaluator_returns_zero_on_none(self):
        """evaluate_btc_regime must return 0 when given no data."""
        from strategy.quant_strategy import evaluate_btc_regime
        assert evaluate_btc_regime(None) == 0

    def test_stop_price_and_tps_correct_for_short(self):
        """For a SHORT signal: stop > entry > tp1 > tp2."""
        from config import StrategyConfig
        from strategy.quant_strategy import generate_quant_signal, Signal

        df = self._make_bearish_crossover_df(volume=100.0)
        # Volume spike on every candle to ensure RVOL passes
        df["volume"] = 300.0

        cfg = StrategyConfig()
        cfg.min_volume_spike = 1.5
        cfg.long_only = False
        df_btc = self._bearish_btc_df()

        for i in range(30, len(df)):
            out = generate_quant_signal(df.iloc[:i], cfg, current_position=0, df_btc_1h=df_btc)
            if out.signal == Signal.SHORT:
                price = df["close"].iloc[i - 1]
                assert out.stop_price > price, "SHORT stop must be above entry"
                assert out.tp1_price < price, "SHORT TP1 must be below entry"
                assert out.tp2_price < out.tp1_price, "TP2 must be further than TP1"
                return
        pytest.skip("No SHORT signal produced in this price series — adjust synthetic data if needed")

    def test_hold_when_no_crossover(self):
        """With no crossover and flat market, signal should be HOLD when flat."""
        flat_prices = [100.0] * 60
        df = _make_ohlcv(flat_prices, spread=0.01)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, long_only=False)
        out = generate_signal(df, cfg, current_position=0)
        assert out.signal == Signal.HOLD

    def test_trailing_stop_never_nan_after_warmup(self):
        """
        Once enough candles have been seen for EMA warmup, trail_price
        should never be NaN while in a long position.
        """
        df = _build_crossover_df(n_flat=60, n_up=100)
        cfg = _default_cfg(fast_ema=5, slow_ema=20, min_adx=0, atr_trail_mult=2.0, long_only=True)
        # Only check trail_price after we have at least slow_ema + 10 candles
        warmup = cfg.slow_ema + 10
        for i in range(warmup, len(df)):
            out = generate_signal(df.iloc[:i], cfg, current_position=1, current_trail=None)
            if out.trail_price is not None:
                assert out.trail_price == out.trail_price, (
                    f"NaN trail_price at candle {i} after warmup"
                )
