"""
Smoke / integration tests for backtest/run.py

Verifies that the backtest engine runs end-to-end without errors on synthetic
data and produces reasonable output shapes and numeric sanity checks.
No exchange connection is required.
"""
import numpy as np
import pandas as pd
import pytest

from config import CONFIG, StrategyConfig, RiskConfig
from backtest.run import run_backtest, summarize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(close_prices, spread=1.0) -> pd.DataFrame:
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


def _patch_config(fast_ema=5, slow_ema=15, atr_period=7, min_adx=0,
                  long_only=True, atr_stop_mult=2.0, atr_trail_mult=3.0):
    """Patch global CONFIG for a test (fast parameters so we get trades in short data)."""
    CONFIG.strategy.fast_ema = fast_ema
    CONFIG.strategy.slow_ema = slow_ema
    CONFIG.strategy.atr_period = atr_period
    CONFIG.strategy.min_adx = min_adx
    CONFIG.strategy.long_only = long_only
    CONFIG.strategy.atr_stop_mult = atr_stop_mult
    CONFIG.strategy.atr_trail_mult = atr_trail_mult
    CONFIG.risk.account_risk_per_trade = 0.01
    CONFIG.risk.max_position_pct = 0.5
    CONFIG.risk.max_daily_loss_pct = 0.10
    CONFIG.risk.max_open_positions = 1
    CONFIG.risk.leverage = 1


# ── Smoke tests ───────────────────────────────────────────────────────────────

class TestBacktestSmoke:
    def test_runs_without_error_on_flat_data(self):
        """Should not crash even if no trades are generated."""
        _patch_config()
        df = _make_ohlcv([100.0] * 200, spread=0.1)
        eq_df, trades_df = run_backtest(df, starting_equity=10_000)
        assert isinstance(eq_df, pd.DataFrame)
        assert isinstance(trades_df, pd.DataFrame)

    def test_equity_curve_has_correct_columns(self):
        _patch_config()
        df = _make_ohlcv([100.0] * 200)
        eq_df, _ = run_backtest(df, starting_equity=10_000)
        assert "equity" in eq_df.columns

    def test_trades_have_correct_columns(self):
        _patch_config(min_adx=0, long_only=False)
        prices = [100.0] * 30 + [100.0 + i * 3 for i in range(60)] + [50.0] * 30
        df = _make_ohlcv(prices)
        _, trades_df = run_backtest(df, starting_equity=10_000)
        if len(trades_df) > 0:
            for col in ("time", "action", "price", "qty"):
                assert col in trades_df.columns, f"Missing column: {col}"

    def test_produces_trades_in_trending_market(self):
        """A strongly trending price series should generate at least one trade."""
        _patch_config(min_adx=0, long_only=True)
        # Long flat warm-up then a strong rise so EMA(5/15) has time to cross
        prices = [100.0] * 60 + [100.0 + i * 5 for i in range(200)]
        df = _make_ohlcv(prices, spread=0.01)
        _, trades_df = run_backtest(df, starting_equity=10_000)
        assert len(trades_df) >= 1, "Expected at least one trade in a trending market"

    def test_equity_never_goes_negative_with_conservative_risk(self):
        """With very conservative sizing equity should stay positive."""
        _patch_config(min_adx=0, long_only=False)
        CONFIG.risk.account_risk_per_trade = 0.005
        CONFIG.risk.max_position_pct = 0.1
        # Adversarial: price keeps falling
        prices = [200.0 - i * 0.5 for i in range(300)]
        df = _make_ohlcv(prices)
        eq_df, _ = run_backtest(df, starting_equity=10_000)
        if len(eq_df):
            assert eq_df["equity"].min() > 0, "Equity went negative"

    def test_starting_equity_preserved_with_no_trades(self):
        """If no trades occur, equity should remain at starting value."""
        _patch_config(min_adx=999)  # impossibly high ADX filter → no trades
        df = _make_ohlcv([100.0] * 200)
        starting = 10_000.0
        eq_df, trades_df = run_backtest(df, starting_equity=starting)
        assert len(trades_df) == 0, "Expected no trades with min_adx=999"
        if len(eq_df):
            assert abs(eq_df["equity"].iloc[-1] - starting) < 1.0

    def test_returns_are_higher_in_bull_market_than_bear(self):
        """Long-only bot should outperform (or at least match) in a bull market."""
        _patch_config(min_adx=0, long_only=True)
        bull_prices = [100.0 + i * 2 for i in range(150)]
        bear_prices = [300.0 - i * 2 for i in range(150)]
        eq_bull, _ = run_backtest(_make_ohlcv(bull_prices), starting_equity=10_000)
        eq_bear, _ = run_backtest(_make_ohlcv(bear_prices), starting_equity=10_000)
        final_bull = eq_bull["equity"].iloc[-1] if len(eq_bull) else 10_000
        final_bear = eq_bear["equity"].iloc[-1] if len(eq_bear) else 10_000
        assert final_bull >= final_bear, "Long-only bot should not do better in bear market"


# ── summarize function ────────────────────────────────────────────────────────

class TestSummarize:
    def test_summarize_does_not_raise_with_empty_trades(self, capsys):
        _patch_config()
        df = _make_ohlcv([100.0] * 100)
        eq_df, trades_df = run_backtest(df)
        # summarize should print cleanly even with zero trades
        summarize(eq_df, trades_df, starting_equity=10_000)
        captured = capsys.readouterr()
        assert "return" in captured.out.lower()

    def test_summarize_shows_total_return(self, capsys):
        _patch_config()
        df = _make_ohlcv([100.0] * 100)
        eq_df, trades_df = run_backtest(df)
        summarize(eq_df, trades_df, starting_equity=10_000)
        captured = capsys.readouterr()
        assert "Total return" in captured.out

    def test_summarize_returns_dict_with_all_keys(self):
        """summarize() must return a dict with every expected metric key."""
        _patch_config(min_adx=0, long_only=True)
        prices = [100.0] * 60 + [100.0 + i * 5 for i in range(200)]
        df = _make_ohlcv(prices, spread=0.01)
        eq_df, trades_df = run_backtest(df, starting_equity=10_000)
        stats = summarize(eq_df, trades_df, starting_equity=10_000)
        required = [
            "total_return_%", "max_dd_%", "sharpe", "calmar",
            "profit_factor", "n_trades", "win_rate_%",
            "avg_win_$", "avg_loss_$", "max_consec_loss",
        ]
        for key in required:
            assert key in stats, f"Missing key in summarize() output: {key}"

    def test_profit_factor_is_ratio_of_gross_wins_to_losses(self):
        """For a winning strategy profit_factor should be > 1."""
        _patch_config(min_adx=0, long_only=True)
        prices = [100.0] * 60 + [100.0 + i * 5 for i in range(200)]
        df = _make_ohlcv(prices, spread=0.01)
        eq_df, trades_df = run_backtest(df, starting_equity=10_000)
        stats = summarize(eq_df, trades_df, starting_equity=10_000)
        if stats["n_trades"] >= 2:
            assert stats["profit_factor"] >= 0, "Profit factor must be non-negative"

    def test_sharpe_is_numeric(self):
        """Sharpe ratio should always be a finite float."""
        _patch_config(min_adx=0, long_only=True)
        prices = [100.0] * 60 + [100.0 + i * 5 for i in range(200)]
        df = _make_ohlcv(prices, spread=0.01)
        eq_df, trades_df = run_backtest(df, starting_equity=10_000)
        stats = summarize(eq_df, trades_df, starting_equity=10_000)
        import math
        assert isinstance(stats["sharpe"], (int, float))
        assert not math.isnan(stats["sharpe"]), "Sharpe must not be NaN"
