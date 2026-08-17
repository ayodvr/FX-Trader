"""Tests for cross-sectional momentum."""
import numpy as np
import pandas as pd
import pytest

from backtest.run_xsmom import run_xsmom_backtest, summarize
from strategy.xsmom_strategy import (
    XsMomConfig,
    compute_momentum,
    inverse_vol_weights,
    target_weights,
)


def _panel(specs: dict, n: int = 200) -> pd.DataFrame:
    """specs: {symbol: total_drift_over_window}"""
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {sym: np.linspace(100.0, 100.0 * (1 + d), n) for sym, d in specs.items()},
        index=idx,
    )


class TestMomentum:
    def test_ranks_by_formation_return(self):
        prices = _panel({"WIN": 0.5, "MID": 0.1, "LOSE": -0.3})
        mom = compute_momentum(prices, XsMomConfig(formation_bars=50))
        assert mom["WIN"] > mom["MID"] > mom["LOSE"]

    def test_skip_bars_excludes_recent_window(self):
        """A spike in the final bars must not count when skip_bars covers it."""
        prices = _panel({"A": 0.0}, n=100)
        prices.iloc[-5:] = 500.0
        with_skip = compute_momentum(prices, XsMomConfig(formation_bars=50, skip_bars=10))
        without   = compute_momentum(prices, XsMomConfig(formation_bars=50, skip_bars=0))
        assert without["A"] > with_skip["A"]

    def test_returns_empty_when_history_too_short(self):
        prices = _panel({"A": 0.1}, n=10)
        assert compute_momentum(prices, XsMomConfig(formation_bars=50)).empty


class TestTargetWeights:
    def test_longs_winners_and_shorts_losers(self):
        specs = {f"S{i}": (i - 5) * 0.1 for i in range(12)}   # S11 best, S0 worst
        w = target_weights(_panel(specs), XsMomConfig(formation_bars=50, n_long=3, n_short=3))
        assert w["S11"] > 0 and w["S0"] < 0

    def test_legs_are_dollar_neutral(self):
        specs = {f"S{i}": (i - 5) * 0.1 for i in range(12)}
        w = target_weights(_panel(specs), XsMomConfig(formation_bars=50, n_long=3, n_short=3))
        assert w[w > 0].sum() == pytest.approx(1.0)
        assert w[w < 0].sum() == pytest.approx(-1.0)

    def test_long_only_holds_no_shorts(self):
        specs = {f"S{i}": (i - 5) * 0.1 for i in range(12)}
        w = target_weights(_panel(specs), XsMomConfig(formation_bars=50, n_long=3, long_only=True))
        assert (w >= 0).all()
        assert w.sum() == pytest.approx(1.0)

    def test_refuses_a_universe_below_the_minimum(self):
        specs = {f"S{i}": i * 0.1 for i in range(4)}
        w = target_weights(_panel(specs), XsMomConfig(formation_bars=50, min_universe=10))
        assert w.empty

    def test_leg_size_capped_at_half_the_universe(self):
        specs = {f"S{i}": i * 0.1 for i in range(10)}
        w = target_weights(_panel(specs),
                           XsMomConfig(formation_bars=50, n_long=99, n_short=99, min_universe=5))
        assert len(w[w > 0]) <= 5 and len(w[w < 0]) <= 5


class TestInverseVolWeights:
    def test_quieter_symbol_gets_more_weight(self):
        idx = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC")
        rng = np.random.default_rng(0)
        rets = pd.DataFrame({"CALM": rng.normal(0, 0.001, 100),
                             "WILD": rng.normal(0, 0.05, 100)}, index=idx)
        w = inverse_vol_weights(rets, ["CALM", "WILD"], 100)
        assert w["CALM"] > w["WILD"]
        assert w.sum() == pytest.approx(1.0)

    def test_handles_zero_volatility_without_dividing_by_zero(self):
        idx = pd.date_range("2026-01-01", periods=50, freq="4h", tz="UTC")
        rets = pd.DataFrame({"FLAT": np.zeros(50), "MOVE": np.full(50, 0.01)}, index=idx)
        w = inverse_vol_weights(rets, ["FLAT", "MOVE"], 50)
        assert np.isfinite(w).all()


class TestEngine:
    def test_turnover_is_charged_and_reduces_equity(self):
        specs = {f"S{i}": (i - 5) * 0.05 for i in range(12)}
        prices = _panel(specs, n=400)
        cfg = XsMomConfig(formation_bars=42, holding_bars=42, n_long=3, n_short=3)
        free, _ = run_xsmom_backtest(prices, cfg, fee_rate=0, slippage_pct=0,
                                     funding_rate_8h=0, quiet=True)
        paid, st = run_xsmom_backtest(prices, cfg, fee_rate=0.001, slippage_pct=0.001,
                                      funding_rate_8h=0, quiet=True)
        assert st["rebalances"] > 0
        assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]

    def test_rebalance_count_follows_holding_period(self):
        prices = _panel({f"S{i}": (i - 5) * 0.05 for i in range(12)}, n=600)
        short_hold = run_xsmom_backtest(prices, XsMomConfig(formation_bars=42, holding_bars=42),
                                        quiet=True)[1]
        long_hold  = run_xsmom_backtest(prices, XsMomConfig(formation_bars=42, holding_bars=168),
                                        quiet=True)[1]
        assert short_hold["rebalances"] > long_hold["rebalances"]

    def test_equity_curve_covers_every_bar(self):
        prices = _panel({f"S{i}": (i - 5) * 0.05 for i in range(12)}, n=300)
        eq, _ = run_xsmom_backtest(prices, XsMomConfig(formation_bars=42, holding_bars=42),
                                   quiet=True)
        assert len(eq) == len(prices)

    def test_dollar_neutral_book_is_insulated_from_a_market_wide_move(self):
        """Every symbol rallying together should not create a large P&L either way."""
        idx = pd.date_range("2026-01-01", periods=400, freq="4h", tz="UTC")
        base = np.linspace(100, 300, 400)          # everything triples
        jitter = {f"S{i}": base * (1 + i * 0.001) for i in range(12)}
        prices = pd.DataFrame(jitter, index=idx)
        eq, _ = run_xsmom_backtest(prices, XsMomConfig(formation_bars=42, holding_bars=42,
                                                       n_long=3, n_short=3),
                                   fee_rate=0, slippage_pct=0, funding_rate_8h=0, quiet=True)
        assert abs(eq["equity"].iloc[-1] / 10_000.0 - 1) < 0.5
