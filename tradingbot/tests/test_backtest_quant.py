"""Tests for the quant scanner backtest engine."""
import numpy as np
import pandas as pd
import pytest

from backtest.run_quant import btc_regime_series, run_quant_backtest, summarize
from strategy.quant_strategy import evaluate_btc_regime


def _klines(n: int, start: float = 100.0, drift: float = 0.0, vol: float = 1.0,
            freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = start + np.cumsum(rng.normal(drift, vol, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open":   close,
        "high":   close + vol,
        "low":    close - vol,
        "close":  close,
        "volume": rng.uniform(500, 2000, n),
    }, index=idx).rename_axis("timestamp")


class TestBtcRegimeSeries:
    def test_matches_scalar_implementation(self):
        """The vectorised regime series must agree with evaluate_btc_regime."""
        df = _klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h")
        series = btc_regime_series(df)
        # evaluate_btc_regime looks at the last row of whatever it is given,
        # so slicing to i+1 rows reproduces the regime as of bar i.
        for i in (60, 120, 199):
            assert series.iloc[i] == evaluate_btc_regime(df.iloc[: i + 1])

    def test_returns_neutral_during_warmup(self):
        df = _klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h")
        assert (btc_regime_series(df).iloc[:30] == 0).all()

    def test_values_are_only_minus_one_zero_one(self):
        df = _klines(300, start=60000.0, drift=-3.0, vol=80.0, freq="1h")
        assert set(btc_regime_series(df).unique()).issubset({-1, 0, 1})


class TestRunQuantBacktest:
    def test_runs_and_returns_frames(self):
        data = {"BTCUSDT": _klines(400, start=60000.0, drift=8.0, vol=60.0)}
        regime = btc_regime_series(_klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h"))
        eq, trades = run_quant_backtest(data, regime, starting_equity=10_000.0)
        assert len(eq) > 0
        assert "equity" in eq.columns
        assert isinstance(trades, pd.DataFrame)

    def test_respects_max_active_trades(self):
        data = {s: _klines(400, start=100.0 + i, drift=0.4, vol=1.5)
                for i, s in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"])}
        regime = btc_regime_series(_klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h"))
        _, trades = run_quant_backtest(data, regime, max_active_trades=1)
        if len(trades) == 0:
            pytest.skip("no trades generated for this fixture")
        # Walk the ledger: concurrent open positions must never exceed the cap.
        open_now, peak = 0, 0
        for _, row in trades.sort_values("time").iterrows():
            open_now += 1 if row["action"] == "ENTER" else 0
            if row["action"] == "EXIT" and row["reason"] != "TP1":
                open_now -= 1   # TP1 is a partial scale-out, position stays open
            peak = max(peak, open_now)
        assert peak <= 1

    def test_every_exit_has_a_reason(self):
        data = {"BTCUSDT": _klines(400, start=60000.0, drift=8.0, vol=60.0)}
        regime = btc_regime_series(_klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h"))
        _, trades = run_quant_backtest(data, regime)
        if len(trades) == 0:
            pytest.skip("no trades generated for this fixture")
        exits = trades[trades["action"] == "EXIT"]
        assert exits["reason"].isin({"SL", "BE", "TP1", "TP2", "TIMEOUT", "SIGNAL"}).all()

    def test_max_hold_timeout_is_enforced(self):
        data = {"BTCUSDT": _klines(600, start=60000.0, drift=1.0, vol=40.0)}
        regime = btc_regime_series(_klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h"))
        _, trades = run_quant_backtest(data, regime, max_hold_hours=1.0)
        if len(trades) == 0:
            pytest.skip("no trades generated for this fixture")
        exits = trades[trades["action"] == "EXIT"]
        # Nothing may outlive the cap (small tolerance for the closing bar itself)
        assert (exits["duration_h"] <= 1.0 + 0.26).all()

    def test_costs_reduce_returns(self):
        """A costly run must never finish above a free one on identical data."""
        data = {"BTCUSDT": _klines(500, start=60000.0, drift=6.0, vol=55.0)}
        regime = btc_regime_series(_klines(200, start=60000.0, drift=5.0, vol=50.0, freq="1h"))
        eq_free, t_free = run_quant_backtest(data, regime, fee_rate=0, slippage_pct=0, funding_rate_8h=0)
        eq_cost, _ = run_quant_backtest(data, regime, fee_rate=0.00055, slippage_pct=0.001)
        if len(t_free) == 0:
            pytest.skip("no trades generated for this fixture")
        assert eq_cost["equity"].iloc[-1] <= eq_free["equity"].iloc[-1]


class TestWalkForwardSplit:
    def test_splits_all_symbols_at_the_same_instant(self):
        """A per-symbol row-count split would leak future data between symbols."""
        from sweep_quant import split_data
        data = {
            "AAAUSDT": _klines(400),
            "BBBUSDT": _klines(250),   # deliberately shorter history
        }
        is_data, oos_data, cut = split_data(data, 0.7)
        for sym in data:
            if len(is_data[sym]):
                assert is_data[sym].index.max() <= cut
            if len(oos_data[sym]):
                assert oos_data[sym].index.min() > cut

    def test_split_loses_no_rows(self):
        from sweep_quant import split_data
        data = {"AAAUSDT": _klines(400), "BBBUSDT": _klines(400)}
        is_data, oos_data, _ = split_data(data, 0.7)
        for sym in data:
            assert len(is_data[sym]) + len(oos_data[sym]) == len(data[sym])


class TestSummarize:
    def test_handles_zero_trades(self):
        eq = pd.DataFrame({"equity": [10_000.0, 10_000.0]},
                          index=pd.date_range("2026-01-01", periods=2, freq="15min"))
        out = summarize(eq, pd.DataFrame(), 10_000.0)
        assert out["n_trades"] == 0
