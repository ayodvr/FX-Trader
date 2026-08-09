"""Tests for the funding-rate carry analysis (pure math only, no network)."""
import numpy as np
import pytest
import pandas as pd

from analyze_funding import PERIODS_PER_YEAR, analyse, net_yield


def _funding(rates) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(rates), freq="8h", tz="UTC")
    return pd.DataFrame({"funding_rate": rates}, index=idx)


class TestAnalyse:
    def test_annualises_over_the_right_number_of_periods(self):
        df = _funding([0.0001] * 300)          # 0.01% every 8h
        out = analyse("TEST", df, round_trip_cost=0.003)
        assert out["gross_annual_%"] == pytest.approx(0.0001 * PERIODS_PER_YEAR * 100)
        assert out["gross_annual_%"] == pytest.approx(10.95)

    def test_counts_negative_share_and_longest_run(self):
        df = _funding([0.001, -0.001, -0.001, -0.001, 0.001, -0.001])
        out = analyse("TEST", df, round_trip_cost=0.003)
        assert out["neg_share_%"] == 4 / 6 * 100
        assert out["longest_neg_run"] == 3

    def test_worst_cumulative_drawdown_captures_negative_stretch(self):
        # rises, then gives back 0.003 of cumulative funding
        df = _funding([0.001] * 5 + [-0.001] * 3)
        out = analyse("TEST", df, round_trip_cost=0.003)
        assert round(out["worst_cum_dd_%"], 6) == round(-0.003 * 100, 6)

    def test_breakeven_is_infinite_when_funding_is_net_negative(self):
        df = _funding([-0.0001] * 100)
        out = analyse("TEST", df, round_trip_cost=0.003)
        assert out["breakeven_days"] == float("inf")

    def test_breakeven_days_matches_hand_calculation(self):
        # 0.01%/8h = 0.03%/day; a 0.3% round trip needs 10 days
        df = _funding([0.0001] * 100)
        out = analyse("TEST", df, round_trip_cost=0.003)
        assert round(out["breakeven_days"], 6) == 10.0


class TestNetYield:
    def test_short_holds_are_penalised_by_the_round_trip(self):
        df = _funding([0.0001] * 300)
        assert net_yield(df, 7, 0.003) < net_yield(df, 90, 0.003)

    def test_zero_cost_yield_is_independent_of_holding_period(self):
        df = _funding([0.0001] * 300)
        assert round(net_yield(df, 7, 0.0), 6) == round(net_yield(df, 365, 0.0), 6)

    def test_yield_is_negative_below_breakeven_hold(self):
        df = _funding([0.0001] * 300)   # breakeven is 10 days
        assert net_yield(df, 5, 0.003) < 0
        assert net_yield(df, 30, 0.003) > 0
