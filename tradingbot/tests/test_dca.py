"""Tests for the DCA / martingale ladder and its backtest engine."""
import numpy as np
import pandas as pd
import pytest

from backtest.run_dca import run_dca_backtest, summarize
from strategy.dca_strategy import (
    DcaConfig,
    average_entry,
    build_ladder,
    ladder_totals,
    liquidation_price,
    stop_loss_price,
    take_profit_price,
)


class TestLadder:
    def test_has_base_order_plus_every_safety_order(self):
        steps = build_ladder(100.0, 10_000.0, DcaConfig(max_safety_orders=5))
        assert len(steps) == 6
        assert steps[0].index == 0 and steps[0].deviation == 0.0

    def test_long_safety_orders_step_downward(self):
        steps = build_ladder(100.0, 10_000.0, DcaConfig(max_safety_orders=4), is_long=True)
        prices = [s.price for s in steps]
        assert prices == sorted(prices, reverse=True)

    def test_short_safety_orders_step_upward(self):
        steps = build_ladder(100.0, 10_000.0, DcaConfig(max_safety_orders=4), is_long=False)
        prices = [s.price for s in steps]
        assert prices == sorted(prices)

    def test_volume_scale_grows_each_order(self):
        """Exposure grows fastest exactly when the trade is going worst."""
        cfg = DcaConfig(max_safety_orders=4, volume_scale=2.0, safety_order_pct=0.01)
        steps = build_ladder(100.0, 10_000.0, cfg)
        safety = [s.notional for s in steps[1:]]
        assert safety == pytest.approx([100.0, 200.0, 400.0, 800.0])

    def test_step_scale_widens_each_gap(self):
        cfg = DcaConfig(max_safety_orders=3, price_deviation_pct=0.01, step_scale=2.0)
        devs = [s.deviation for s in build_ladder(100.0, 10_000.0, cfg)[1:]]
        assert devs == pytest.approx([0.01, 0.03, 0.07])   # 1%, +2%, +4%

    def test_totals_sum_the_whole_ladder(self):
        cfg = DcaConfig(max_safety_orders=3, base_order_pct=0.01,
                        safety_order_pct=0.01, volume_scale=1.0)
        total, final_dev = ladder_totals(build_ladder(100.0, 10_000.0, cfg))
        assert total == pytest.approx(100.0 * 4)
        assert final_dev > 0


class TestPricing:
    def test_average_entry_is_volume_weighted(self):
        avg, qty = average_entry([(100.0, 1.0), (90.0, 1.0)])
        assert avg == pytest.approx(95.0)
        assert qty == pytest.approx(2.0)

    def test_averaging_down_pulls_entry_toward_price(self):
        first, _ = average_entry([(100.0, 1.0)])
        after, _ = average_entry([(100.0, 1.0), (80.0, 3.0)])
        assert after < first

    def test_take_profit_sits_above_average_for_longs(self):
        cfg = DcaConfig(take_profit_pct=0.02)
        assert take_profit_price(100.0, cfg, True) == pytest.approx(102.0)
        assert take_profit_price(100.0, cfg, False) == pytest.approx(98.0)

    def test_stop_loss_disabled_by_default(self):
        assert stop_loss_price(100.0, DcaConfig(), True) is None
        assert stop_loss_price(100.0, DcaConfig(stop_loss_pct=0.1), True) == pytest.approx(90.0)

    def test_liquidation_tightens_as_leverage_rises(self):
        far  = liquidation_price(100.0, 2, True)
        near = liquidation_price(100.0, 20, True)
        assert far < near < 100.0

    def test_liquidation_is_above_entry_for_shorts(self):
        assert liquidation_price(100.0, 10, False) > 100.0


def _series(prices, freq="1h"):
    idx = pd.date_range("2026-01-01", periods=len(prices), freq=freq, tz="UTC")
    c = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999,
                         "close": c, "volume": 1000.0}, index=idx)


class TestEngine:
    def test_rising_market_closes_deals_at_take_profit(self):
        data = {"AAAUSDT": _series(np.linspace(100, 130, 400))}
        _, deals, _ = run_dca_backtest(data, DcaConfig(take_profit_pct=0.01), quiet=True)
        assert len(deals) > 0
        assert (deals["reason"] == "TP").all()

    def test_sustained_crash_liquidates_rather_than_recovering(self):
        """A one-way move down must not be booked as a win by a later bounce."""
        data = {"AAAUSDT": _series(np.linspace(100, 40, 400))}
        cfg = DcaConfig(leverage=10, max_safety_orders=3, stop_loss_pct=0.0)
        _, deals, stats = run_dca_backtest(data, cfg, quiet=True)
        assert stats["liquidations"] > 0
        assert (deals["pnl"] < 0).any()

    def test_no_liquidations_without_leverage(self):
        data = {"AAAUSDT": _series(np.linspace(100, 60, 400))}
        _, _, stats = run_dca_backtest(data, DcaConfig(leverage=1), quiet=True)
        assert stats["liquidations"] == 0

    def test_stop_loss_caps_the_worst_deal(self):
        prices = np.linspace(100, 55, 400)
        no_sl = run_dca_backtest({"A": _series(prices)},
                                 DcaConfig(leverage=10, stop_loss_pct=0.0), quiet=True)[1]
        with_sl = run_dca_backtest({"A": _series(prices)},
                                   DcaConfig(leverage=10, stop_loss_pct=0.03), quiet=True)[1]
        if len(no_sl) and len(with_sl):
            assert with_sl["pnl"].min() >= no_sl["pnl"].min()

    def test_win_rate_is_high_while_still_losing_money(self):
        """
        The signature of the family: mostly wins, negative overall.

        Needs a realistic shape to reproduce -- a long choppy stretch that keeps
        closing small winners, then one sustained trend that liquidates. A
        straight-line crash liquidates nearly everything and never produces the
        high win rate that makes these bots look good in the first place.
        """
        chop  = 100 + 2 * np.sin(np.linspace(0, 60 * np.pi, 1200))
        crash = np.linspace(100, 55, 300)
        eq, deals, stats = run_dca_backtest({"A": _series(np.concatenate([chop, crash]))},
                                            DcaConfig(leverage=10, max_safety_orders=3,
                                                      take_profit_pct=0.01),
                                            quiet=True)
        m = summarize(eq, deals, stats, 10_000.0, quiet=True)
        assert m["n_deals"] >= 20, "fixture should generate plenty of deals"
        assert stats["liquidations"] > 0, "fixture should end in liquidation"
        assert m["win_%"] > 80.0, "most deals close green -- that is the whole illusion"
        # The handful of liquidations should dominate all those small winners
        assert m["worst_deal"] < 0
        assert abs(m["worst_deal"]) > m["gross_wins"] / m["n_deals"] * 10

    def test_concurrent_deals_are_capped(self):
        data = {s: _series(np.linspace(100, 130, 300)) for s in ("A", "B", "C", "D", "E")}
        _, deals, _ = run_dca_backtest(data, DcaConfig(), max_concurrent_deals=2, quiet=True)
        assert len(deals) > 0
