"""
Unit tests for risk/risk_manager.py

All tests pass an explicit `now` datetime to avoid any wall-clock dependence.
"""
from datetime import datetime, timedelta, date
import pytest

from config import RiskConfig
from risk.risk_manager import RiskManager, SizingResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rm(**kwargs) -> RiskManager:
    defaults = dict(
        account_risk_per_trade=0.01,   # 1% risk per trade
        max_position_pct=0.5,
        max_daily_loss_pct=0.05,       # 5% daily loss limit
        max_open_positions=2,
        leverage=2,
    )
    defaults.update(kwargs)
    return RiskManager(RiskConfig(**defaults))


NOW = datetime(2024, 6, 1, 12, 0, 0)
NEXT_DAY = datetime(2024, 6, 2, 12, 0, 0)


# ── size_position ─────────────────────────────────────────────────────────────

class TestSizePosition:
    def test_basic_approval(self):
        rm = _make_rm()
        result = rm.size_position(
            equity=10_000, entry_price=50_000, stop_price=49_000,
            open_positions=0, now=NOW
        )
        assert result.approved
        # dollar_risk = 10000 * 0.01 = 100; stop_distance = 1000; qty = 0.1
        assert abs(result.qty - 0.1) < 1e-9

    def test_qty_capped_by_max_notional(self):
        """max_position_pct * leverage caps the notional regardless of stop distance."""
        rm = _make_rm(account_risk_per_trade=0.5, max_position_pct=0.1, leverage=1)
        result = rm.size_position(
            equity=10_000, entry_price=100, stop_price=99,
            open_positions=0, now=NOW
        )
        # Uncapped: dollar_risk=5000, stop=1 → qty=5000
        # Cap: max_notional = 10000*0.1*1 = 1000; max_qty = 1000/100 = 10
        assert result.approved
        assert abs(result.qty - 10.0) < 1e-9

    def test_rejects_zero_stop_distance(self):
        rm = _make_rm()
        result = rm.size_position(
            equity=10_000, entry_price=50_000, stop_price=50_000,
            open_positions=0, now=NOW
        )
        assert not result.approved
        assert "stop" in result.reason.lower()

    def test_rejects_when_max_positions_reached(self):
        rm = _make_rm(max_open_positions=1)
        result = rm.size_position(
            equity=10_000, entry_price=50_000, stop_price=49_000,
            open_positions=1, now=NOW
        )
        assert not result.approved
        assert "position" in result.reason.lower()

    def test_rejects_zero_equity(self):
        rm = _make_rm()
        result = rm.size_position(
            equity=0, entry_price=50_000, stop_price=49_000,
            open_positions=0, now=NOW
        )
        # dollar_risk = 0 → qty = 0 → rejected
        assert not result.approved


# ── Kill switch ───────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_not_active_initially(self):
        rm = _make_rm()
        assert not rm.kill_switch_active(now=NOW)

    def test_activates_after_daily_loss_limit(self):
        rm = _make_rm(max_daily_loss_pct=0.05)
        equity = 10_000
        # Record a loss of 6% (> 5% threshold)
        rm.record_realized_pnl(-600, equity, now=NOW)
        assert rm.kill_switch_active(now=NOW)

    def test_does_not_activate_below_threshold(self):
        rm = _make_rm(max_daily_loss_pct=0.05)
        equity = 10_000
        rm.record_realized_pnl(-400, equity, now=NOW)  # only 4%
        assert not rm.kill_switch_active(now=NOW)

    def test_resets_on_new_day(self):
        rm = _make_rm(max_daily_loss_pct=0.05)
        equity = 10_000
        rm.record_realized_pnl(-600, equity, now=NOW)
        assert rm.kill_switch_active(now=NOW)
        # Next day — should reset
        assert not rm.kill_switch_active(now=NEXT_DAY)

    def test_blocks_trade_when_active(self):
        rm = _make_rm(max_daily_loss_pct=0.05)
        rm.record_realized_pnl(-600, 10_000, now=NOW)
        result = rm.size_position(
            equity=10_000, entry_price=50_000, stop_price=49_000,
            open_positions=0, now=NOW
        )
        assert not result.approved
        assert "kill" in result.reason.lower() or "daily" in result.reason.lower()

    def test_cumulative_losses_trigger_switch(self):
        """Multiple small losses within the same day accumulate."""
        rm = _make_rm(max_daily_loss_pct=0.05)
        equity = 10_000
        rm.record_realized_pnl(-200, equity, now=NOW)
        rm.record_realized_pnl(-200, equity, now=NOW)
        rm.record_realized_pnl(-200, equity, now=NOW)  # total: 600 = 6%
        assert rm.kill_switch_active(now=NOW)

    def test_profits_do_not_unlock_kill_switch_once_active(self):
        """Once the kill switch is active, subsequent profits should not deactivate it."""
        rm = _make_rm(max_daily_loss_pct=0.05)
        equity = 10_000
        # Trigger the kill switch with a big loss
        rm.record_realized_pnl(-600, equity, now=NOW)   # -6% -> switch activates
        assert rm.kill_switch_active(now=NOW)
        # Record a profit — kill switch should stay on for the rest of the day
        rm.record_realized_pnl(+1000, equity, now=NOW)  # profit does not reset the switch
        assert rm.kill_switch_active(now=NOW)


# ── reset_if_new_day ──────────────────────────────────────────────────────────

class TestResetIfNewDay:
    def test_same_day_no_reset(self):
        rm = _make_rm(max_daily_loss_pct=0.05)
        rm.record_realized_pnl(-600, 10_000, now=NOW)
        later_same_day = NOW + timedelta(hours=6)
        rm.reset_if_new_day(later_same_day)
        assert rm.kill_switch_active(now=later_same_day)

    def test_new_day_clears_pnl(self):
        rm = _make_rm(max_daily_loss_pct=0.05)
        rm.record_realized_pnl(-600, 10_000, now=NOW)
        rm.reset_if_new_day(NEXT_DAY)
        assert rm._daily_loss_tracker["realized_pnl"] == 0.0
        assert rm._daily_loss_tracker["date"] == NEXT_DAY.date()
