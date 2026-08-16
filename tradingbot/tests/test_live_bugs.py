"""
Regression tests for bugs found in live Telegram/scanner logs (Aug 2026).

Each of these was observed in production output, not hypothesised.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from config import AlertConfig, RiskConfig
from live.alerts import Alerter
from risk.risk_manager import RiskManager


class TestDailySummaryReportsTheDayItSummarises:
    """
    Every summary sent said "No trades closed today" regardless of activity,
    because the rotate-on-new-day call ran before the counters were read.
    """

    def _alerter(self):
        return Alerter(AlertConfig(telegram_bot_token="", telegram_chat_id=""))

    def test_summary_includes_trades_recorded_that_day(self, monkeypatch):
        a = self._alerter()
        a.record_trade(100.0)
        a.record_trade(-40.0)
        sent = {}
        monkeypatch.setattr(a, "send", lambda msg, force=False: sent.update(msg=msg))
        a.send_daily_summary()
        assert "no trades closed" not in sent["msg"].lower()
        assert "Trades: 2" in sent["msg"]
        assert "60.00" in sent["msg"]          # net pnl

    def test_summary_at_day_turn_still_reports_the_finished_day(self, monkeypatch):
        """The exact production case: summary fires after midnight."""
        a = self._alerter()
        a.record_trade(25.0)
        # Pretend the recorded day was yesterday, as it is when the turn-of-day
        # summary fires just after midnight.
        a._daily["date"] = a._daily["date"] - timedelta(days=1)
        sent = {}
        monkeypatch.setattr(a, "send", lambda msg, force=False: sent.update(msg=msg))
        a.send_daily_summary()
        assert "Trades: 1" in sent["msg"]
        assert "no trades closed" not in sent["msg"].lower()

    def test_counters_still_roll_over_on_the_next_trade(self):
        a = self._alerter()
        a.record_trade(10.0)
        a._daily["date"] = a._daily["date"] - timedelta(days=1)
        a.record_trade(5.0)                      # new day -> fresh counters
        assert a._daily["trades"] == 1
        assert a._daily["total_pnl"] == 5.0

    def test_empty_day_message_names_the_date(self, monkeypatch):
        a = self._alerter()
        sent = {}
        monkeypatch.setattr(a, "send", lambda msg, force=False: sent.update(msg=msg))
        a.send_daily_summary()
        assert str(a._daily["date"]) in sent["msg"]


class TestStopDistanceBounds:
    """
    A live SOLUSDT signal produced entry 76.41 with stop 33.00 -- a 57% stop,
    far beyond the ~20% liquidation distance at 5x, so it could never fire.
    """

    def _rm(self, **kw):
        return RiskManager(RiskConfig(**kw))

    def test_absurdly_wide_stop_is_rejected(self):
        rm = self._rm(max_stop_pct=0.15)
        out = rm.size_position(10_000.0, 76.41, 33.00, 0, datetime.now())
        assert not out.approved
        assert "maximum" in out.reason

    def test_the_exact_production_case_is_rejected(self):
        rm = self._rm()
        out = rm.size_position(10_000.0, 76.41, 33.00, 0, datetime.now())
        assert not out.approved

    def test_normal_stop_still_approved(self):
        rm = self._rm()
        out = rm.size_position(10_000.0, 100.0, 98.0, 0, datetime.now())   # 2%
        assert out.approved

    def test_boundary_stop_is_allowed(self):
        rm = self._rm(max_stop_pct=0.15)
        out = rm.size_position(10_000.0, 100.0, 85.0, 0, datetime.now())   # exactly 15%
        assert out.approved

    def test_short_side_wide_stop_also_rejected(self):
        rm = self._rm(max_stop_pct=0.15)
        out = rm.size_position(10_000.0, 100.0, 140.0, 0, datetime.now())
        assert not out.approved

    def test_zero_disables_the_upper_bound(self):
        rm = self._rm(max_stop_pct=0.0)
        assert rm.size_position(10_000.0, 76.41, 33.00, 0, datetime.now()).approved

    def test_too_tight_stop_still_rejected(self):
        rm = self._rm()
        out = rm.size_position(10_000.0, 100.0, 99.99, 0, datetime.now())
        assert not out.approved
        assert "minimum" in out.reason


class TestScannerConflictDetection:
    """
    live/run.py adopted the scanner's BTCUSDT positions because both read
    position state from the exchange, producing 110093 and 110017 errors.
    """

    def _write(self, tmp_path, monkeypatch, payload):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "state").mkdir()
        (tmp_path / "state" / "scanner_state.json").write_text(json.dumps(payload))

    def test_fresh_scanner_state_is_detected(self, tmp_path, monkeypatch):
        from live.run import scanner_is_active
        self._write(tmp_path, monkeypatch, {
            "last_update": datetime.now().isoformat(),
            "active_trades": {"BTCUSDT": {}},
        })
        active, detail = scanner_is_active()
        assert active
        assert "BTCUSDT" in detail

    def test_stale_scanner_state_is_ignored(self, tmp_path, monkeypatch):
        from live.run import scanner_is_active
        self._write(tmp_path, monkeypatch, {
            "last_update": (datetime.now() - timedelta(hours=6)).isoformat(),
            "active_trades": {},
        })
        assert scanner_is_active()[0] is False

    def test_missing_state_file_is_not_a_conflict(self, tmp_path, monkeypatch):
        from live.run import scanner_is_active
        monkeypatch.chdir(tmp_path)
        assert scanner_is_active()[0] is False

    def test_corrupt_state_file_does_not_raise(self, tmp_path, monkeypatch):
        from live.run import scanner_is_active
        monkeypatch.chdir(tmp_path)
        (tmp_path / "state").mkdir()
        (tmp_path / "state" / "scanner_state.json").write_text("{not json")
        assert scanner_is_active()[0] is False

    def test_running_scanner_with_no_open_trades_still_conflicts(self, tmp_path, monkeypatch):
        """The scanner picks symbols dynamically, so an idle one is still a risk."""
        from live.run import scanner_is_active
        self._write(tmp_path, monkeypatch, {
            "last_update": datetime.now().isoformat(),
            "active_trades": {},
        })
        active, detail = scanner_is_active()
        assert active
        assert "none" in detail
