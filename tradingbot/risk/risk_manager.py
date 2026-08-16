"""
Risk management — deliberately separate from strategy logic.

The strategy decides DIRECTION (long/short/flat). This module decides SIZE,
and can VETO any trade regardless of what the strategy wants. That separation
is intentional: a strategy bug should never be able to bypass risk limits.

IMPORTANT: every method takes an explicit `now` (datetime) parameter rather than
reading the system clock internally. This is required so the same code works
correctly in both live trading (pass datetime.now()) and backtesting (pass the
simulated candle timestamp) — using the real system clock inside a backtest
caused the kill switch to never reset, silently halting the simulation early.
"""
from dataclasses import dataclass
from datetime import datetime

from config import RiskConfig


@dataclass
class SizingResult:
    approved: bool
    qty: float = 0.0
    take_profit_price: float = 0.0
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._daily_loss_tracker = {"date": None, "realized_pnl": 0.0}
        self._kill_switch_active = False

    def reset_if_new_day(self, now: datetime):
        today = now.date()
        if self._daily_loss_tracker["date"] != today:
            self._daily_loss_tracker = {"date": today, "realized_pnl": 0.0}
            self._kill_switch_active = False

    def record_realized_pnl(self, pnl: float, equity: float, now: datetime):
        self.reset_if_new_day(now)
        self._daily_loss_tracker["realized_pnl"] += pnl
        loss_pct = -self._daily_loss_tracker["realized_pnl"] / equity if equity > 0 else 0
        if loss_pct >= self.cfg.max_daily_loss_pct:
            self._kill_switch_active = True

    def size_position(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        open_positions: int,
        now: datetime,
    ) -> SizingResult:
        self.reset_if_new_day(now)

        if self._kill_switch_active:
            return SizingResult(False, reason="Daily loss limit hit — trading halted until next day")

        if open_positions >= self.cfg.max_open_positions:
            return SizingResult(False, reason="Max open positions reached")

        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return SizingResult(False, reason="Invalid stop distance")

        # Reject stops that are too tight relative to entry price.
        # Sub-0.3% stops at leverage get blown through by normal crypto noise.
        min_stop_pct = getattr(self.cfg, "min_stop_pct", 0.003)
        stop_pct = stop_distance / entry_price if entry_price > 0 else 0.0
        if stop_pct < min_stop_pct:
            return SizingResult(
                False,
                reason=f"Stop distance {stop_pct*100:.3f}% < minimum {min_stop_pct*100:.3f}% — too tight for leverage"
            )

        # A stop wider than the liquidation distance is not a stop -- the position
        # gets liquidated before price ever reaches it. Usually a symptom of a
        # corrupt ATR from a bad candle rather than a real volatility regime.
        max_stop_pct = getattr(self.cfg, "max_stop_pct", 0.15)
        if max_stop_pct > 0 and stop_pct > max_stop_pct:
            return SizingResult(
                False,
                reason=f"Stop distance {stop_pct*100:.2f}% > maximum {max_stop_pct*100:.2f}% — "
                       f"beyond liquidation range at {self.cfg.leverage}x (likely bad ATR)"
            )

        dollar_risk = equity * self.cfg.account_risk_per_trade
        qty = dollar_risk / stop_distance

        # Take Profit at 2R target
        take_profit_price = entry_price + (2 * stop_distance) if entry_price > stop_price else entry_price - (2 * stop_distance)

        max_notional = equity * self.cfg.max_position_pct * self.cfg.leverage
        max_qty_by_notional = max_notional / entry_price
        qty = min(qty, max_qty_by_notional)

        if qty <= 0:
            return SizingResult(False, reason="Computed size is zero or negative")

        return SizingResult(True, qty=qty, take_profit_price=take_profit_price, reason="ok")

    def kill_switch_active(self, now: datetime) -> bool:
        self.reset_if_new_day(now)
        return self._kill_switch_active
