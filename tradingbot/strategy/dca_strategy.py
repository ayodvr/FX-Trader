"""
DCA / martingale grid ladder — the strategy family behind most retail "AI" bots.

Mechanically this is not a forecast. There is no entry signal to speak of: you
open a position, and if it moves against you, you buy more at a worse price so
the average entry moves toward the market. You close the whole stack as soon as
price crosses that average by a small margin.

Because a deal only ever closes in profit, the win rate is enormous -- typically
90-98%. That is the entire reason these bots look so good on a short chart: a
staircase of small green closes. What the win rate does not show is where the
losses live, which is in the tail:

  - Each safety order is usually LARGER than the last (volume_scale > 1), so
    exposure grows fastest exactly when the trade is going worst.
  - There is usually no stop loss, by design. That is what keeps the win rate
    high, and it is also what converts a bad trend into an open-ended loss.
  - Under leverage, the ladder can exhaust its margin before price ever
    retraces to the average entry, which liquidates the position at the worst
    possible point.

So the meaningful question about a DCA bot is never "what is its win rate" but
"what happens on the worst run it has not met yet". This module only builds the
ladder; backtest/run_dca.py models the margin and liquidation that decide it.
"""
from dataclasses import dataclass


@dataclass
class DcaConfig:
    direction: str = "long"           # "long", "short", or "both"
    base_order_pct: float = 0.02      # base order notional, as a fraction of equity
    safety_order_pct: float = 0.02    # first safety order notional, same units
    max_safety_orders: int = 5
    price_deviation_pct: float = 0.01  # distance to the first safety order
    step_scale: float = 1.5           # each step sits this much further than the last
    volume_scale: float = 1.5         # each safety order is this much bigger
    take_profit_pct: float = 0.01     # target, measured from the AVERAGE entry
    stop_loss_pct: float = 0.0        # from average entry; 0 disables (the usual default)
    leverage: int = 10


@dataclass
class LadderStep:
    index: int            # 0 = base order
    deviation: float      # fractional distance from the base entry
    price: float
    notional: float


def build_ladder(entry_price: float, equity: float, cfg: DcaConfig,
                 is_long: bool = True) -> list[LadderStep]:
    """
    Full order ladder for one deal: the base order plus every safety order.

    Deviations compound by step_scale and sizes by volume_scale, which is what
    makes the ladder's exposure grow super-linearly as price runs away:

        deviation_n = price_deviation * (1 + step_scale + step_scale^2 + ...)
        notional_n  = safety_order   *  volume_scale^(n-1)
    """
    steps = [LadderStep(0, 0.0, entry_price, equity * cfg.base_order_pct)]

    cumulative_dev = 0.0
    step_dev = cfg.price_deviation_pct
    notional = equity * cfg.safety_order_pct

    for n in range(1, cfg.max_safety_orders + 1):
        cumulative_dev += step_dev
        price = (entry_price * (1 - cumulative_dev) if is_long
                 else entry_price * (1 + cumulative_dev))
        steps.append(LadderStep(n, cumulative_dev, price, notional))
        step_dev *= cfg.step_scale
        notional *= cfg.volume_scale

    return steps


def ladder_totals(steps: list[LadderStep]) -> tuple[float, float]:
    """(total notional if every step fills, deviation at the final step)."""
    return sum(s.notional for s in steps), steps[-1].deviation if steps else 0.0


def average_entry(filled: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Volume-weighted average entry over (price, qty) fills.
    Returns (avg_price, total_qty).
    """
    qty = sum(q for _, q in filled)
    if qty <= 0:
        return 0.0, 0.0
    return sum(p * q for p, q in filled) / qty, qty


def take_profit_price(avg_entry: float, cfg: DcaConfig, is_long: bool) -> float:
    return (avg_entry * (1 + cfg.take_profit_pct) if is_long
            else avg_entry * (1 - cfg.take_profit_pct))


def stop_loss_price(avg_entry: float, cfg: DcaConfig, is_long: bool) -> float | None:
    if cfg.stop_loss_pct <= 0:
        return None
    return (avg_entry * (1 - cfg.stop_loss_pct) if is_long
            else avg_entry * (1 + cfg.stop_loss_pct))


def liquidation_price(avg_entry: float, leverage: int, is_long: bool,
                      maintenance_margin: float = 0.005) -> float:
    """
    Approximate isolated-margin liquidation price.

    A position posts 1/leverage of notional as margin and is liquidated once
    unrealised loss eats it, less the maintenance requirement. At 10x that is
    roughly a 9.5% adverse move from the AVERAGE entry -- and because averaging
    down keeps pulling the average toward price, each safety order also drags
    the liquidation price closer to the market.
    """
    edge = (1.0 / leverage) - maintenance_margin
    return avg_entry * (1 - edge) if is_long else avg_entry * (1 + edge)
