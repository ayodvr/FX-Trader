"""
Live trading loop — supports per-symbol CLI overrides for multi-bot portfolio.

Safety defaults:
  - DRY_RUN=true by default (config.py) — logs intended actions but places NO real orders.
  - Starts on testnet unless BYBIT_TESTNET=false is explicitly set.
  - Reconciles local position state against the exchange every iteration, rather than
    trusting its own memory — exchange is always the source of truth.

Single symbol:
    python -m live.run

Override symbol / EMA / risk for portfolio use:
    python -m live.run --symbol SOLUSDT --fast-ema 75 --slow-ema 200 --risk 0.005
    python -m live.run --symbol ETHUSDT --fast-ema 50 --slow-ema 200 --risk 0.005
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from config import CONFIG
from exchange.bybit_client import BybitExchange
from strategy.trend_strategy import generate_signal, Signal
from risk.risk_manager import RiskManager
from live.alerts import Alerter


def _apply_cli_overrides(args):
    """Patch CONFIG in-place from CLI args so every module sees the same object."""
    if args.symbol:
        CONFIG.strategy.symbol = args.symbol
    if args.fast_ema:
        CONFIG.strategy.fast_ema = args.fast_ema
    if args.slow_ema:
        CONFIG.strategy.slow_ema = args.slow_ema
    if args.risk:
        CONFIG.risk.account_risk_per_trade = args.risk


def _setup_logging(symbol: str) -> logging.Logger:
    """Log to both stdout and a per-symbol log file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{symbol}.log"

    logger = logging.getLogger(f"live.{symbol}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        logger.propagate = False

    return logger


def _write_state(symbol: str, state: dict):
    """Write bot state to state/<SYMBOL>.json for the dashboard to read."""
    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    state_file = state_dir / f"{symbol}.json"
    state["last_update"] = datetime.now().isoformat()
    try:
        state_file.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _append_trade(symbol: str, action: str, side: str, price: float, qty: float,
                  pnl: float | None = None, fee: float = 0.0):
    """Append a single trade record to trades/<SYMBOL>_trades.csv."""
    trades_dir = Path("trades")
    trades_dir.mkdir(exist_ok=True)
    trade_file = trades_dir / f"{symbol}_trades.csv"
    is_new = not trade_file.exists()
    try:
        with open(trade_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "action", "side", "price", "qty", "pnl", "fee"])
            if is_new:
                writer.writeheader()
            writer.writerow({
                "time":   datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "side":   side,
                "price":  round(price, 6),
                "qty":    round(qty, 8),
                "pnl":    round(pnl, 6) if pnl is not None else "",
                "fee":    round(fee, 6),
            })
    except Exception:
        pass


def _append_equity(symbol: str, equity: float):
    """Append an equity snapshot to <SYMBOL>_equity.csv for the dashboard chart."""
    equity_file = Path(f"{symbol}_equity.csv")
    is_new = not equity_file.exists()
    try:
        with open(equity_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "equity"])
            if is_new:
                writer.writeheader()
            writer.writerow({
                "time":   datetime.now().isoformat(timespec="seconds"),
                "equity": round(equity, 6),
            })
    except Exception:
        pass


class LiveBot:
    def __init__(self, logger: logging.Logger):
        self.cfg    = CONFIG
        self.logger = logger
        self.exchange = BybitExchange(CONFIG.exchange)
        self.risk     = RiskManager(CONFIG.risk)
        self.alerter  = Alerter(CONFIG.alerts)
        self.trail_price   = None
        self.stop_order_id = None   # tracks the live exchange stop order ID
        self.symbol        = CONFIG.strategy.symbol

        mode = "DRY RUN (no real orders)" if CONFIG.dry_run else "LIVE TRADING"
        net  = "TESTNET" if CONFIG.exchange.testnet else "MAINNET"
        self.logger.warning("Starting bot -- mode=%s, network=%s, symbol=%s, EMA(%d/%d)",
                            mode, net, self.symbol,
                            CONFIG.strategy.fast_ema, CONFIG.strategy.slow_ema)
        self.alerter.send(
            f"Bot starting -- {mode} on {net}, {self.symbol} "
            f"EMA({CONFIG.strategy.fast_ema}/{CONFIG.strategy.slow_ema})"
        )

        if not CONFIG.dry_run:
            self.exchange.set_leverage(self.symbol, CONFIG.risk.leverage)

    def get_current_position_direction(self) -> int:
        """Source of truth = exchange, not local memory."""
        pos = self.exchange.get_open_position(self.symbol)
        if pos is None:
            return 0
        side = pos.get("side")
        return 1 if side == "Buy" else (-1 if side == "Sell" else 0)

    def run_once(self):
        symbol = self.symbol
        try:
            df = self.exchange.get_klines(symbol, CONFIG.strategy.timeframe, limit=300)
            df_daily = self.exchange.get_klines(symbol, "D", limit=300)
        except Exception as e:
            self.logger.error("Failed to fetch klines: %s", e)
            self.alerter.send(f"[{symbol}] ERROR fetching klines: {e}")
            return

        current_position = self.get_current_position_direction()
        out   = generate_signal(df, CONFIG.strategy, current_position, self.trail_price, df_daily)
        price = df.iloc[-1]["close"]
        now   = datetime.now()

        # Update local trail and sync stop order on the exchange if it has moved
        if out.trail_price is not None and not (out.trail_price != out.trail_price):  # guard NaN
            new_trail = out.trail_price
            old_trail = self.trail_price
            self.trail_price = new_trail

            # Only replace the exchange stop if trail moved by ≥0.05% (avoid churn on tiny moves)
            if (old_trail is not None and current_position != 0
                    and abs(new_trail - old_trail) / max(abs(old_trail), 1e-9) >= 0.0005):
                stop_side = "Sell" if current_position == 1 else "Buy"
                self.logger.info(
                    "[%s] TRAIL STOP updated: %.2f -> %.2f — replacing exchange stop",
                    self.symbol, old_trail, new_trail,
                )
                if not CONFIG.dry_run:
                    # Cancel existing stop, place new one at updated trail level
                    if self.stop_order_id:
                        self.exchange.cancel_order(self.symbol, self.stop_order_id)
                    pos = self.exchange.get_open_position(self.symbol)
                    qty = float(pos["size"]) if pos else 0.0
                    if qty > 0:
                        resp = self.exchange.place_stop_order(
                            self.symbol, stop_side, qty, new_trail
                        )
                        self.stop_order_id = (
                            resp.get("result", {}).get("orderId") if resp else None
                        )

        # Fetch equity for state (best-effort)
        try:
            equity = self.exchange.get_equity()
            if CONFIG.dry_run and equity == 0.0:
                equity = 10000.0  # Fallback paper-trading balance
        except Exception:
            equity = 10000.0 if CONFIG.dry_run else None

        # Snapshot equity for the dashboard chart
        if equity is not None:
            _append_equity(symbol, equity)

        # Write state for dashboard
        _write_state(symbol, {
            "symbol":        symbol,
            "position":      current_position,
            "entry_price":   None,   # populated on entry below
            "equity":        equity,
            "signal":        out.signal.value,
            "fast_ema":      round(out.fast_ema, 2),
            "slow_ema":      round(out.slow_ema, 2),
            "atr":           round(out.atr, 2),
            "price":         round(price, 2),
            "dry_run":       CONFIG.dry_run,
            "testnet":       CONFIG.exchange.testnet,
            "ema_config":    f"EMA({CONFIG.strategy.fast_ema}/{CONFIG.strategy.slow_ema})",
        })

        self.logger.info(
            "pos=%d signal=%s price=%.2f fast_ema=%.2f slow_ema=%.2f atr=%.2f",
            current_position, out.signal.value, price, out.fast_ema, out.slow_ema, out.atr,
        )

        if self.risk.kill_switch_active(now=now):
            self.logger.warning("[%s] Kill switch active -- skipping this cycle", symbol)
            return

        # --- Entry ---
        if current_position == 0 and out.signal in (Signal.LONG, Signal.SHORT):
            if equity is None:
                equity = self.exchange.get_equity()
                if CONFIG.dry_run and equity == 0.0:
                    equity = 10000.0
            sizing = self.risk.size_position(equity, price, out.stop_price,
                                             open_positions=0, now=now)
            if not sizing.approved:
                self.logger.info("Trade not approved by risk manager: %s", sizing.reason)
                return

            side  = "Buy" if out.signal == Signal.LONG else "Sell"
            msg   = (f"[{symbol}] ENTER {side} qty={sizing.qty:.6f} "
                     f"@ ~{price:.2f}  stop={out.stop_price:.2f}")
            self.logger.info(msg)
            self.alerter.send(msg)

            fee = sizing.qty * price * 0.00055
            _append_trade(symbol, "ENTER", side.lower(), price, sizing.qty, pnl=None, fee=fee)

            if not CONFIG.dry_run:
                self.exchange.place_market_order(symbol, side, sizing.qty)
                stop_side = "Sell" if side == "Buy" else "Buy"
                try:
                    resp = self.exchange.place_stop_order(symbol, stop_side, sizing.qty, out.stop_price)
                    self.stop_order_id = resp.get("result", {}).get("orderId") if resp else None
                except Exception as e:
                    # Position is open with no protective stop -- fail safe by
                    # closing it immediately rather than riding unprotected.
                    self.logger.error("[%s] Failed to place protective stop after entry -- "
                                       "emergency closing position: %s", symbol, e)
                    self.alerter.send(f"🚨 [{symbol}] CRITICAL: stop-loss placement failed after entry "
                                       f"({e}) -- emergency closing position")
                    self.exchange.close_all_positions(symbol)
                    return

                # Place 50% Scale-Out Limit Order
                scale_qty = round(sizing.qty * 0.5, 6)
                tp_price = getattr(sizing, 'take_profit_price', 0)
                if scale_qty > 0 and tp_price > 0:
                    self.exchange.place_limit_order(symbol, stop_side, scale_qty, tp_price, reduce_only=True)

            self.trail_price = out.stop_price
            _write_state(symbol, {"symbol": symbol, "position": 1 if side == "Buy" else -1,
                                  "entry_price": price, "equity": equity,
                                  "signal": out.signal.value,
                                  "fast_ema": round(out.fast_ema, 2),
                                  "slow_ema": round(out.slow_ema, 2),
                                  "atr": round(out.atr, 2), "price": round(price, 2),
                                  "dry_run": CONFIG.dry_run,
                                  "testnet": CONFIG.exchange.testnet,
                                  "ema_config": f"EMA({CONFIG.strategy.fast_ema}/{CONFIG.strategy.slow_ema})"})

        # --- Exit ---
        elif current_position != 0 and out.signal == Signal.FLAT:
            pos         = self.exchange.get_open_position(symbol)
            qty         = float(pos["size"])     if pos else 0.0
            entry_price = float(pos["avgPrice"]) if pos else price
            side        = "Sell" if current_position == 1 else "Buy"
            msg         = f"[{symbol}] EXIT qty={qty:.6f} @ ~{price:.2f}"
            self.logger.info(msg)
            self.alerter.send(msg)

            if not CONFIG.dry_run and qty > 0:
                # Cancel any open stop order first to prevent a race with our market close
                if self.stop_order_id:
                    self.exchange.cancel_order(symbol, self.stop_order_id)
                else:
                    self.exchange.cancel_all_stops(symbol)  # belt-and-suspenders
                self.exchange.place_market_order(symbol, side, qty, reduce_only=True)

            realized_pnl = (price - entry_price) * qty * current_position
            fee = qty * price * 0.00055
            _append_trade(symbol, "EXIT", side.lower(), price, qty, pnl=realized_pnl - fee, fee=fee)

            if equity is None:
                equity = self.exchange.get_equity()
                if CONFIG.dry_run and equity == 0.0:
                    equity = 10000.0
            self.risk.record_realized_pnl(realized_pnl, equity, now=now)
            self.trail_price    = None
            self.stop_order_id  = None

    def run_forever(self):
        interval_min = int(CONFIG.strategy.timeframe)
        while True:
            try:
                self.run_once()
            except Exception as e:
                self.logger.exception("Unhandled error in run_once")
                self.alerter.send(f"[{self.symbol}] UNHANDLED ERROR: {e}")
            self.exchange.wait_for_next_candle_close(interval_min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading bot")
    parser.add_argument("--symbol",   type=str,   help="Symbol override, e.g. SOLUSDT")
    parser.add_argument("--fast-ema", type=int,   dest="fast_ema", help="Fast EMA period")
    parser.add_argument("--slow-ema", type=int,   dest="slow_ema", help="Slow EMA period")
    parser.add_argument("--risk",     type=float, help="Account risk per trade, e.g. 0.005")
    args = parser.parse_args()

    _apply_cli_overrides(args)
    symbol = CONFIG.strategy.symbol
    logger = _setup_logging(symbol)
    bot    = LiveBot(logger)
    bot.run_forever()
