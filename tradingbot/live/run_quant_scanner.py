"""
Autonomous Institutional Quant Scanner & Auto-Trader Daemon

Features:
  1. Top 30 Market Volume Scanner (Dynamic Symbol Discovery)
  2. 3-Point Confluence Strategy (EMA Stack + RSI + Relative Volume RVOL)
  3. Top-Down BTC Market Regime Guard
  4. Dual TP Scaling (TP1 @ 1.5R with Auto-Breakeven SL, TP2 @ 3.0R runner)
  5. 3-Hour Stagnant Trade Auto-Exit (Time Decay Protection)
  6. Rich Telegram Signals with Smart Money RVOL metrics
  7. State persistence across restarts (active trades survive a process restart)
  8. Native exchange Stop-Market orders in live mode
  9. Daily kill switch wired to every exit path
"""
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path so modules import reliably regardless of CWD
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import CONFIG
from exchange.bybit_client import BybitExchange
from strategy.quant_strategy import generate_quant_signal, Signal
from risk.risk_manager import RiskManager
from live.alerts import Alerter

logger = logging.getLogger("quant_scanner")

# File that persists active trade state across restarts
_STATE_FILE = Path("state") / "scanner_state.json"


def _setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "quant_scanner.log"

    lg = logging.getLogger("quant_scanner")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        lg.addHandler(sh)

        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(formatter)
        lg.addHandler(fh)

        lg.propagate = False
    return lg


def fmt_price(val: float | None) -> str:
    """Format a price for human-readable display, handling sub-cent altcoins correctly."""
    if val is None:
        return "—"
    if val < 0.001:
        return f"${val:.7f}"
    if val < 0.01:
        return f"${val:.6f}"
    if val < 1.0:
        return f"${val:.4f}"
    return f"${val:,.2f}"


class QuantScannerBot:
    def __init__(self):
        self.cfg = CONFIG
        self.logger = _setup_logging()
        self.exchange = BybitExchange(CONFIG.exchange)
        self.risk = RiskManager(CONFIG.risk)
        self.alerter = Alerter(CONFIG.alerts)

        # Read scanner settings from typed config (previously inline os.getenv)
        self.top_symbols_limit = CONFIG.scanner.top_symbols_count
        self.max_hold_hours    = CONFIG.scanner.max_hold_hours
        self.max_active_trades = CONFIG.scanner.max_active_trades
        self.leverage          = CONFIG.risk.leverage

        # Active trade tracking — re-hydrated from disk on startup
        self.active_scanner_trades: dict = {}
        self.paper_balance = self._load_paper_balance()

        # Per-trade exchange stop-order IDs for live mode (symbol -> orderId)
        self._stop_order_ids: dict[str, str] = {}

        # Restore any open trades from the previous session
        self._load_scanner_state()

        mode = "DRY RUN (Paper Trading)" if CONFIG.dry_run else "LIVE TRADING"
        net  = "TESTNET" if CONFIG.exchange.testnet else "MAINNET"
        self.logger.warning(
            "Starting Quant Scanner Daemon -- mode=%s, net=%s, Top %d symbols, max_hold=%.1fh, leverage=%dX",
            mode, net, self.top_symbols_limit, self.max_hold_hours, self.leverage,
        )
        self.alerter.send(
            f"🏛️ Quant Scanner Daemon Active -- {mode} on {net} | Top {self.top_symbols_limit} Symbols | "
            f"Max Hold: {self.max_hold_hours}h | {self.leverage}X Leverage | 15m Confluence Engine\n"
            f"💰 Starting Balance: {fmt_price(self.paper_balance)}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_paper_balance(self) -> float:
        """Restore paper balance from previous session if available."""
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text())
                bal = data.get("account_balance")
                if bal and bal > 0:
                    self.logger.info("Restored paper balance from state: $%.2f", bal)
                    return float(bal)
        except Exception as e:
            self.logger.warning("Could not load paper balance from state: %s", e)
        return 10_000.0

    def _load_scanner_state(self):
        """
        Re-hydrate active_scanner_trades from the persisted scanner_state.json so
        trades are not lost across process restarts. The entry_time is restored as
        the original datetime so timeout logic remains accurate.
        """
        try:
            if not _STATE_FILE.exists():
                return
            data = json.loads(_STATE_FILE.read_text())
            raw_trades = data.get("active_trades", {})
            if not raw_trades:
                return
            restored = 0
            for sym, td in raw_trades.items():
                try:
                    self.active_scanner_trades[sym] = {
                        "side":        td["side"],
                        "entry_price": float(td["entry_price"]),
                        "qty":         float(td["qty"]),
                        "entry_time":  datetime.fromisoformat(td["entry_time"]),
                        "stop_price":  float(td["stop_price"]),
                        "tp1":         float(td["tp1"]) if td.get("tp1") is not None else None,
                        "tp2":         float(td["tp2"]) if td.get("tp2") is not None else None,
                        "tp1_hit":     bool(td.get("tp1_hit", False)),
                    }
                    restored += 1
                except (KeyError, TypeError, ValueError) as e:
                    self.logger.warning("Skipping malformed trade for %s during state reload: %s", sym, e)
            if restored:
                self.logger.warning(
                    "Restored %d active trade(s) from previous session: %s",
                    restored, list(self.active_scanner_trades.keys())
                )
                self.alerter.send(
                    f"♻️ [RESTART] Restored {restored} active trade(s) from previous session: "
                    + ", ".join(self.active_scanner_trades.keys())
                )
        except Exception as e:
            self.logger.warning("Failed to load scanner state: %s", e)

    def write_scanner_state(self, top_symbols: list[str], scanned_data: list[dict]):
        state_dir = Path("state")
        state_dir.mkdir(exist_ok=True)

        state = {
            "last_update":    datetime.now().isoformat(),
            "dry_run":        CONFIG.dry_run,
            "testnet":        CONFIG.exchange.testnet,
            "top_symbols":    top_symbols,
            "account_balance": round(self.paper_balance, 2),
            "active_trades": {
                sym: {
                    "side":        data["side"],
                    "entry_price": data["entry_price"],
                    "qty":         data["qty"],
                    "entry_time":  data["entry_time"].isoformat(),
                    "stop_price":  data["stop_price"],
                    "tp1":         data["tp1"],
                    "tp2":         data["tp2"],
                    "tp1_hit":     data.get("tp1_hit", False),
                }
                for sym, data in self.active_scanner_trades.items()
            },
            "scanned_data": scanned_data,
        }
        try:
            _STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            self.logger.warning("Failed to write scanner_state.json: %s", e)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_equity(self) -> float:
        """Return current equity — paper balance in dry-run, live account in live mode."""
        if CONFIG.dry_run:
            return self.paper_balance
        equity = self.exchange.get_equity()
        return equity if equity > 0 else self.paper_balance

    def _record_pnl(self, pnl: float):
        """
        Update paper balance and notify the risk manager of every realised PnL.
        This ensures the daily kill switch fires correctly.
        Also records the trade in the alerter for the daily summary.
        """
        self.paper_balance += pnl
        equity = self._current_equity()
        self.risk.record_realized_pnl(pnl, equity, now=datetime.now())
        self.alerter.record_trade(pnl)

    def _place_exchange_stop(self, symbol: str, side: str, qty: float, stop_price: float) -> bool:
        """Place a Stop-Market order on the exchange and cache its order ID.

        Returns True only if the exchange accepted the stop -- callers use this
        to decide whether the position is actually protected.
        """
        try:
            resp = self.exchange.place_stop_order(symbol, side, qty, stop_price)
            order_id = resp.get("result", {}).get("orderId") if resp else None
            if order_id:
                self._stop_order_ids[symbol] = order_id
                self.logger.info("[%s] Exchange SL placed @ %s (orderId=%s)", symbol, fmt_price(stop_price), order_id)
                return True
            self.logger.error("[%s] Stop order response had no orderId: %s", symbol, resp)
            return False
        except Exception as e:
            self.logger.error("[%s] Failed to place exchange stop order: %s", symbol, e)
            return False

    def _cancel_exchange_stop(self, symbol: str):
        """Cancel the cached exchange stop order for a symbol."""
        order_id = self._stop_order_ids.pop(symbol, None)
        if order_id:
            self.exchange.cancel_order(symbol, order_id)
        else:
            # Belt-and-suspenders: cancel everything
            self.exchange.cancel_all_stops(symbol)

    def _close_trade(self, symbol: str, trade: dict, pnl: float, reason: str, alert_msg: str):
        """
        Unified trade-close helper. Records PnL, sends alert, and removes the
        trade from active_scanner_trades.
        """
        self._record_pnl(pnl)
        self.logger.info(alert_msg)
        self.alerter.send(alert_msg)

        if self.risk.kill_switch_active(now=datetime.now()):
            self.alerter.send(
                "🚨 [KILL SWITCH] Daily loss limit reached — no new trades until tomorrow."
            )

        if not CONFIG.dry_run:
            close_side = "Sell" if trade["side"] == "Buy" else "Buy"
            # If TP1 already scaled out 50% on the exchange, only the remaining
            # half is still open — closing the original full qty would send a
            # reduce-only order larger than the actual position.
            remaining_qty = trade["qty"] * 0.5 if trade.get("tp1_hit", False) else trade["qty"]
            try:
                self._cancel_exchange_stop(symbol)
                self.exchange.place_market_order(symbol, close_side, remaining_qty, reduce_only=True)
            except Exception as e:
                self.logger.error("Failed to close position for %s: %s", symbol, e)

        del self.active_scanner_trades[symbol]

    # ──────────────────────────────────────────────────────────────────────────
    # Timeout check
    # ──────────────────────────────────────────────────────────────────────────

    def check_stagnant_timeouts(self, now: datetime):
        """Auto-close trades that stay stagnant longer than max_hold_hours."""
        for symbol, trade in list(self.active_scanner_trades.items()):
            entry_time = trade["entry_time"]
            age_hours = (now - entry_time).total_seconds() / 3600.0

            if age_hours >= self.max_hold_hours:
                pnl = 0.0
                try:
                    df = self.exchange.get_klines(symbol, "15", limit=5)
                    if df is not None and len(df) > 0:
                        last_p  = df.iloc[-1]["close"]
                        entry_p = trade["entry_price"]
                        qty     = trade["qty"]
                        factor  = 1.0 if trade["side"] == "Buy" else -1.0
                        tp1_hit = trade.get("tp1_hit", False)
                        rem_qty = qty * 0.5 if tp1_hit else qty
                        pnl = (last_p - entry_p) * rem_qty * factor
                except Exception:
                    pass

                msg = (
                    f"⏰ [SCANNER TIMEOUT] Closing stagnant trade {symbol} "
                    f"after {age_hours:.1f}h (Limit: {self.max_hold_hours}h) | PnL: ${pnl:,.2f}"
                )
                self._close_trade(symbol, trade, pnl, reason="timeout", alert_msg=msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Main scan loop
    # ──────────────────────────────────────────────────────────────────────────

    def scan_once(self):
        now = datetime.now()

        # Kill switch check — skip everything if daily limit is blown
        if self.risk.kill_switch_active(now=now):
            self.logger.warning("Kill switch active — skipping scan cycle")
            return

        # Send daily summary at the turn of the UTC day (first scan after midnight)
        if not hasattr(self, "_last_summary_date"):
            self._last_summary_date = now.date()
        if now.date() > self._last_summary_date:
            self.alerter.send_daily_summary()
            self._last_summary_date = now.date()

        # 1. Fetch Top N Volume Symbols from Bybit
        top_symbols = self.exchange.get_top_symbols(limit=self.top_symbols_limit)

        # 2. Fetch BTC 1h data for Market Regime Guard
        try:
            df_btc = self.exchange.get_klines("BTCUSDT", "60", limit=100)
        except Exception:
            df_btc = None

        # 3. Check timeouts on active trades
        self.check_stagnant_timeouts(now)

        scanned_data = []

        # 4. Scan each top symbol
        for symbol in top_symbols:
            try:
                df = self.exchange.get_klines(symbol, "15", limit=150)
                if df is None or len(df) < 50:
                    continue

                curr_pos_dir = 0
                if symbol in self.active_scanner_trades:
                    curr_pos_dir = 1 if self.active_scanner_trades[symbol]["side"] == "Buy" else -1

                out = generate_quant_signal(df, CONFIG.strategy, current_position=curr_pos_dir, df_btc_1h=df_btc)
                last_price = df.iloc[-1]["close"]

                scanned_data.append({
                    "symbol":     symbol,
                    "price":      round(last_price, 6),
                    "signal":     out.signal.value,
                    "rvol":       round(out.rvol, 2),
                    "rsi":        round(out.rsi, 1),
                    "fast_ema":   round(out.fast_ema, 4),
                    "slow_ema":   round(out.slow_ema, 4),
                    "btc_regime": out.btc_regime,
                })

                # ── Handle New Entry Signal ───────────────────────────────────
                if curr_pos_dir == 0 and out.signal in (Signal.LONG, Signal.SHORT):
                    if len(self.active_scanner_trades) >= self.max_active_trades:
                        self.logger.info(
                            "[%s] Signal %s ignored: Max active trades (%d) reached",
                            symbol, out.signal.value, self.max_active_trades,
                        )
                        continue

                    equity  = self._current_equity()
                    sizing  = self.risk.size_position(
                        equity, last_price, out.stop_price, open_positions=0, now=now
                    )

                    if not sizing.approved:
                        self.logger.info("[%s] Trade sizing rejected: %s", symbol, sizing.reason)
                        continue

                    # Skip symbols where our size rounds below the exchange minimum --
                    # the order would just be rejected, and in live mode a rejected
                    # entry mid-sequence is worse than never starting one.
                    if not CONFIG.dry_run and not self.exchange.meets_min_qty(symbol, sizing.qty, last_price):
                        self.logger.info("[%s] Skipping: qty %.8f fails exchange minimum size/value",
                                         symbol, sizing.qty)
                        continue

                    side     = "Buy" if out.signal == Signal.LONG else "Sell"
                    stop_side = "Sell" if side == "Buy" else "Buy"
                    msg = (
                        f"🏛️ [QUANT SIGNAL ENTRY] {symbol} ({side}) {self.leverage}X\n"
                        f"✦ Price: {fmt_price(last_price)}\n"
                        f"✦ Stop-Loss: {fmt_price(out.stop_price)}\n"
                        f"✦ TP1 (1.5R): {fmt_price(out.tp1_price)} (Auto-Breakeven)\n"
                        f"✦ TP2 (3.0R): {fmt_price(out.tp2_price)}\n"
                        f"✦ Smart Money RVOL: {out.rvol:.2f}x | BTC Regime: {out.btc_regime}"
                    )
                    self.logger.info(msg)
                    self.alerter.send(msg)

                    if not CONFIG.dry_run:
                        self.exchange.set_leverage(symbol, self.leverage)
                        self.exchange.place_market_order(symbol, side, sizing.qty)
                        if not self._place_exchange_stop(symbol, stop_side, sizing.qty, out.stop_price):
                            # Position is open with no protective stop -- close it
                            # immediately rather than tracking an unprotected trade.
                            self.logger.error("[%s] Stop placement failed after entry -- emergency closing", symbol)
                            self.alerter.send(
                                f"🚨 [{symbol}] CRITICAL: stop-loss placement failed after entry "
                                f"-- emergency closing position"
                            )
                            self.exchange.close_all_positions(symbol)
                            continue

                    self.active_scanner_trades[symbol] = {
                        "side":        side,
                        "entry_price": last_price,
                        "qty":         sizing.qty,
                        "entry_time":  now,
                        "stop_price":  out.stop_price,
                        "tp1":         out.tp1_price,
                        "tp2":         out.tp2_price,
                        "tp1_hit":     False,
                    }

                # ── Check TP1 / TP2 / SL Exits on Active Trades ─────────────
                elif curr_pos_dir != 0:
                    trade   = self.active_scanner_trades.get(symbol)
                    if not trade:
                        continue

                    entry_p = trade["entry_price"]
                    qty     = trade["qty"]
                    side    = trade["side"]
                    stop_p  = trade["stop_price"]
                    tp1_p   = trade["tp1"]
                    tp2_p   = trade["tp2"]
                    tp1_hit = trade.get("tp1_hit", False)
                    factor  = 1.0 if side == "Buy" else -1.0

                    # ── Stop-Loss check (candle close; exchange SL handles intra-candle) ──
                    sl_hit = (last_price <= stop_p) if side == "Buy" else (last_price >= stop_p)
                    if sl_hit:
                        pnl = (last_price - entry_p) * qty * factor
                        msg = (
                            f"🛑 [QUANT STOP-LOSS] {symbol} hit SL @ {fmt_price(last_price)} "
                            f"| PnL: ${pnl:,.2f}"
                        )
                        self._close_trade(symbol, trade, pnl, reason="sl", alert_msg=msg)
                        continue

                    # ── TP1 check (50% scale-out & move SL to breakeven) ──────
                    if tp1_p is not None:
                        tp1_reached = (last_price >= tp1_p) if side == "Buy" else (last_price <= tp1_p)
                        if tp1_reached and not tp1_hit:
                            trade["tp1_hit"]    = True
                            trade["stop_price"] = entry_p  # Move SL to breakeven
                            half_qty = qty * 0.5
                            pnl = (tp1_p - entry_p) * half_qty * factor
                            self._record_pnl(pnl)
                            msg = (
                                f"🎯 [QUANT TP1 HIT] {symbol} hit TP1 @ {fmt_price(last_price)} "
                                f"| Locked PnL: +${pnl:,.2f} "
                                f"| SL moved to Breakeven ({fmt_price(entry_p)})"
                            )
                            self.logger.info(msg)
                            self.alerter.send(msg)

                            # Actually scale out 50% on the exchange, then move the
                            # remaining stop to breakeven for the reduced size.
                            if not CONFIG.dry_run:
                                close_side = "Sell" if side == "Buy" else "Buy"
                                self._cancel_exchange_stop(symbol)
                                try:
                                    self.exchange.place_market_order(symbol, close_side, half_qty, reduce_only=True)
                                except Exception as e:
                                    self.logger.error("[%s] Failed to execute TP1 scale-out order: %s", symbol, e)
                                    self.alerter.send(f"🚨 [{symbol}] CRITICAL: TP1 scale-out order failed ({e}) — position may be oversized")
                                self._place_exchange_stop(symbol, close_side, half_qty, entry_p)

                    # ── TP2 check (100% exit runner) ─────────────────────────
                    if tp2_p is not None:
                        tp2_reached = (last_price >= tp2_p) if side == "Buy" else (last_price <= tp2_p)
                        if tp2_reached:
                            rem_qty = qty * 0.5 if tp1_hit else qty
                            pnl = (tp2_p - entry_p) * rem_qty * factor
                            msg = (
                                f"🚀 [QUANT TP2 HIT] {symbol} hit TP2 Runner @ {fmt_price(last_price)} "
                                f"| Runner PnL: +${pnl:,.2f}"
                            )
                            self._close_trade(symbol, trade, pnl, reason="tp2", alert_msg=msg)
                            continue

                    # ── Signal Flat Exit ──────────────────────────────────────
                    if out.signal == Signal.FLAT:
                        rem_qty = qty * 0.5 if tp1_hit else qty
                        pnl = (last_price - entry_p) * rem_qty * factor
                        msg = (
                            f"🏁 [QUANT SIGNAL EXIT] Closing {symbol} @ {fmt_price(last_price)} "
                            f"| PnL: ${pnl:,.2f}"
                        )
                        self._close_trade(symbol, trade, pnl, reason="signal_flat", alert_msg=msg)

            except Exception as e:
                self.logger.warning("Error scanning symbol %s: %s", symbol, e)

        # Write state for dashboard and persistence
        self.write_scanner_state(top_symbols, scanned_data)

    def run_forever(self):
        while True:
            try:
                self.scan_once()
            except Exception as e:
                self.logger.exception("Unhandled error in scan_once")
                self.alerter.send(f"UNHANDLED SCANNER ERROR: {e}")
            time.sleep(CONFIG.poll_interval_sec)


if __name__ == "__main__":
    bot = QuantScannerBot()
    bot.run_forever()
