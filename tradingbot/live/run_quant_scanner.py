"""
Autonomous Institutional Quant Scanner & Auto-Trader Daemon

Features:
  1. Top 30 Market Volume Scanner (Dynamic Symbol Discovery)
  2. 3-Point Confluence Strategy (EMA Stack + RSI + Relative Volume RVOL)
  3. Top-Down BTC Market Regime Guard
  4. Dual TP Scaling (TP1 @ 1.5R with Auto-Breakeven SL, TP2 @ 3.0R runner)
  5. 3-Hour Stagnant Trade Auto-Exit (Time Decay Protection)
  6. Rich Telegram Signals with Smart Money RVOL metrics
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


class QuantScannerBot:
    def __init__(self):
        self.cfg = CONFIG
        self.logger = _setup_logging()
        self.exchange = BybitExchange(CONFIG.exchange)
        self.risk = RiskManager(CONFIG.risk)
        self.alerter = Alerter(CONFIG.alerts)
        
        self.top_symbols_limit = int(os.getenv("TOP_SYMBOLS_COUNT", "30"))
        self.max_hold_hours = float(os.getenv("MAX_HOLD_HOURS", "3.0"))
        self.max_active_trades = int(os.getenv("MAX_ACTIVE_TRADES", "3"))

        # Active trade tracking dictionary:
        self.active_scanner_trades = {}
        self.paper_balance = 10000.0

        mode = "DRY RUN (Paper Trading)" if CONFIG.dry_run else "LIVE TRADING"
        net  = "TESTNET" if CONFIG.exchange.testnet else "MAINNET"
        self.logger.warning(
            "Starting Quant Scanner Daemon -- mode=%s, net=%s, Top %d symbols, max_hold=%.1fh",
            mode, net, self.top_symbols_limit, self.max_hold_hours,
        )
        self.alerter.send(
            f"🏛️ Quant Scanner Daemon Active -- {mode} on {net} | Top {self.top_symbols_limit} Symbols | "
            f"Max Hold: {self.max_hold_hours}h | 15m Confluence Engine"
        )

    def write_scanner_state(self, top_symbols: list[str], scanned_data: list[dict]):
        state_dir = Path("state")
        state_dir.mkdir(exist_ok=True)
        state_file = state_dir / "scanner_state.json"
        
        state = {
            "last_update": datetime.now().isoformat(),
            "dry_run": CONFIG.dry_run,
            "testnet": CONFIG.exchange.testnet,
            "top_symbols": top_symbols,
            "account_balance": round(self.paper_balance, 2),
            "active_trades": {
                sym: {
                    "side": data["side"],
                    "entry_price": data["entry_price"],
                    "qty": data["qty"],
                    "entry_time": data["entry_time"].isoformat(),
                    "stop_price": data["stop_price"],
                    "tp1": data["tp1"],
                    "tp2": data["tp2"],
                    "tp1_hit": data.get("tp1_hit", False),
                }
                for sym, data in self.active_scanner_trades.items()
            },
            "scanned_data": scanned_data,
        }
        try:
            state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            self.logger.warning("Failed to write scanner_state.json: %s", e)

    def check_stagnant_timeouts(self, now: datetime):
        """Auto-close trades that stay stagnant longer than max_hold_hours."""
        for symbol, trade in list(self.active_scanner_trades.items()):
            entry_time = trade["entry_time"]
            age_hours = (now - entry_time).total_seconds() / 3600.0
            
            if age_hours >= self.max_hold_hours:
                # Fetch last price to compute PnL on timeout
                pnl = 0.0
                try:
                    df = self.exchange.get_klines(symbol, "15", limit=5)
                    if df is not None and len(df) > 0:
                        last_p = df.iloc[-1]["close"]
                        entry_p = trade["entry_price"]
                        qty = trade["qty"]
                        factor = 1.0 if trade["side"] == "Buy" else -1.0
                        pnl = (last_p - entry_p) * qty * factor
                        self.paper_balance += pnl
                except Exception:
                    pass

                pnl_str = f" | PnL: ${pnl:,.2f}" if pnl != 0.0 else ""
                msg = (
                    f"⏰ [SCANNER TIMEOUT] Closing stagnant trade {symbol} "
                    f"after {age_hours:.1f}h (Limit: {self.max_hold_hours}h){pnl_str}"
                )
                self.logger.info(msg)
                self.alerter.send(msg)

                # Close position on exchange if live
                if not CONFIG.dry_run:
                    close_side = "Sell" if trade["side"] == "Buy" else "Buy"
                    try:
                        self.exchange.place_market_order(symbol, close_side, trade["qty"], reduce_only=True)
                    except Exception as e:
                        self.logger.error("Failed to close stagnant position for %s: %s", symbol, e)

                del self.active_scanner_trades[symbol]

    def scan_once(self):
        now = datetime.now()
        
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
                    "symbol": symbol,
                    "price": round(last_price, 4),
                    "signal": out.signal.value,
                    "rvol": round(out.rvol, 2),
                    "rsi": round(out.rsi, 1),
                    "fast_ema": round(out.fast_ema, 2),
                    "slow_ema": round(out.slow_ema, 2),
                    "btc_regime": out.btc_regime,
                })

                # --- Handle New Entry Signal ---
                if curr_pos_dir == 0 and out.signal in (Signal.LONG, Signal.SHORT):
                    if len(self.active_scanner_trades) >= self.max_active_trades:
                        self.logger.info(
                            "[%s] Signal %s ignored: Max active trades (%d) reached",
                            symbol, out.signal.value, self.max_active_trades,
                        )
                        continue

                    equity = 10000.0 if CONFIG.dry_run else self.exchange.get_equity()
                    sizing = self.risk.size_position(equity, last_price, out.stop_price, open_positions=0, now=now)
                    
                    if not sizing.approved:
                        self.logger.info("[%s] Trade sizing rejected: %s", symbol, sizing.reason)
                        continue

                    def fmt_price(val: float | None) -> str:
                        if val is None:
                            return "—"
                        if val < 0.01:
                            return f"${val:.6f}"
                        if val < 1.0:
                            return f"${val:.4f}"
                        return f"${val:,.2f}"

                    side = "Buy" if out.signal == Signal.LONG else "Sell"
                    msg = (
                        f"🏛️ [QUANT SIGNAL ENTRY] {symbol} ({side}) 10X\n"
                        f"✦ Price: {fmt_price(last_price)}\n"
                        f"✦ Stop-Loss: {fmt_price(out.stop_price)}\n"
                        f"✦ TP1 (1.5R): {fmt_price(out.tp1_price)} (Auto-Breakeven)\n"
                        f"✦ TP2 (3.0R): {fmt_price(out.tp2_price)}\n"
                        f"✦ Smart Money RVOL: {out.rvol:.2f}x | BTC Regime: {out.btc_regime}"
                    )
                    self.logger.info(msg)
                    self.alerter.send(msg)

                    if not CONFIG.dry_run:
                        self.exchange.set_leverage(symbol, 5)
                        self.exchange.place_market_order(symbol, side, sizing.qty)
                        stop_side = "Sell" if side == "Buy" else "Buy"
                        self.exchange.place_stop_order(symbol, stop_side, sizing.qty, out.stop_price)

                    self.active_scanner_trades[symbol] = {
                        "side": side,
                        "entry_price": last_price,
                        "qty": sizing.qty,
                        "entry_time": now,
                        "stop_price": out.stop_price,
                        "tp1": out.tp1_price,
                        "tp2": out.tp2_price,
                        "tp1_hit": False,
                    }

                # --- Check TP1 / TP2 / SL Exits on Active Trades ---
                elif curr_pos_dir != 0:
                    trade = self.active_scanner_trades.get(symbol)
                    if trade:
                        entry_p = trade["entry_price"]
                        qty     = trade["qty"]
                        side    = trade["side"]
                        stop_p  = trade["stop_price"]
                        tp1_p   = trade["tp1"]
                        tp2_p   = trade["tp2"]
                        tp1_hit = trade.get("tp1_hit", False)

                        # Determine PnL factor
                        factor = 1.0 if side == "Buy" else -1.0
                        
                        # Stop-Loss Check
                        sl_hit = (last_price <= stop_p) if side == "Buy" else (last_price >= stop_p)
                        if sl_hit:
                            pnl = (last_price - entry_p) * qty * factor
                            self.paper_balance += pnl
                            msg = f"🛑 [QUANT STOP-LOSS] {symbol} hit SL @ ${last_price:,.4f} | PnL: ${pnl:,.2f}"
                            self.logger.info(msg)
                            self.alerter.send(msg)
                            if not CONFIG.dry_run:
                                close_side = "Sell" if side == "Buy" else "Buy"
                                self.exchange.place_market_order(symbol, close_side, qty, reduce_only=True)
                            del self.active_scanner_trades[symbol]
                            continue

                        # TP1 Check (50% scale-out & move SL to breakeven)
                        tp1_reached = (last_price >= tp1_p) if side == "Buy" else (last_price <= tp1_p)
                        if tp1_reached and not tp1_hit:
                            trade["tp1_hit"] = True
                            trade["stop_price"] = entry_p  # Move SL to Breakeven
                            half_qty = qty * 0.5
                            pnl = (tp1_p - entry_p) * half_qty * factor
                            self.paper_balance += pnl
                            msg = f"🎯 [QUANT TP1 HIT] {symbol} hit TP1 @ ${last_price:,.4f} | Locked PnL: +${pnl:,.2f} | SL moved to Breakeven (${entry_p:,.4f})"
                            self.logger.info(msg)
                            self.alerter.send(msg)

                        # TP2 Check (100% exit runner)
                        tp2_reached = (last_price >= tp2_p) if side == "Buy" else (last_price <= tp2_p)
                        if tp2_reached:
                            rem_qty = qty * 0.5 if tp1_hit else qty
                            pnl = (tp2_p - entry_p) * rem_qty * factor
                            self.paper_balance += pnl
                            msg = f"🚀 [QUANT TP2 HIT] {symbol} hit TP2 Runner @ ${last_price:,.4f} | Runner PnL: +${pnl:,.2f}"
                            self.logger.info(msg)
                            self.alerter.send(msg)
                            if not CONFIG.dry_run:
                                close_side = "Sell" if side == "Buy" else "Buy"
                                self.exchange.place_market_order(symbol, close_side, rem_qty, reduce_only=True)
                            del self.active_scanner_trades[symbol]
                            continue

                        # Signal Flat Exit
                        if out.signal == Signal.FLAT:
                            rem_qty = qty * 0.5 if tp1_hit else qty
                            pnl = (last_price - entry_p) * rem_qty * factor
                            self.paper_balance += pnl
                            msg = f"🏁 [QUANT SIGNAL EXIT] Closing {symbol} @ ~${last_price:,.4f} | PnL: ${pnl:,.2f}"
                            self.logger.info(msg)
                            self.alerter.send(msg)

                            if not CONFIG.dry_run:
                                close_side = "Sell" if side == "Buy" else "Buy"
                                self.exchange.place_market_order(symbol, close_side, rem_qty, reduce_only=True)

                            del self.active_scanner_trades[symbol]

            except Exception as e:
                self.logger.warning("Error scanning symbol %s: %s", symbol, e)

        # Write state for dashboard
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
