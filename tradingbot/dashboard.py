"""
Dashboard server — multi-symbol trading bot web UI.

Run with:
    python dashboard.py

Then open http://localhost:8080
"""
import json
import os
import subprocess
import sys
import signal
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file variables into os.environ
load_dotenv()

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import logging
from config import CONFIG
from exchange.bybit_client import BybitExchange
from risk.risk_manager import RiskManager
from strategy.trend_strategy import compute_indicators

from flask import Flask, jsonify, render_template_string, request, Response

app = Flask(__name__)

def check_auth(username, password):
    expected_user = os.getenv("DASHBOARD_USERNAME", "admin")
    expected_pass = os.getenv("DASHBOARD_PASSWORD", "admin123")
    return username == expected_user and password == expected_pass

def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

@app.before_request
def require_auth():
    if request.path in ['/manifest.json', '/sw.js', '/icon.svg']:
        return
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()

BASE_DIR   = Path(__file__).parent
STATE_DIR  = BASE_DIR / "state"
LOG_DIR    = BASE_DIR / "logs"
TRADES_DIR = BASE_DIR / "trades"

# Fallback CSV paths (backtest output)
BACKTEST_TRADES = BASE_DIR / "backtest_trades.csv"
BACKTEST_EQUITY = BASE_DIR / "backtest_equity_curve.csv"

PORTFOLIO_SYMBOLS = ["SOLUSDT", "ETHUSDT", "BTCUSDT"]


# ── PWA Endpoints ──────────────────────────────────────────────────────────────

@app.route('/manifest.json')
def pwa_manifest():
    return jsonify({
        "name": "Cybrox Ultra Trader",
        "short_name": "CBX-Trader",
        "description": "Algorithmic Trading Bot Dashboard",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a0b2e",
        "theme_color": "#1a0b2e",
        "icons": [
            {
                "src": "/icon.svg",
                "sizes": "192x192",
                "type": "image/svg+xml",
                "purpose": "any maskable"
            },
            {
                "src": "/icon.svg",
                "sizes": "512x512",
                "type": "image/svg+xml",
                "purpose": "any maskable"
            },
            {
                "src": "/icon.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any maskable"
            }
        ]
    })

@app.route('/sw.js')
def pwa_sw():
    js = """
    self.addEventListener('install', (e) => {
      self.skipWaiting();
    });
    self.addEventListener('fetch', (e) => {
      // Basic pass-through for PWA install requirements
    });
    """
    return Response(js, mimetype='application/javascript')

@app.route('/icon.svg')
def pwa_icon():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
    <rect width="512" height="512" rx="100" fill="#1a0b2e"/>
    <path d="M100 400 L200 250 L300 300 L420 150" stroke="#00ffa3" stroke-width="40" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="420" cy="150" r="30" fill="#00ffa3"/>
</svg>'''
    return Response(svg, mimetype='image/svg+xml')


# ── Helpers ────────────────────────────────────────────────────────────────────

def bot_is_running(symbol: str) -> bool:
    """Cross-platform check using psutil when available."""
    pid_file = STATE_DIR / f"{symbol}.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return False

    if _HAS_PSUTIL:
        alive = psutil.pid_exists(pid)
    else:
        try:
            os.kill(pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False

    if not alive:
        pid_file.unlink(missing_ok=True)
    return alive


def read_state(symbol: str) -> dict:
    sf = STATE_DIR / f"{symbol}.json"
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {}


def read_csv_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        import csv
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def tail_log(symbol: str, n: int = 150) -> list[str]:
    log_file = LOG_DIR / f"{symbol}.log"
    if not log_file.exists():
        # Fall back to generic bot.log
        log_file = BASE_DIR / "bot.log"
    if not log_file.exists():
        return []
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def read_config() -> dict:
    try:
        sys.path.insert(0, str(BASE_DIR))
        import importlib, config as cfg_mod
        importlib.reload(cfg_mod)
        s = cfg_mod.CONFIG.strategy
        r = cfg_mod.CONFIG.risk
        return {
            "symbol": s.symbol, "timeframe": s.timeframe,
            "fast_ema": s.fast_ema, "slow_ema": s.slow_ema,
            "min_adx": s.min_adx, "long_only": s.long_only,
            "atr_stop_mult": s.atr_stop_mult, "atr_trail_mult": s.atr_trail_mult,
            "account_risk_per_trade": r.account_risk_per_trade,
            "max_daily_loss_pct": r.max_daily_loss_pct, "leverage": r.leverage,
        }
    except Exception as e:
        return {"error": str(e)}


# ── API ────────────────────────────────────────────────────────────────────────

def calculate_analytics(sym):
    rows = read_csv_safe(TRADES_DIR / f"{sym}_trades.csv")
    if not rows:
        return {"win_rate": "0.0", "profit_factor": "0.00", "drawdown": "0.0"}
    
    wins, losses = 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    
    for r in rows:
        if r.get("action") == "EXIT" and r.get("pnl"):
            try:
                p = float(r["pnl"])
                if p > 0:
                    wins += 1
                    gross_profit += p
                elif p < 0:
                    losses += 1
                    gross_loss += abs(p)
            except:
                pass
                
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    
    # Calculate drawdown from equity curve
    eq_rows = read_csv_safe(BASE_DIR / f"{sym}_equity.csv")
    dd = 0.0
    if eq_rows:
        peak = 0.0
        for e in eq_rows:
            try:
                val = float(e.get("equity", 0))
                if val > peak:
                    peak = val
                elif peak > 0:
                    cur_dd = (peak - val) / peak * 100
                    if cur_dd > dd:
                        dd = cur_dd
            except:
                pass
                
    return {
        "win_rate": f"{win_rate:.1f}",
        "profit_factor": f"{pf:.2f}",
        "drawdown": f"{dd:.1f}"
    }

def read_scanner_state() -> dict:
    sf = STATE_DIR / "scanner_state.json"
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {}


@app.route("/api/portfolio")
def api_portfolio():
    result = []
    for sym in PORTFOLIO_SYMBOLS:
        state = read_state(sym)
        analytics = calculate_analytics(sym)
        result.append({
            "symbol":  sym,
            "running": bot_is_running(sym),
            **state,
            **analytics
        })
    return jsonify(result)


@app.route("/api/scanner/state")
def api_scanner_state():
    return jsonify(read_scanner_state())


@app.route("/api/scanner/tail_log")
def api_scanner_tail_log():
    log_file = LOG_DIR / "quant_scanner.log"
    if not log_file.exists():
        return jsonify({"lines": []})
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        return jsonify({"lines": lines[-150:]})
    except Exception as e:
        return jsonify({"lines": [f"Error reading log: {e}"]})


@app.route("/api/webhook/tradingview", methods=["POST"])
def api_webhook_tv():
    try:
        data = request.json
        if not data or data.get("secret") != os.getenv("TV_WEBHOOK_SECRET", "default_secret"):
            return jsonify({"ok": False, "msg": "Unauthorized"}), 401
            
        sym = data.get("symbol")
        action = data.get("action")
        if not sym or not action:
            return jsonify({"ok": False, "msg": "Missing symbol or action"}), 400
            
        def log_to_file(msg):
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                with open(LOG_DIR / f"{sym}.log", "a") as f:
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    f.write(f"{ts} [INFO] webhook.{sym}: {msg}\n")
            except Exception:
                pass
            logging.info(f"Webhook {sym}: {msg}")
            
        log_to_file(f"TradingView Webhook Received: {action.upper()}")
        
        # Instantiate clients
        logger = logging.getLogger("webhook")
        bybit = BybitExchange(CONFIG.exchange)
        risk = RiskManager(CONFIG.risk)
        
        # Get position state
        pos = bybit.get_open_position(sym)
        current_qty = float(pos["size"]) if pos else 0.0
        current_side = pos["side"] if pos else ""
        current_position = 1 if current_side == "Buy" else (-1 if current_side == "Sell" else 0)
        
        # If action is sell/flat, close the LONG position
        if action.lower() in ["sell", "flat"]:
            if current_position == 1:
                if not CONFIG.dry_run:
                    bybit.cancel_all_stops(sym)
                    bybit.place_market_order(sym, "Sell", current_qty, reduce_only=True)
                msg = f"Closed LONG qty={current_qty}"
                log_to_file(msg)
                return jsonify({"ok": True, "msg": msg})
            return jsonify({"ok": True, "msg": "No open LONG position to close."})
            
        # If action is buy, open a LONG position
        if action.lower() == "buy":
            if current_position == 1:
                return jsonify({"ok": True, "msg": "Already LONG, ignoring buy."})
                
            # Fetch data to calculate ATR for stop loss
            klines = bybit.get_klines(sym, interval="60", limit=50)
            if klines.empty:
                return jsonify({"ok": False, "msg": "Failed to fetch market data"}), 500
                
            indicators = compute_indicators(klines, CONFIG.strategy)
            price = float(indicators["close"].iloc[-1])
            atr = float(indicators["atr"].iloc[-1])
            stop_price = price - (atr * CONFIG.strategy.atr_stop_mult)
            
            equity = bybit.get_equity()
            if CONFIG.dry_run and equity == 0.0:
                equity = 10000.0
                
            now = datetime.now()
            sizing = risk.size_position(equity, price, stop_price, now)
            
            if sizing.approved:
                if not CONFIG.dry_run:
                    bybit.place_market_order(sym, "Buy", sizing.qty)
                    bybit.place_stop_order(sym, "Sell", sizing.qty, stop_price)
                msg = f"Entered LONG qty={sizing.qty} at price={price} (Stop: {stop_price})"
                log_to_file(msg)
                return jsonify({"ok": True, "msg": msg})
            else:
                msg = f"Risk Manager rejected trade: {sizing.reason}"
                log_to_file(msg)
                return jsonify({"ok": False, "msg": msg})
                
        return jsonify({"ok": False, "msg": f"Unknown action: {action}"}), 400
    except Exception as e:
        logging.exception("Webhook Error")
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/status/<symbol>")
def api_status(symbol):
    state = read_state(symbol)
    return jsonify({
        "symbol":  symbol,
        "running": bot_is_running(symbol),
        "dry_run": os.getenv("DRY_RUN", "true").lower() == "true",
        "testnet": os.getenv("BYBIT_TESTNET", "true").lower() == "true",
        **state,
    })


@app.route("/api/trades/<symbol>")
def api_trades(symbol):
    rows = (read_csv_safe(TRADES_DIR / f"{symbol}_trades.csv") or
            read_csv_safe(BACKTEST_TRADES))
    return jsonify(rows[-200:])


@app.route("/api/equity/<symbol>")
def api_equity(symbol):
    rows = (read_csv_safe(BASE_DIR / f"{symbol}_equity.csv") or
            read_csv_safe(BACKTEST_EQUITY))
    step = max(1, len(rows) // 500)
    return jsonify(rows[::step])


@app.route("/api/logs/<symbol>")
def api_logs(symbol):
    return jsonify(tail_log(symbol))


@app.route("/api/config")
def api_config():
    return jsonify(read_config())


@app.route("/api/start/<symbol>", methods=["POST"])
def api_start(symbol):
    if bot_is_running(symbol):
        return jsonify({"ok": False, "msg": f"{symbol} already running"})
    # Read EMA from environment or defaults
    fast_default, slow_default = {"SOLUSDT": (75, 200), "ETHUSDT": (50, 200), "BTCUSDT": (100, 200)}.get(symbol, (100, 200))
    fast = int(os.getenv(f"{symbol}_FAST_EMA", fast_default))
    slow = int(os.getenv(f"{symbol}_SLOW_EMA", slow_default))
    risk = float(os.getenv("RISK_PER_TRADE", "0.0075"))
    STATE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"{symbol}.log"
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "live.run",
             "--symbol", symbol,
             "--fast-ema", str(fast),
             "--slow-ema", str(slow),
             "--risk", str(risk)],
            cwd=str(BASE_DIR),
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT,
        )
        (STATE_DIR / f"{symbol}.pid").write_text(str(proc.pid))
        return jsonify({"ok": True, "pid": proc.pid})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/stop/<symbol>", methods=["POST"])
def api_stop(symbol):
    pid_file = STATE_DIR / f"{symbol}.pid"
    if not pid_file.exists():
        return jsonify({"ok": False, "msg": f"{symbol} not running"})
    try:
        pid = int(pid_file.read_text().strip())
        if _HAS_PSUTIL:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        else:
            import time
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            # wait up to 5s for graceful exit
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
            # Force kill if still alive
            try:
                os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
            except OSError:
                pass
        pid_file.unlink(missing_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/start_all", methods=["POST"])
def api_start_all():
    results = {}
    for sym in PORTFOLIO_SYMBOLS:
        r = app.test_client().post(f"/api/start/{sym}")
        results[sym] = r.get_json()
    return jsonify(results)


@app.route("/api/stop_all", methods=["POST"])
def api_stop_all():
    results = {}
    for sym in PORTFOLIO_SYMBOLS:
        r = app.test_client().post(f"/api/stop/{sym}")
        results[sym] = r.get_json()
    return jsonify(results)


def update_env(key, value):
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        env_path.write_text("")
    lines = env_path.read_text().splitlines()
    new_lines = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n")


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        data = {
            "TIMEFRAME": os.getenv("TIMEFRAME", "60"),
            "RISK_PER_TRADE": os.getenv("RISK_PER_TRADE", "0.0075"),
            "MIN_ADX": os.getenv("MIN_ADX", "20.0"),
            "MIN_VOLUME_SPIKE": os.getenv("MIN_VOLUME_SPIKE", "1.5"),
            "BYBIT_TESTNET": os.getenv("BYBIT_TESTNET", "true")
        }
        for sym, (f, s) in {"SOLUSDT": (75, 200), "ETHUSDT": (50, 200), "BTCUSDT": (100, 200)}.items():
            data[f"{sym}_FAST_EMA"] = os.getenv(f"{sym}_FAST_EMA", str(f))
            data[f"{sym}_SLOW_EMA"] = os.getenv(f"{sym}_SLOW_EMA", str(s))
        return jsonify(data)
    else:
        data = request.json
        for k, v in data.items():
            update_env(k, str(v))
            os.environ[k] = str(v)
        return jsonify({"ok": True})


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.json
    symbol = data.get("symbol", "BTCUSDT")
    fast = int(data.get("fast_ema", 100))
    slow = int(data.get("slow_ema", 200))
    limit = int(data.get("limit", 1000))
    
    try:
        sys.path.insert(0, str(BASE_DIR))
        from exchange.bybit_client import BybitExchange
        from backtest.run import run_backtest
        from config import CONFIG
        
        # Override config for this backtest run
        CONFIG.strategy.fast_ema = fast
        CONFIG.strategy.slow_ema = slow
        CONFIG.strategy.min_volume_spike = float(data.get("min_volume_spike", 1.5))
        
        client = BybitExchange(CONFIG.exchange)
        df = client.get_klines(symbol, CONFIG.strategy.timeframe, limit=limit)
        eq_df, trades_df = run_backtest(df, starting_equity=10000.0)
        
        # Format for UI chart
        equity_curve = []
        for ts, row in eq_df.iterrows():
            equity_curve.append({"time": ts.isoformat(), "equity": row["equity"]})
            
        trades = []
        trades_df = trades_df.fillna("")
        for _, row in trades_df.iterrows():
            pnl = row.get("pnl", 0)
            if pnl == "":
                pnl = 0
            trades.append({
                "time": row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
                "action": row.get("action", ""),
                "side": row.get("side", ""),
                "price": row.get("price", 0),
                "pnl": pnl
            })
            
        return jsonify({"ok": True, "equity_curve": equity_curve, "trades": trades})
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "msg": str(e), "trace": traceback.format_exc()})


@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    try:
        # Kill all local bots
        api_stop_all()
        
        # Write kill switch override
        update_env("MAX_DAILY_LOSS", "-1.0") # Forces kill switch on restart
        os.environ["MAX_DAILY_LOSS"] = "-1.0"
        
        # Cancel and Close on Exchange
        sys.path.insert(0, str(BASE_DIR))
        from exchange.bybit_client import BybitExchange
        from config import CONFIG
        client = BybitExchange(CONFIG.exchange)
        
        for sym in ["SOLUSDT", "ETHUSDT", "BTCUSDT"]:
            if hasattr(client, "cancel_all_orders"):
                client.cancel_all_orders(sym)
            if hasattr(client, "close_all_positions"):
                client.close_all_positions(sym)
            
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/positions")
def api_positions():
    try:
        sys.path.insert(0, str(BASE_DIR))
        from exchange.bybit_client import BybitExchange
        from config import CONFIG
        client = BybitExchange(CONFIG.exchange)
        
        res = []
        for sym in ["SOLUSDT", "ETHUSDT", "BTCUSDT"]:
            pos = client.get_open_position(sym)
            if pos and float(pos.get("size", 0)) > 0:
                res.append({
                    "symbol": sym,
                    "side": pos.get("side", ""),
                    "size": pos.get("size", "0"),
                    "entry_price": pos.get("avgPrice", "0"),
                    "mark_price": pos.get("markPrice", "0"),
                    "unrealised_pnl": pos.get("unrealisedPnl", "0")
                })
        return jsonify({"ok": True, "positions": res})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Cybrox Ultra Trader</title>
<link rel="manifest" href="/manifest.json"/>
<meta name="theme-color" content="#1a0b2e"/>
<link rel="icon" href="/icon.svg" type="image/svg+xml"/>
<link rel="apple-touch-icon" href="/icon.svg"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #050505; 
    --surface: rgba(17, 24, 39, 0.65); 
    --surface2: rgba(30, 41, 59, 0.7); 
    --border: rgba(255, 255, 255, 0.08);
    --accent: #00f0ff; 
    --accent2: #b026ff; 
    --green: #00ffa3; 
    --red: #ff3366;
    --yellow: #ffd600; 
    --orange: #ff6b00; 
    --text: #f8fafc; 
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { overflow-x: hidden; max-width: 100vw; }
  body {
    background: radial-gradient(circle at top left, #1a0b2e 0%, var(--bg) 40%, var(--bg) 100%);
    background-attachment: fixed;
    color: var(--text);
    font-family: 'Outfit', 'Inter', sans-serif;
    min-height: 100vh;
  }

  header {
    background: rgba(10, 14, 26, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 16px 28px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
  }
  .logo { display: flex; align-items: center; gap: 14px; }
  .logo-icon {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 0 15px rgba(176, 38, 255, 0.4);
  }
  .logo-text { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
  .logo-sub { font-size: 12px; color: var(--muted); letter-spacing: 0.5px; }

  /* Symbol tabs */
  .symbol-tabs { display: flex; gap: 8px; padding: 24px 28px 0; }
  .tab {
    padding: 12px 24px; border-radius: 12px 12px 0 0; font-size: 14px; font-weight: 600;
    cursor: pointer; border: 1px solid transparent; border-bottom: none;
    background: rgba(255, 255, 255, 0.02); color: var(--muted);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex; align-items: center; gap: 10px;
  }
  .tab.active {
    background: var(--surface); color: var(--text);
    border-color: var(--border);
    box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
  }
  .tab:hover:not(.active) { color: var(--text); background: rgba(255, 255, 255, 0.05); }
  .tab-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--red); box-shadow: 0 0 8px var(--red); }
  .tab-dot.running { background: var(--green); box-shadow: 0 0 10px var(--green); animation: pulseGlow 2s infinite; }
  @keyframes pulseGlow { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.1); } }

  /* Portfolio bar */
  .portfolio-bar {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;
    padding: 0 28px 24px; background: var(--surface); border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
  }
  .portfolio-card {
    background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px; display: flex; align-items: center; gap: 14px;
    transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
  }
  .portfolio-card:hover {
    transform: translateY(-3px); background: rgba(255, 255, 255, 0.06);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  }
  .pc-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 8px currentColor; }
  .pc-symbol { font-size: 15px; font-weight: 700; letter-spacing: 0.5px; }
  .pc-ema { font-size: 12px; color: var(--muted); }
  .pc-ret { font-size: 15px; font-weight: 600; text-shadow: 0 0 10px currentColor; }
  .pc-controls { margin-left: auto; display: flex; gap: 8px; }
  .pc-btn {
    padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 700;
    border: none; cursor: pointer; font-family: inherit; transition: all 0.2s;
  }
  .pc-btn.start { background: rgba(0, 255, 163, 0.1); color: var(--green); border: 1px solid rgba(0, 255, 163, 0.2); }
  .pc-btn.start:hover { background: rgba(0, 255, 163, 0.2); box-shadow: 0 0 10px rgba(0, 255, 163, 0.3); }
  .pc-btn.stop { background: rgba(255, 51, 102, 0.1); color: var(--red); border: 1px solid rgba(255, 51, 102, 0.2); }
  .pc-btn.stop:hover { background: rgba(255, 51, 102, 0.2); box-shadow: 0 0 10px rgba(255, 51, 102, 0.3); }
  .pc-btn:disabled { opacity: 0.3; cursor: not-allowed; box-shadow: none; }

  /* Main layout */
  main { padding: 32px 28px; max-width: 1400px; margin: 0 auto; }
  .positions-wrapper { max-width: 1400px; margin: 0 auto; padding: 0 28px; }
  .tab-content { display: none; opacity: 0; transition: opacity 0.3s; } 
  .tab-content.active { display: block; opacity: 1; animation: fadeIn 0.4s ease-out; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 28px; }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 20px; transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(12px); box-shadow: 0 4px 15px rgba(0,0,0,0.2);
  }
  .card:hover { transform: translateY(-4px); border-color: rgba(0, 240, 255, 0.4); box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15); }
  .card-label { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .card-value { font-size: 28px; font-weight: 800; line-height: 1; text-shadow: 0 0 20px transparent; }
  .card-value.green { color: var(--green); text-shadow: 0 0 15px rgba(0, 255, 163, 0.4); } 
  .card-value.red { color: var(--red); text-shadow: 0 0 15px rgba(255, 51, 102, 0.4); }
  .card-value.blue { color: var(--accent); text-shadow: 0 0 15px rgba(0, 240, 255, 0.4); } 
  .card-value.yellow { color: var(--yellow); text-shadow: 0 0 15px rgba(255, 214, 0, 0.4); }
  .card-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }

  .section {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 24px; margin-bottom: 24px; backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2); min-width: 0;
  }
  .section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; min-width: 0; }
  .section-title { font-size: 16px; font-weight: 700; letter-spacing: 0.5px; }
  .section-badge { font-size: 12px; color: var(--accent); background: rgba(0, 240, 255, 0.1); padding: 4px 12px; border-radius: 8px; border: 1px solid rgba(0, 240, 255, 0.2); font-weight: 600; }
  .chart-wrap { position: relative; height: 300px; }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  @media(max-width:900px){ .two-col { grid-template-columns: 1fr; } }

  .table-wrap { overflow-x: auto; max-height: 320px; overflow-y: auto; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; font-family: 'Inter', sans-serif; font-size: 13px; }
  th { position: sticky; top: 0; background: rgba(17, 24, 39, 0.95); color: var(--muted); text-align: left; padding: 12px 16px; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; backdrop-filter: blur(10px); z-index: 10;}
  td { padding: 12px 16px; border-bottom: 1px solid var(--border); transition: background 0.2s; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255, 255, 255, 0.03); }
  
  .tag { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
  .tag.long { background: rgba(0, 255, 163, 0.15); color: var(--green); box-shadow: 0 0 10px rgba(0, 255, 163, 0.2); }
  .tag.short { background: rgba(255, 51, 102, 0.15); color: var(--red); box-shadow: 0 0 10px rgba(255, 51, 102, 0.2); }
  .tag.enter { background: rgba(0, 240, 255, 0.15); color: var(--accent); }
  .tag.exit { background: rgba(176, 38, 255, 0.15); color: var(--accent2); }
  .positive { color: var(--green); font-weight: 600; text-shadow: 0 0 10px rgba(0,255,163,0.3); } 
  .negative { color: var(--red); font-weight: 600; text-shadow: 0 0 10px rgba(255,51,102,0.3); }

  .log-wrap { background: #03060a; border: 1px solid var(--border); border-radius: 12px; padding: 16px; max-height: 320px; overflow-y: auto; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 12px; line-height: 1.7; box-shadow: inset 0 2px 10px rgba(0,0,0,0.5); }
  .log-line { color: #94a3b8; }
  .log-line.error { color: var(--red); } .log-line.warning { color: var(--yellow); }
  .log-line.info-enter { color: var(--green); } .log-line.info-exit { color: var(--accent2); }

  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .btn { display: flex; align-items: center; gap: 8px; padding: 10px 22px; border-radius: 10px; font-size: 14px; font-weight: 700; border: none; cursor: pointer; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); font-family: 'Outfit', sans-serif; position: relative; overflow: hidden; }
  .btn::after { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%); transform: rotate(45deg); transition: all 0.5s; opacity: 0; }
  .btn:hover::after { opacity: 1; transform: rotate(45deg) translate(50%, 50%); }
  
  .btn-start { background: linear-gradient(135deg, #00ffa3, #059669); color: #000; box-shadow: 0 4px 15px rgba(0, 255, 163, 0.3); }
  .btn-start:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 255, 163, 0.5); }
  .btn-stop { background: linear-gradient(135deg, #ff3366, #b91c1c); color: #fff; box-shadow: 0 4px 15px rgba(255, 51, 102, 0.3); }
  .btn-stop:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(255, 51, 102, 0.5); }
  
  .btn-refresh { background: rgba(255, 255, 255, 0.05); color: var(--text); border: 1px solid var(--border); backdrop-filter: blur(8px); }
  .btn-refresh:hover { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 15px rgba(0, 240, 255, 0.2); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none!important; box-shadow: none!important; }
  .btn:disabled::after { display: none; }
  
  .refresh-info { font-size: 13px; color: var(--muted); font-weight: 500; }
  .empty { text-align: center; padding: 40px; color: var(--muted); font-size: 14px; font-weight: 500; }

  .status-pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; border: 1px solid; backdrop-filter: blur(8px); }
  .status-pill.running { background: rgba(0, 255, 163, 0.1); border-color: rgba(0, 255, 163, 0.3); color: var(--green); box-shadow: 0 0 15px rgba(0, 255, 163, 0.1); }
  .status-pill.stopped { background: rgba(255, 51, 102, 0.1); border-color: rgba(255, 51, 102, 0.3); color: var(--red); box-shadow: 0 0 15px rgba(255, 51, 102, 0.1); }
  .s-dot { width: 8px; height: 8px; border-radius: 50%; }
  .running .s-dot { background: var(--green); animation: pulseGlow 2s infinite; box-shadow: 0 0 8px var(--green); }
  .stopped .s-dot { background: var(--red); box-shadow: 0 0 8px var(--red); }
  /* Mobile Responsiveness */
  @media(max-width: 768px) {
    header { flex-direction: column; align-items: flex-start; gap: 16px; padding: 16px; }
    .logo-text { font-size: 16px; }
    .logo-sub { font-size: 10px; }
    .portfolio-bar { padding: 0 16px 16px; grid-template-columns: 1fr; }
    .symbol-tabs { padding: 16px 16px 0; overflow-x: auto; white-space: nowrap; -webkit-overflow-scrolling: touch; }
    main { padding: 16px; }
    .positions-wrapper { padding: 0 16px; }
    .cards { grid-template-columns: 1fr 1fr; gap: 12px; }
    .card { padding: 16px; }
    .card-value { font-size: 22px; }
    .section { padding: 16px; margin-bottom: 16px; }
    .section-header { flex-direction: column; align-items: flex-start; gap: 12px; }
    .controls { width: 100%; justify-content: space-between; }
    .btn { padding: 10px 16px; font-size: 12px; flex: 1; justify-content: center; }
  }
  @media(max-width: 480px) {
    .cards { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">📈</div>
    <div>
      <div class="logo-text">Cybrox Ultra Trader</div>
      <div class="logo-text">Cybrox Quant Terminal</div>
      <div class="logo-sub">Top 30 Volume Scanner &nbsp;|&nbsp; 15M Confluence Engine &nbsp;|&nbsp; Dual TP & 3H Timeout</div>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:center;">
    <button class="btn" onclick="emergencyKillSwitch()" style="background:var(--red);color:white;padding:7px 12px;font-size:12px;font-weight:bold;border-radius:6px;box-shadow: 0 0 15px rgba(255, 51, 102, 0.4); white-space: nowrap;">🚨 HALT ALL</button>
    <button class="btn btn-refresh" onclick="refreshAll()" style="padding:7px 14px;font-size:12px;">↻ Refresh</button>
    <span class="refresh-info" id="last-update">—</span>
  </div>
</header>

<!-- Portfolio overview bar -->
<div class="portfolio-bar" id="portfolio-bar">
  <!-- populated by JS -->
</div>

<div class="positions-wrapper">
  <div class="section" style="margin-top: 10px;">
    <div class="section-header"><div class="section-title">Live Active Positions</div></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>Mark Price</th><th>Live PnL</th></tr></thead>
        <tbody id="positions-body"><tr><td colspan="6" class="empty">No active positions</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<!-- Symbol tabs -->
<div class="symbol-tabs" id="symbol-tabs">
  <div class="tab active" data-sym="QUANT" onclick="switchTab('QUANT')">
    🏛️ Quant Scanner (Top 30)
  </div>
  <div class="tab" data-sym="SOLUSDT" onclick="switchTab('SOLUSDT')">
    <span class="tab-dot" id="dot-SOLUSDT"></span> SOL/USDT
  </div>
  <div class="tab" data-sym="ETHUSDT" onclick="switchTab('ETHUSDT')">
    <span class="tab-dot" id="dot-ETHUSDT"></span> ETH/USDT
  </div>
  <div class="tab" data-sym="BTCUSDT" onclick="switchTab('BTCUSDT')">
    <span class="tab-dot" id="dot-BTCUSDT"></span> BTC/USDT
  </div>
  <div class="tab" data-sym="BACKTEST" onclick="switchTab('BACKTEST')" style="margin-left: auto;">
    🧪 Backtester
  </div>
  <div class="tab" data-sym="SETTINGS" onclick="switchTab('SETTINGS')">
    ⚙️ Settings
  </div>
</div>

<main>
  <!-- QUANT SCANNER TAB -->
  <div class="tab-content active" id="tab-QUANT">
    <div class="cards">
      <div class="card"><div class="card-label">Scanner Status</div><div class="card-value blue" id="qs-status">Active (Top 30)</div><div class="card-sub">scanning 15m candles</div></div>
      <div class="card"><div class="card-label">BTC Regime</div><div class="card-value green" id="qs-regime">Bullish Guard</div><div class="card-sub">1h EMA alignment</div></div>
      <div class="card"><div class="card-label">Active Scanner Trades</div><div class="card-value blue" id="qs-active-count">0 / 3</div><div class="card-sub">max 3 concurrent</div></div>
      <div class="card"><div class="card-label">Max Trade Duration</div><div class="card-value yellow" id="qs-max-hold">3.0 Hours</div><div class="card-sub">stagnant trade auto-exit</div></div>
    </div>

    <!-- Active Scanner Positions & Timeout Tracker -->
    <div class="section" style="margin-top:16px;">
      <div class="section-header">
        <div class="section-title">⏱️ Active Quant Trades & Time Decay Tracker</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side</th>
              <th>Entry Price</th>
              <th>Stop Loss</th>
              <th>TP1 / TP2</th>
              <th>Stagnant Age Timer</th>
              <th>Risk Status</th>
            </tr>
          </thead>
          <tbody id="quant-active-trades">
            <tr><td colspan="7" class="empty">No active scanner positions</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Top 30 Market Volume Scanner Grid -->
    <div class="section" style="margin-top:16px;">
      <div class="section-header">
        <div class="section-title">📊 Top-30 Market Volume Scanner Grid</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Symbol</th>
              <th>Last Price</th>
              <th>Smart Money RVOL</th>
              <th>RSI (14)</th>
              <th>15m Confluence Signal</th>
            </tr>
          </thead>
          <tbody id="quant-scanner-grid">
            <tr><td colspan="6" class="empty">Loading market scanner data...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Quant Terminal Log Stream -->
    <div class="section" style="margin-top:16px;">
      <div class="section-header">
        <div class="section-title">📜 Quant Terminal Activity Log</div>
      </div>
      <div class="log-wrap" id="quant-log-box">Initializing quant scanner log stream...</div>
    </div>
  </div>

  <!-- SOL tab -->
  <div class="tab-content" id="tab-SOLUSDT">
    <div class="cards">
      <div class="card"><div class="card-label">Equity</div><div class="card-value blue" id="eq-SOLUSDT">—</div><div class="card-sub">account value</div></div>
      <div class="card"><div class="card-label">Return</div><div class="card-value" id="ret-SOLUSDT">—</div><div class="card-sub">vs start</div></div>
      <div class="card"><div class="card-label">Position</div><div class="card-value" id="pos-SOLUSDT">Flat</div><div class="card-sub" id="pos-sub-SOLUSDT">no open trade</div></div>
      <div class="card"><div class="card-label">Trades</div><div class="card-value blue" id="trd-SOLUSDT">—</div><div class="card-sub">closed</div></div>
      <div class="card"><div class="card-label">Win Rate</div><div class="card-value" id="wr-SOLUSDT">—</div><div class="card-sub">target ≥ 40%</div></div>
      <div class="card"><div class="card-label">Max Drawdown</div><div class="card-value red" id="dd-SOLUSDT">—</div><div class="card-sub">peak-to-trough</div></div>
    </div>
    <div class="section" style="padding:16px 22px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
        <div class="controls">
          <button class="btn btn-start" id="start-SOLUSDT" onclick="startBot('SOLUSDT')">▶ Start SOL Bot</button>
          <button class="btn btn-stop"  id="stop-SOLUSDT"  onclick="stopBot('SOLUSDT')" disabled>⏹ Stop SOL Bot</button>
        </div>
        <div class="status-pill stopped" id="badge-SOLUSDT"><span class="s-dot"></span><span id="badge-txt-SOLUSDT">Stopped</span></div>
      </div>
    </div>
    <div class="section"><div class="section-header"><div class="section-title">Equity Curve</div><div class="section-badge">SOLUSDT · EMA(75/200)</div></div><div class="chart-wrap"><canvas id="chart-SOLUSDT"></canvas></div></div>
    <div class="two-col">
      <div class="section"><div class="section-header"><div class="section-title">Recent Trades</div><div class="section-badge" id="tc-SOLUSDT">0 trades</div></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Action</th><th>Side</th><th>Price</th><th>PnL</th></tr></thead><tbody id="tb-SOLUSDT"><tr><td colspan="5" class="empty">No trades yet</td></tr></tbody></table></div></div>
      <div class="section"><div class="section-header"><div class="section-title">Live Log</div><div class="section-badge">SOLUSDT.log</div></div><div class="log-wrap" id="log-SOLUSDT"><div class="log-line">Start the bot to see logs...</div></div></div>
    </div>
  </div>

  <!-- ETH tab -->
  <div class="tab-content" id="tab-ETHUSDT">
    <div class="cards">
      <div class="card"><div class="card-label">Equity</div><div class="card-value blue" id="eq-ETHUSDT">—</div><div class="card-sub">account value</div></div>
      <div class="card"><div class="card-label">Return</div><div class="card-value" id="ret-ETHUSDT">—</div><div class="card-sub">vs start</div></div>
      <div class="card"><div class="card-label">Position</div><div class="card-value" id="pos-ETHUSDT">Flat</div><div class="card-sub" id="pos-sub-ETHUSDT">no open trade</div></div>
      <div class="card"><div class="card-label">Trades</div><div class="card-value blue" id="trd-ETHUSDT">—</div><div class="card-sub">closed</div></div>
      <div class="card"><div class="card-label">Win Rate</div><div class="card-value" id="wr-ETHUSDT">—</div><div class="card-sub">target ≥ 40%</div></div>
      <div class="card"><div class="card-label">Max Drawdown</div><div class="card-value red" id="dd-ETHUSDT">—</div><div class="card-sub">peak-to-trough</div></div>
    </div>
    <div class="section" style="padding:16px 22px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
        <div class="controls">
          <button class="btn btn-start" id="start-ETHUSDT" onclick="startBot('ETHUSDT')">▶ Start ETH Bot</button>
          <button class="btn btn-stop"  id="stop-ETHUSDT"  onclick="stopBot('ETHUSDT')" disabled>⏹ Stop ETH Bot</button>
        </div>
        <div class="status-pill stopped" id="badge-ETHUSDT"><span class="s-dot"></span><span id="badge-txt-ETHUSDT">Stopped</span></div>
      </div>
    </div>
    <div class="section"><div class="section-header"><div class="section-title">Equity Curve</div><div class="section-badge">ETHUSDT · EMA(50/200)</div></div><div class="chart-wrap"><canvas id="chart-ETHUSDT"></canvas></div></div>
    <div class="two-col">
      <div class="section"><div class="section-header"><div class="section-title">Recent Trades</div><div class="section-badge" id="tc-ETHUSDT">0 trades</div></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Action</th><th>Side</th><th>Price</th><th>PnL</th></tr></thead><tbody id="tb-ETHUSDT"><tr><td colspan="5" class="empty">No trades yet</td></tr></tbody></table></div></div>
      <div class="section"><div class="section-header"><div class="section-title">Live Log</div><div class="section-badge">ETHUSDT.log</div></div><div class="log-wrap" id="log-ETHUSDT"><div class="log-line">Start the bot to see logs...</div></div></div>
    </div>
  </div>

  <!-- BTC tab -->
  <div class="tab-content" id="tab-BTCUSDT">
    <div class="cards">
      <div class="card"><div class="card-label">Equity</div><div class="card-value blue" id="eq-BTCUSDT">—</div><div class="card-sub">account value</div></div>
      <div class="card"><div class="card-label">Return</div><div class="card-value" id="ret-BTCUSDT">—</div><div class="card-sub">vs start</div></div>
      <div class="card"><div class="card-label">Position</div><div class="card-value" id="pos-BTCUSDT">Flat</div><div class="card-sub" id="pos-sub-BTCUSDT">no open trade</div></div>
      <div class="card"><div class="card-label">Trades</div><div class="card-value blue" id="trd-BTCUSDT">—</div><div class="card-sub">closed</div></div>
      <div class="card"><div class="card-label">Win Rate</div><div class="card-value" id="wr-BTCUSDT">—</div><div class="card-sub">target ≥ 40%</div></div>
      <div class="card"><div class="card-label">Max Drawdown</div><div class="card-value red" id="dd-BTCUSDT">—</div><div class="card-sub">peak-to-trough</div></div>
    </div>
    <div class="section" style="padding:16px 22px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
        <div class="controls">
          <button class="btn btn-start" id="start-BTCUSDT" onclick="startBot('BTCUSDT')">▶ Start BTC Bot</button>
          <button class="btn btn-stop"  id="stop-BTCUSDT"  onclick="stopBot('BTCUSDT')" disabled>⏹ Stop BTC Bot</button>
        </div>
        <div class="status-pill stopped" id="badge-BTCUSDT"><span class="s-dot"></span><span id="badge-txt-BTCUSDT">Stopped</span></div>
      </div>
    </div>
    <div class="section"><div class="section-header"><div class="section-title">Equity Curve</div><div class="section-badge">BTCUSDT · EMA(100/200)</div></div><div class="chart-wrap"><canvas id="chart-BTCUSDT"></canvas></div></div>
    <div class="two-col">
      <div class="section"><div class="section-header"><div class="section-title">Recent Trades</div><div class="section-badge" id="tc-BTCUSDT">0 trades</div></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Action</th><th>Side</th><th>Price</th><th>PnL</th></tr></thead><tbody id="tb-BTCUSDT"><tr><td colspan="5" class="empty">No trades yet</td></tr></tbody></table></div></div>
      <div class="section"><div class="section-header"><div class="section-title">Live Log</div><div class="section-badge">BTCUSDT.log</div></div><div class="log-wrap" id="log-BTCUSDT"><div class="log-line">Start the bot to see logs...</div></div></div>
    </div>
    </div>
  </div>

  <!-- BACKTEST tab -->
  <div class="tab-content" id="tab-BACKTEST">
    <div class="section" style="padding: 24px;">
      <div class="section-header"><div class="section-title">Web-Based Backtester</div></div>
      <div style="display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap;">
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Symbol</label>
          <select id="bt-symbol" onchange="updateBacktestDefaults()" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px; border-radius:6px; cursor:pointer;">
            <option value="BTCUSDT">BTCUSDT</option>
            <option value="ETHUSDT">ETHUSDT</option>
            <option value="SOLUSDT">SOLUSDT</option>
          </select>
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Fast EMA</label>
          <input type="number" id="bt-fast" value="100" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px; border-radius:6px; width:80px;">
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Slow EMA</label>
          <input type="number" id="bt-slow" value="200" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px; border-radius:6px; width:80px;">
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Candles (1H)</label>
          <input type="number" id="bt-limit" value="1000" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px; border-radius:6px; width:100px;">
        </div>
        <div style="display:flex; flex-direction:column; justify-content:flex-end;">
          <button class="btn btn-start" onclick="runBacktest()" id="btn-run-bt">▶ Run Backtest</button>
        </div>
      </div>
      <div class="chart-wrap" style="height: 350px; margin-bottom: 24px;"><canvas id="chart-BACKTEST"></canvas></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>Action</th><th>Side</th><th>Price</th><th>PnL</th></tr></thead>
          <tbody id="tb-BACKTEST"><tr><td colspan="5" class="empty">Run a backtest to see trades</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- SETTINGS tab -->
  <div class="tab-content" id="tab-SETTINGS">
    <div class="section" style="padding: 24px; max-width: 600px; margin: 0 auto;">
      <div class="section-header"><div class="section-title">Live Configuration</div></div>
      <div style="display:flex; flex-direction:column; gap: 20px; margin-top: 20px;">
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Testnet Enabled (true/false)</label>
          <input type="text" id="set-testnet" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px;">
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Risk Per Trade (e.g., 0.0075 for 0.75%)</label>
          <input type="text" id="set-risk" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px;">
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Min ADX Filter (e.g., 20.0)</label>
          <input type="text" id="set-adx" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px;">
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">Min Volume Spike Multiplier (e.g., 1.5)</label>
          <input type="text" id="set-volume" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px;">
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">BTCUSDT EMAs (Fast, Slow)</label>
          <div style="display:flex; gap:10px;">
            <input type="text" id="set-BTCUSDT_FAST_EMA" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px; flex:1;">
            <input type="text" id="set-BTCUSDT_SLOW_EMA" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px; flex:1;">
          </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">ETHUSDT EMAs (Fast, Slow)</label>
          <div style="display:flex; gap:10px;">
            <input type="text" id="set-ETHUSDT_FAST_EMA" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px; flex:1;">
            <input type="text" id="set-ETHUSDT_SLOW_EMA" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px; flex:1;">
          </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <label style="color:var(--muted); font-size:12px;">SOLUSDT EMAs (Fast, Slow)</label>
          <div style="display:flex; gap:10px;">
            <input type="text" id="set-SOLUSDT_FAST_EMA" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px; flex:1;">
            <input type="text" id="set-SOLUSDT_SLOW_EMA" style="background:var(--bg); border:1px solid var(--border); color:var(--text); padding:10px; border-radius:6px; flex:1;">
          </div>
        </div>
        <button class="btn btn-start" onclick="saveSettings()" style="justify-content:center; padding: 12px; margin-top: 10px;">💾 Save Settings</button>
      </div>
    </div>
  </div>
</main>

<script>
const SYMBOLS = ['SOLUSDT','ETHUSDT','BTCUSDT'];
const COLORS  = {SOLUSDT:'#ff6b00', ETHUSDT:'#b026ff', BTCUSDT:'#00f0ff'};
const charts  = {};
let activeTab = 'SOLUSDT';

function switchTab(sym) {
  activeTab = sym;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.sym === sym));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === 'tab-' + sym));
  if (sym === 'SETTINGS') {
    loadSettings();
  } else if (sym !== 'BACKTEST') {
    loadSymbolData(sym);
  }
}

// ── Portfolio bar ─────────────────────────────────────────────────────────────
async function refreshPortfolioBar() {
  try {
    const r = await fetch('/api/portfolio');
    const data = await r.json();
    const bar = document.getElementById('portfolio-bar');
    const emaMap = {SOLUSDT:'EMA(75/200)',ETHUSDT:'EMA(50/200)',BTCUSDT:'EMA(100/200)'};
    bar.innerHTML = data.map(d => {
      const color = d.running ? 'var(--green)' : 'var(--red)';
      const retTxt = d.total_return_pct != null
        ? (d.total_return_pct >= 0 ? '+' : '') + d.total_return_pct.toFixed(2) + '%' : '—';
      const retCol = d.total_return_pct >= 0 ? 'var(--green)' : 'var(--red)';
      return `<div class="portfolio-card" onclick="switchTab('${d.symbol}')" style="cursor:pointer;">
        <div class="pc-dot" style="background:${color};${d.running?'animation:pulse 2s infinite':''}"></div>
        <div style="flex:1;">
          <div class="pc-symbol">${d.symbol.replace('USDT','')}/USDT</div>
          <div class="pc-ema">WR: ${d.win_rate}% | PF: ${d.profit_factor}</div>
        </div>
        <div class="pc-ret" style="color:${retCol}; text-align:right;">
          ${retTxt}
          <div style="font-size:11px; color:var(--muted); text-shadow:none; font-weight:normal;">DD: ${d.drawdown}%</div>
        </div>
        <div class="pc-controls">
          <button class="pc-btn start" onclick="event.stopPropagation();startBot('${d.symbol}')" ${d.running?'disabled':''}>▶</button>
          <button class="pc-btn stop"  onclick="event.stopPropagation();stopBot('${d.symbol}')"  ${!d.running?'disabled':''}>⏹</button>
        </div>
      </div>`;
    }).join('');
  } catch(e) {}
}

// ── Status / controls ─────────────────────────────────────────────────────────
async function fetchStatus(sym) {
  try {
    const r = await fetch(`/api/status/${sym}`);
    const d = await r.json();
    const running = d.running;
    const badge = document.getElementById(`badge-${sym}`);
    badge.className = 'status-pill ' + (running ? 'running' : 'stopped');
    document.getElementById(`badge-txt-${sym}`).textContent = running ? 'Running' : 'Stopped';
    document.getElementById(`dot-${sym}`).className = 'tab-dot' + (running ? ' running' : '');
    document.getElementById(`start-${sym}`).disabled = running;
    document.getElementById(`stop-${sym}`).disabled  = !running;

    const posEl  = document.getElementById(`pos-${sym}`);
    const posVSub = document.getElementById(`pos-sub-${sym}`);
    const pos = d.position ?? 0;
    if (pos === 1)       { posEl.className = 'card-value green'; posEl.textContent = 'LONG';  posVSub.textContent = d.entry_price ? `Entry: $${parseFloat(d.entry_price).toFixed(2)}` : 'In trade'; }
    else if (pos === -1) { posEl.className = 'card-value red';   posEl.textContent = 'SHORT'; posVSub.textContent = 'In trade'; }
    else                 { posEl.className = 'card-value';        posEl.textContent = 'Flat'; posVSub.textContent = 'no open trade'; }
    if (d.equity) document.getElementById(`eq-${sym}`).textContent = '$' + parseFloat(d.equity).toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});
  } catch(e) {}
}

// ── Equity chart ──────────────────────────────────────────────────────────────
async function fetchEquity(sym) {
  try {
    const r = await fetch(`/api/equity/${sym}`);
    const rows = await r.json();
    if (!rows.length) return;
    const labels = rows.map(r => { const d = new Date(r.time||r.timestamp||''); return isNaN(d)?r.time:d.toLocaleDateString('en-GB',{month:'short',day:'numeric'}); });
    const values = rows.map(r => parseFloat(r.equity));
    const ctx = document.getElementById(`chart-${sym}`).getContext('2d');
    const color = COLORS[sym] || '#00f0ff';
    
    const hexToRgb = hex => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : '0, 240, 255';
    };
    const rgbStr = hexToRgb(color);

    if (charts[sym]) charts[sym].destroy();
    charts[sym] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Equity',
          data: values,
          borderColor: color,
          backgroundColor: function(context) {
            const chart = context.chart;
            const {ctx, chartArea} = chart;
            if (!chartArea) return null;
            const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            gradient.addColorStop(0, `rgba(${rgbStr}, 0.5)`);
            gradient.addColorStop(1, `rgba(${rgbStr}, 0.0)`);
            return gradient;
          },
          borderWidth: 3,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: color,
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(17, 24, 39, 0.9)',
            titleFont: { family: 'Outfit', size: 13 },
            bodyFont: { family: 'Inter', size: 14, weight: 'bold' },
            padding: 12,
            borderColor: `rgba(${rgbStr}, 0.3)`,
            borderWidth: 1,
            displayColors: false,
            callbacks: { label: c => ' $' + parseFloat(c.raw).toLocaleString('en', {minimumFractionDigits: 2, maximumFractionDigits: 2}) }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.03)', drawBorder: false }, ticks: { color: '#64748b', maxTicksLimit: 8, font: { family: 'Inter', size: 11 } } },
          y: { grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false }, ticks: { color: '#64748b', font: { family: 'Inter', size: 11 }, callback: v => '$' + v.toLocaleString() } }
        }
      }
    });
    const start=values[0],end=values[values.length-1];
    const ret = start ? (end-start)/start*100 : 0;
    const retEl=document.getElementById(`ret-${sym}`);
    if (isNaN(ret)) {
      retEl.textContent = '0.00%';
      retEl.className = 'card-value';
    } else {
      retEl.textContent=(ret>=0?'+':'')+ret.toFixed(2)+'%';
      retEl.className='card-value '+(ret>=0?'green':'red');
    }
    let peak=values[0],maxDD=0;
    for(const v of values){if(v>peak)peak=v;const dd=peak?(v-peak)/peak*100:0;if(dd<maxDD)maxDD=dd;}
    document.getElementById(`dd-${sym}`).textContent=maxDD.toFixed(2)+'%';
  } catch(e) {}
}

// ── Trades ────────────────────────────────────────────────────────────────────
async function fetchTrades(sym) {
  try {
    const r = await fetch(`/api/trades/${sym}`);
    const rows = await r.json();
    const exits = rows.filter(r=>r.action==='EXIT');
    document.getElementById(`tc-${sym}`).textContent = exits.length+' trades';
    document.getElementById(`trd-${sym}`).textContent = exits.length;
    const wins=exits.filter(r=>parseFloat(r.pnl||0)>0);
    const wr=exits.length?(wins.length/exits.length*100).toFixed(1):'—';
    const wrEl=document.getElementById(`wr-${sym}`);
    wrEl.textContent=exits.length?wr+'%':'—';
    wrEl.className='card-value '+(exits.length?(parseFloat(wr)>=40?'green':'yellow'):'');
    const tbody=document.getElementById(`tb-${sym}`);
    const display=[...rows].reverse().slice(0,50);
    if(!display.length){tbody.innerHTML='<tr><td colspan="5" class="empty">No trades yet</td></tr>';return;}
    tbody.innerHTML=display.map(row=>{
      const pnl=parseFloat(row.pnl||0);
      const pnlStr=row.pnl?(pnl>=0?'+$':'-$')+Math.abs(pnl).toFixed(2):'—';
      const pnlC=!row.pnl?'':(pnl>=0?'positive':'negative');
      const sideTag=row.side?`<span class="tag ${row.side}">${row.side.toUpperCase()}</span>`:'—';
      const actTag=`<span class="tag ${(row.action||'').toLowerCase()}">${row.action||'—'}</span>`;
      const time=(row.time||'').replace('T',' ').slice(0,16);
      const price=row.price?'$'+parseFloat(row.price).toLocaleString():'—';
      return `<tr><td>${time}</td><td>${actTag}</td><td>${sideTag}</td><td>${price}</td><td class="${pnlC}">${pnlStr}</td></tr>`;
    }).join('');
  } catch(e) {}
}

// ── Logs ──────────────────────────────────────────────────────────────────────
async function fetchLogs(sym) {
  try {
    const r = await fetch(`/api/logs/${sym}`);
    const lines = await r.json();
    const wrap = document.getElementById(`log-${sym}`);
    if(!lines.length){wrap.innerHTML='<div class="log-line">No log output yet.</div>';return;}
    wrap.innerHTML=lines.map(l=>{
      let cls='log-line';
      if(l.includes('ERROR')||l.includes('error'))cls+=' error';
      else if(l.includes('WARNING')||l.includes('Kill switch'))cls+=' warning';
      else if(l.includes('ENTER'))cls+=' info-enter';
      else if(l.includes('EXIT'))cls+=' info-exit';
      return `<div class="${cls}">${l}</div>`;
    }).join('');
    wrap.scrollTop=wrap.scrollHeight;
  } catch(e) {}
}

// ── Bot controls ──────────────────────────────────────────────────────────────
async function startBot(sym) {
  document.getElementById(`start-${sym}`).disabled = true;
  try {
    const r = await fetch(`/api/start/${sym}`,{method:'POST'});
    const d = await r.json();
    if(!d.ok) { alert(`Could not start ${sym}: ${d.msg}`); document.getElementById(`start-${sym}`).disabled=false; }
    else setTimeout(()=>loadSymbolData(sym),1500);
  } catch(e){alert('Error: '+e);}
}
async function stopBot(sym) {
  if(!confirm(`Stop the ${sym} bot? Open position stays on the exchange.`))return;
  document.getElementById(`stop-${sym}`).disabled = true;
  try {
    const r = await fetch(`/api/stop/${sym}`,{method:'POST'});
    const d = await r.json();
    if(!d.ok) alert(`Could not stop ${sym}: ${d.msg}`);
    setTimeout(()=>loadSymbolData(sym),1500);
  } catch(e){alert('Error: '+e);}
}

// ── Data loading ──────────────────────────────────────────────────────────────
async function loadSymbolData(sym) {
  await Promise.all([fetchStatus(sym),fetchEquity(sym),fetchTrades(sym),fetchLogs(sym)]);
}

async function refreshQuantScanner() {
  try {
    const res = await fetch('/api/scanner/state');
    const data = await res.json();
    if (!data) return;

    if (data.scanned_data && data.scanned_data.length > 0) {
      let html = '';
      data.scanned_data.forEach((item, idx) => {
        const sigColor = item.signal === 'long' ? 'var(--green)' : (item.signal === 'short' ? 'var(--red)' : 'var(--muted)');
        const sigBadge = `<span style="color:${sigColor}; font-weight:700; text-transform:uppercase;">${item.signal}</span>`;
        html += `<tr>
          <td>#${idx + 1}</td>
          <td style="font-weight:700;">${item.symbol}</td>
          <td>$${item.price}</td>
          <td>${item.rvol}x</td>
          <td>${item.rsi}</td>
          <td>${sigBadge}</td>
        </tr>`;
      });
      const gridEl = document.getElementById('quant-scanner-grid');
      if (gridEl) gridEl.innerHTML = html;
    }

    if (data.active_trades) {
      let activeHtml = '';
      const keys = Object.keys(data.active_trades);
      if (keys.length === 0) {
        activeHtml = '<tr><td colspan="7" class="empty">No active scanner positions</td></tr>';
      } else {
        keys.forEach(sym => {
          const t = data.active_trades[sym];
          const entryTime = new Date(t.entry_time);
          const now = new Date();
          const diffMins = Math.floor((now - entryTime) / 60000);
          const hrs = Math.floor(diffMins / 60);
          const mins = diffMins % 60;
          const tp1Badge = t.tp1_hit ? '<span style="color:var(--green); font-weight:700;">TP1 HIT (RISK-FREE SL)</span>' : 'ACTIVE';
          activeHtml += `<tr>
            <td style="font-weight:700;">${sym}</td>
            <td style="color:${t.side==='Buy'?'var(--green)':'var(--red)'}; font-weight:700;">${t.side}</td>
            <td>$${t.entry_price}</td>
            <td>$${t.stop_price}</td>
            <td>$${t.tp1} / $${t.tp2}</td>
            <td><span class="badge" style="background:rgba(255,107,0,0.2); color:var(--orange); font-weight:700;">${hrs}h ${mins}m / 3h auto-close</span></td>
            <td>${tp1Badge}</td>
          </tr>`;
        });
      }
      const activeEl = document.getElementById('quant-active-trades');
      if (activeEl) activeEl.innerHTML = activeHtml;
      
      const countEl = document.getElementById('qs-active-count');
      if (countEl) countEl.textContent = `${keys.length} / 3`;
    }
  } catch(e) {}
}

async function refreshQuantLogs() {
  try {
    const res = await fetch('/api/scanner/tail_log');
    const data = await res.json();
    if (data && data.lines && data.lines.length > 0) {
      const logEl = document.getElementById('quant-log-box');
      if (logEl) {
        logEl.textContent = data.lines.join('\n');
        logEl.scrollTop = logEl.scrollHeight;
      }
    }
  } catch(e) {}
}

async function refreshAll() {
  await refreshPortfolioBar();
  await refreshQuantScanner();
  await refreshQuantLogs();
  if (activeTab !== 'SETTINGS' && activeTab !== 'BACKTEST' && activeTab !== 'QUANT') {
    await loadSymbolData(activeTab);
  }
  await fetchPositions();
  document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

async function fetchPositions() {
  try {
    const r = await fetch('/api/positions');
    const d = await r.json();
    const tbody = document.getElementById('positions-body');
    if (!d.ok || !d.positions || d.positions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">No active positions</td></tr>';
      return;
    }
    tbody.innerHTML = d.positions.map(p => {
      const color = parseFloat(p.unrealised_pnl) >= 0 ? 'var(--green)' : 'var(--red)';
      return `<tr>
        <td><strong>${p.symbol}</strong></td>
        <td>${p.side}</td>
        <td>${p.size}</td>
        <td>$${parseFloat(p.entry_price).toFixed(2)}</td>
        <td>$${parseFloat(p.mark_price).toFixed(2)}</td>
        <td style="color:${color};font-weight:bold;">$${parseFloat(p.unrealised_pnl).toFixed(2)}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error('fetchPositions error:', e); }
}

async function emergencyKillSwitch() {
  if(!confirm("🚨 WARNING: This will instantly KILL all bots, CANCEL all pending orders, and MARKET CLOSE all open positions! Are you absolutely sure?")) return;
  try {
    const r = await fetch('/api/kill_switch', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      alert("EMERGENCY HALT EXECUTED SUCCESSFULLY.");
      refreshAll();
    } else {
      alert("Failed to execute Kill Switch: " + d.msg);
    }
  } catch(e) { alert("Error: " + e); }
}

async function runBacktest() {
  const btn = document.getElementById('btn-run-bt');
  btn.innerText = "⏳ Running...";
  btn.disabled = true;
  try {
    const r = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol: document.getElementById('bt-symbol').value,
        fast_ema: document.getElementById('bt-fast').value,
        slow_ema: document.getElementById('bt-slow').value,
        limit: document.getElementById('bt-limit').value,
      })
    });
    const d = await r.json();
    if (!d.ok) {
      alert("Backtest failed: " + d.msg);
      return;
    }
    
    // Draw chart
    const ctx = document.getElementById('chart-BACKTEST');
    if (charts['BACKTEST']) charts['BACKTEST'].destroy();
    charts['BACKTEST'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.equity_curve.map(x => new Date(x.time).toLocaleDateString() + ' ' + new Date(x.time).getHours() + ':00'),
        datasets: [{
          label: 'Equity ($)',
          data: d.equity_curve.map(x => x.equity),
          borderColor: 'var(--green)', backgroundColor: 'rgba(0, 255, 163, 0.1)',
          fill: true, tension: 0.1, pointRadius: 0, borderWidth: 2
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 8, color: '#8b8e96' }, grid: { display: false } },
          y: { ticks: { color: '#8b8e96' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
    
    // Draw trades
    const tbody = document.getElementById('tb-BACKTEST');
    if (d.trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty">No trades executed</td></tr>';
    } else {
      tbody.innerHTML = d.trades.slice(-50).reverse().map(t => {
        const pnl = t.pnl ? (t.pnl > 0 ? '+' : '') + t.pnl.toFixed(2) : '—';
        const pnlCol = t.pnl ? (t.pnl > 0 ? 'var(--green)' : 'var(--red)') : 'inherit';
        return `<tr>
          <td>${new Date(t.time).toLocaleString()}</td>
          <td>${t.action}</td>
          <td>${t.side}</td>
          <td>${t.price.toFixed(2)}</td>
          <td style="color:${pnlCol}">${pnl}</td>
        </tr>`;
      }).join('');
    }
  } catch(e) { alert("Error: " + e); } finally {
    btn.innerText = "▶ Run Backtest";
    btn.disabled = false;
  }
}

async function loadSettings() {
  try {
    const r = await fetch('/api/settings');
    const d = await r.json();
    document.getElementById('set-testnet').value = d.BYBIT_TESTNET || '';
    document.getElementById('set-risk').value = d.RISK_PER_TRADE || '';
    document.getElementById('set-adx').value = d.MIN_ADX || '';
    document.getElementById('set-volume').value = d.MIN_VOLUME_SPIKE || '';
    document.getElementById('set-BTCUSDT_FAST_EMA').value = d.BTCUSDT_FAST_EMA || '';
    document.getElementById('set-BTCUSDT_SLOW_EMA').value = d.BTCUSDT_SLOW_EMA || '';
    document.getElementById('set-ETHUSDT_FAST_EMA').value = d.ETHUSDT_FAST_EMA || '';
    document.getElementById('set-ETHUSDT_SLOW_EMA').value = d.ETHUSDT_SLOW_EMA || '';
    document.getElementById('set-SOLUSDT_FAST_EMA').value = d.SOLUSDT_FAST_EMA || '';
    document.getElementById('set-SOLUSDT_SLOW_EMA').value = d.SOLUSDT_SLOW_EMA || '';
  } catch(e) { console.error('Settings load error:', e); }
  updateBacktestDefaults();
}

function updateBacktestDefaults() {
  const sym = document.getElementById('bt-symbol').value;
  const fast = document.getElementById(`set-${sym}_FAST_EMA`)?.value;
  const slow = document.getElementById(`set-${sym}_SLOW_EMA`)?.value;
  if (fast) document.getElementById('bt-fast').value = fast;
  if (slow) document.getElementById('bt-slow').value = slow;
}

async function saveSettings() {
  try {
    const data = {
      BYBIT_TESTNET: document.getElementById('set-testnet').value,
      RISK_PER_TRADE: document.getElementById('set-risk').value,
      MIN_ADX: document.getElementById('set-adx').value,
      MIN_VOLUME_SPIKE: document.getElementById('set-volume').value,
      BTCUSDT_FAST_EMA: document.getElementById('set-BTCUSDT_FAST_EMA').value,
      BTCUSDT_SLOW_EMA: document.getElementById('set-BTCUSDT_SLOW_EMA').value,
      ETHUSDT_FAST_EMA: document.getElementById('set-ETHUSDT_FAST_EMA').value,
      ETHUSDT_SLOW_EMA: document.getElementById('set-ETHUSDT_SLOW_EMA').value,
      SOLUSDT_FAST_EMA: document.getElementById('set-SOLUSDT_FAST_EMA').value,
      SOLUSDT_SLOW_EMA: document.getElementById('set-SOLUSDT_SLOW_EMA').value
    };
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const d = await r.json();
    if (d.ok) alert("Settings saved! They will apply to new bot processes.");
    else alert("Failed to save: " + d.msg);
  } catch(e) { alert("Error: " + e); }
}

// Init
refreshAll();
setInterval(refreshAll, 30000);

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(err => console.log('SW registration failed:', err));
}
</script>
</body>
</html>"""

import threading
import time
import requests

def telegram_listener():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            params = {"timeout": 30}
            if offset:
                params["offset"] = offset
            r = requests.get(url, params=params, timeout=35)
            data = r.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "")
                if str(msg.get("chat", {}).get("id")) == str(chat_id):
                    if text.startswith("/status"):
                        res = []
                        for sym in PORTFOLIO_SYMBOLS:
                            state = read_state(sym)
                            pos = state.get("position", 0)
                            pnl = state.get("unrealized_pnl", 0.0)
                            run = "🟢" if bot_is_running(sym) else "🔴"
                            res.append(f"{run} {sym}: {pos} qty | PnL: ${pnl:.2f}")
                        out = "\n".join(res) if res else "No status available."
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": f"📊 *Live Status:*\n{out}", "parse_mode": "Markdown"})
                    elif text.startswith("/kill"):
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": "🚨 *EXECUTING EMERGENCY KILL SWITCH!*", "parse_mode": "Markdown"})
                        api_kill_switch()
        except Exception as e:
            time.sleep(5)

if __name__ == "__main__":
    t_thread = threading.Thread(target=telegram_listener, daemon=True)
    t_thread.start()
    port = int(os.getenv("DASHBOARD_PORT", 8080))
    print(f"\n  Portfolio Bot Dashboard -> http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
