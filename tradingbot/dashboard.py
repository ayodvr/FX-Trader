"""
Cybrox Quant Terminal Dashboard Server — Autonomous Top-30 Scanner UI.

Run with:
    python dashboard.py
"""
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR  = Path(__file__).parent
STATE_DIR = BASE_DIR / "state"
LOG_DIR   = BASE_DIR / "logs"

try:
    from config import CONFIG
    from exchange.bybit_client import BybitExchange
except ImportError:
    pass

from flask import Flask, jsonify, render_template_string, request, Response

app = Flask(__name__)


def check_auth(username, password):
    expected_user = os.getenv("DASHBOARD_USERNAME", "dinosaur")
    expected_pass = os.getenv("DASHBOARD_PASSWORD", "dinosaur123")
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


# ── PWA Endpoints ──────────────────────────────────────────────────────────────

@app.route('/manifest.json')
def pwa_manifest():
    return jsonify({
        "name": "Cybrox Quant Terminal",
        "short_name": "QuantTerminal",
        "description": "Autonomous Top-30 Market Volume Scanner Dashboard",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#050505",
        "theme_color": "#050505",
    })


# ── Helper Functions ──────────────────────────────────────────────────────────

def read_scanner_state() -> dict:
    sf = STATE_DIR / "scanner_state.json"
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {
        "last_update": datetime.now().isoformat(),
        "dry_run": os.getenv("DRY_RUN", "true").lower() == "true",
        "testnet": os.getenv("BYBIT_TESTNET", "true").lower() == "true",
        "top_symbols": [],
        "active_trades": {},
        "scanned_data": [],
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/scanner/state")
def api_scanner_state():
    return jsonify(read_scanner_state())


ENV_FILE = BASE_DIR / ".env"

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify({
            "RISK_PER_TRADE": os.getenv("RISK_PER_TRADE", "0.0075"),
            "LEVERAGE": os.getenv("LEVERAGE", "5"),
            "MAX_HOLD_HOURS": os.getenv("MAX_HOLD_HOURS", "3.0"),
            "MAX_OPEN_POSITIONS": os.getenv("MAX_OPEN_POSITIONS", "3"),
            "BYBIT_TESTNET": os.getenv("BYBIT_TESTNET", "true"),
            "DRY_RUN": os.getenv("DRY_RUN", "true"),
        })
    
    data = request.json or {}
    try:
        lines = []
        if ENV_FILE.exists():
            lines = ENV_FILE.read_text().splitlines()
            
        kv_map = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                kv_map[k.strip()] = v.strip()
                
        for key in ["RISK_PER_TRADE", "LEVERAGE", "MAX_HOLD_HOURS", "MAX_OPEN_POSITIONS", "BYBIT_TESTNET", "DRY_RUN"]:
            if key in data:
                val = str(data[key]).strip()
                kv_map[key] = val
                os.environ[key] = val
                
        new_lines = [f"{k}={v}" for k, v in kv_map.items()]
        ENV_FILE.write_text("\n".join(new_lines) + "\n")
        return jsonify({"ok": True, "msg": "Settings saved successfully!"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/scanner/tail_log")
def api_scanner_tail_log():
    log_file = LOG_DIR / "quant_scanner.log"
    if not log_file.exists():
        return jsonify({"lines": ["Waiting for quant_scanner.log..."]})
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        return jsonify({"lines": lines[-150:]})
    except Exception as e:
        return jsonify({"lines": [f"Error reading log: {e}"]})


@app.route("/api/positions")
def api_positions():
    try:
        bybit = BybitExchange(CONFIG.exchange)
        top_symbols = bybit.get_top_symbols(limit=10)
        res = []
        for sym in top_symbols:
            pos = bybit.get_open_position(sym)
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
        return jsonify({"ok": True, "positions": []})


@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    try:
        bybit = BybitExchange(CONFIG.exchange)
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "AVAXUSDT"]
        for sym in symbols:
            try:
                bybit.cancel_all_orders(sym)
                bybit.close_all_positions(sym)
            except Exception:
                pass
        return jsonify({"ok": True, "msg": "Kill switch triggered"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


# ── Frontend Template ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Cybrox Quant Terminal</title>
<link rel="manifest" href="/manifest.json"/>
<meta name="theme-color" content="#050505"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
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

  /* Portfolio bar */
  .portfolio-bar {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;
    padding: 24px 28px; background: var(--surface); border-bottom: 1px solid var(--border);
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
  .pc-symbol { font-size: 14px; font-weight: 700; letter-spacing: 0.5px; }
  .pc-ema { font-size: 11px; color: var(--muted); }
  .pc-ret { font-size: 14px; font-weight: 700; text-shadow: 0 0 10px currentColor; }

  main { padding: 24px 28px; max-width: 1600px; margin: 0 auto; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 20px;
    backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    transition: transform 0.2s;
  }
  .card:hover { transform: translateY(-2px); }
  .card-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
  .card-value { font-size: 24px; font-weight: 800; margin: 8px 0 4px; letter-spacing: -0.5px; }
  .card-value.green { color: var(--green); text-shadow: 0 0 15px rgba(0,255,163,0.3); }
  .card-value.blue { color: var(--accent); text-shadow: 0 0 15px rgba(0,240,255,0.3); }
  .card-value.red { color: var(--red); text-shadow: 0 0 15px rgba(255,51,102,0.3); }
  .card-value.yellow { color: var(--yellow); }
  .card-sub { font-size: 12px; color: var(--muted); }

  .section {
    background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
    padding: 24px; margin-bottom: 24px; backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  }
  .section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
  .section-title { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }

  .table-wrap { overflow-x: auto; width: 100%; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
  th { padding: 12px 16px; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
  td { padding: 14px 16px; border-bottom: 1px solid var(--border); transition: background 0.2s; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255, 255, 255, 0.03); }
  
  .log-wrap { background: #03060a; border: 1px solid var(--border); border-radius: 12px; padding: 16px; max-height: 320px; overflow-y: auto; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.7; box-shadow: inset 0 2px 10px rgba(0,0,0,0.5); }

  .btn { display: inline-flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 10px; font-size: 13px; font-weight: 700; border: none; cursor: pointer; transition: all 0.2s; font-family: 'Outfit', sans-serif; }
  .btn-refresh { background: rgba(255, 255, 255, 0.05); color: var(--text); border: 1px solid var(--border); }
  .btn-refresh:hover { border-color: var(--accent); color: var(--accent); }
  .empty { text-align: center; padding: 30px; color: var(--muted); font-size: 14px; }
  .refresh-info { font-size: 12px; color: var(--muted); }

  /* Mobile Responsiveness */
  @media (max-width: 768px) {
    header {
      flex-direction: column;
      align-items: stretch;
      gap: 12px;
      padding: 14px 16px;
    }
    .header-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      gap: 10px;
    }
    .logo-text { font-size: 18px; white-space: nowrap; }
    .logo-sub { font-size: 11px; }
    .portfolio-bar { padding: 16px; grid-template-columns: 1fr; }
    main { padding: 16px; }
    .cards { grid-template-columns: 1fr 1fr; gap: 10px; }
    .card { padding: 14px; }
    .card-value { font-size: 20px; }
    .section { padding: 16px; border-radius: 14px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🏛️</div>
    <div>
      <div class="logo-text">Cybrox Quant Terminal</div>
      <div class="logo-sub">Top 30 Volume Scanner &nbsp;|&nbsp; 15M Confluence &nbsp;|&nbsp; Dual TP & 3H Timeout</div>
    </div>
  </div>
  <div class="header-actions">
    <button class="btn btn-refresh" onclick="toggleSettingsPanel()" style="border-color:var(--accent); color:var(--accent);">⚙️ Settings</button>
    <button class="btn" onclick="emergencyKillSwitch()" style="background:var(--red);color:white;box-shadow: 0 0 15px rgba(255, 51, 102, 0.4);">🚨 HALT ALL</button>
    <button class="btn btn-refresh" onclick="refreshAll()">↻ Refresh</button>
    <span class="refresh-info" id="last-update">—</span>
  </div>
</header>

<!-- Collapsible Live Settings Panel -->
<div id="settings-panel" style="display:none; background: var(--surface2); border-bottom: 1px solid var(--border); padding: 20px 28px; backdrop-filter: blur(16px);">
  <div style="max-width:1200px; margin:0 auto;">
    <div style="font-size:16px; font-weight:700; margin-bottom:14px; color:var(--accent);">⚙️ Live Bot Configuration Panel</div>
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin-bottom:16px;">
      <div>
        <label style="font-size:12px; color:var(--muted); display:block; margin-bottom:6px;">Risk Per Trade (e.g. 0.0075 = 0.75%)</label>
        <input type="text" id="cfg-risk" value="0.0075" style="width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px 12px; border-radius:8px; font-size:14px;">
      </div>
      <div>
        <label style="font-size:12px; color:var(--muted); display:block; margin-bottom:6px;">Futures Leverage</label>
        <select id="cfg-leverage" style="width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px 12px; border-radius:8px; font-size:14px;">
          <option value="2">2X Leverage</option>
          <option value="5" selected>5X Leverage</option>
          <option value="10">10X Leverage</option>
          <option value="20">20X Leverage</option>
        </select>
      </div>
      <div>
        <label style="font-size:12px; color:var(--muted); display:block; margin-bottom:6px;">Stagnant Timeout (Hours)</label>
        <input type="text" id="cfg-hold" value="3.0" style="width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px 12px; border-radius:8px; font-size:14px;">
      </div>
      <div>
        <label style="font-size:12px; color:var(--muted); display:block; margin-bottom:6px;">Max Concurrent Trades</label>
        <input type="text" id="cfg-positions" value="3" style="width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px 12px; border-radius:8px; font-size:14px;">
      </div>
    </div>
    <div style="display:flex; justify-content:flex-end; gap:10px;">
      <button class="btn btn-refresh" onclick="toggleSettingsPanel()">Cancel</button>
      <button class="btn" onclick="saveWebSettings()" style="background:linear-gradient(135deg, var(--accent2), var(--accent)); color:#000; font-weight:800; box-shadow:0 0 15px rgba(0,240,255,0.4);">💾 Save & Apply Live</button>
    </div>
  </div>
</div>

<!-- Portfolio summary bar -->
<div class="portfolio-bar" id="portfolio-bar">
  <!-- populated by JS -->
</div>

<main>
  <div class="cards">
    <div class="card"><div class="card-label">Scanner Status</div><div class="card-value blue" id="qs-status">Active (Top 30)</div><div class="card-sub">scanning 15m candles</div></div>
    <div class="card"><div class="card-label">BTC Regime Guard</div><div class="card-value green" id="qs-regime">Bullish Alignment</div><div class="card-sub">1h EMA trend check</div></div>
    <div class="card"><div class="card-label">Active Quant Trades</div><div class="card-value blue" id="qs-active-count">0 / 3</div><div class="card-sub">max 3 concurrent</div></div>
    <div class="card"><div class="card-label">Time Decay Timeout</div><div class="card-value yellow" id="qs-max-hold">3.0 Hours</div><div class="card-sub">stagnant trade auto-exit</div></div>
  </div>

  <!-- Active Quant Trades & Time Decay Tracker -->
  <div class="section">
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
  <div class="section">
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
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:16px;">
      <button class="btn btn-refresh" onclick="changePage(-1)" id="btn-prev" style="padding:6px 14px; font-size:12px;">◀ Previous</button>
      <span id="page-info" style="font-size:12px; color:var(--muted); font-weight:600;">Page 1 of 3 (10 coins/page)</span>
      <button class="btn btn-refresh" onclick="changePage(1)" id="btn-next" style="padding:6px 14px; font-size:12px;">Next ▶</button>
    </div>
  </div>

  <!-- Quant Terminal Activity Log -->
  <div class="section">
    <div class="section-header">
      <div class="section-title">📜 Quant Terminal Activity Log</div>
    </div>
    <div class="log-wrap" id="quant-log-box">Initializing quant scanner log stream...</div>
  </div>
</main>

<script>
let currentPage = 1;
const pageSize = 10;
let lastScannedData = [];

function changePage(delta) {
  const maxPages = Math.ceil((lastScannedData.length || 30) / pageSize);
  currentPage += delta;
  if (currentPage < 1) currentPage = 1;
  if (currentPage > maxPages) currentPage = maxPages;
  renderScannerGrid();
}

function renderScannerGrid() {
  if (!lastScannedData || lastScannedData.length === 0) return;
  
  const maxPages = Math.ceil(lastScannedData.length / pageSize);
  if (currentPage > maxPages) currentPage = maxPages;
  
  const startIdx = (currentPage - 1) * pageSize;
  const endIdx = startIdx + pageSize;
  const pageItems = lastScannedData.slice(startIdx, endIdx);

  let html = '';
  pageItems.forEach((item, idx) => {
    const globalRank = startIdx + idx + 1;
    const sigColor = item.signal === 'long' ? 'var(--green)' : (item.signal === 'short' ? 'var(--red)' : 'var(--muted)');
    const sigBadge = `<span style="color:${sigColor}; font-weight:700; text-transform:uppercase;">${item.signal}</span>`;
    html += `<tr>
      <td>#${globalRank}</td>
      <td style="font-weight:700;">${item.symbol}</td>
      <td>$${item.price}</td>
      <td>${item.rvol}x</td>
      <td>${item.rsi}</td>
      <td>${sigBadge}</td>
    </tr>`;
  });

  const gridEl = document.getElementById('quant-scanner-grid');
  if (gridEl) gridEl.innerHTML = html;

  const pageInfo = document.getElementById('page-info');
  if (pageInfo) pageInfo.textContent = `Page ${currentPage} of ${maxPages} (${pageSize} coins/page)`;

  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  if (btnPrev) btnPrev.disabled = (currentPage === 1);
  if (btnNext) btnNext.disabled = (currentPage === maxPages);
}

async function refreshPortfolioBar() {
  try {
    const res = await fetch('/api/scanner/state');
    const d = await res.json();
    const bar = document.getElementById('portfolio-bar');
    if (!bar) return;
    
    const activeTrades = d.active_trades ? Object.keys(d.active_trades).length : 0;
    const isDryRun = d.dry_run !== false;
    const modeStr = isDryRun ? 'Paper Trading (Testnet)' : 'Live Trading (Mainnet)';
    
    bar.innerHTML = `
      <div class="portfolio-card">
        <div class="pc-dot" style="background:var(--green); animation:pulse 2s infinite;"></div>
        <div style="flex:1;">
          <div class="pc-symbol">QUANT MARKET SCANNER</div>
          <div class="pc-ema">Top 30 Volume Altcoins | 15M Confluence Engine</div>
        </div>
        <div class="pc-ret" style="color:var(--green); text-align:right;">
          ${modeStr}
        </div>
      </div>

      <div class="portfolio-card">
        <div class="pc-dot" style="background:var(--accent);"></div>
        <div style="flex:1;">
          <div class="pc-symbol">ACTIVE SCANNER POSITIONS</div>
          <div class="pc-ema">Max 3 concurrent trades allowed</div>
        </div>
        <div class="pc-ret" style="color:var(--accent); text-align:right;">
          ${activeTrades} / 3 Active
        </div>
      </div>

      <div class="portfolio-card">
        <div class="pc-dot" style="background:var(--yellow);"></div>
        <div style="flex:1;">
          <div class="pc-symbol">TIME DECAY PROTECTION</div>
          <div class="pc-ema">Stagnant Trade Auto-Exit Limit</div>
        </div>
        <div class="pc-ret" style="color:var(--yellow); text-align:right;">
          3.0 Hours Limit
        </div>
      </div>
    `;
  } catch(e) {}
}

async function refreshQuantScanner() {
  try {
    const res = await fetch('/api/scanner/state');
    const data = await res.json();
    if (!data) return;

    if (data.scanned_data && data.scanned_data.length > 0) {
      lastScannedData = data.scanned_data;
      renderScannerGrid();
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
            <td><span style="background:rgba(255,107,0,0.2); color:var(--orange); font-weight:700; padding:3px 8px; border-radius:6px;">${hrs}h ${mins}m / 3h auto-close</span></td>
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

async function emergencyKillSwitch() {
  if(!confirm("🚨 WARNING: This will instantly CANCEL all pending orders and MARKET CLOSE open positions! Are you sure?")) return;
  try {
    const r = await fetch('/api/kill_switch', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      alert("EMERGENCY HALT EXECUTED.");
      refreshAll();
    } else {
      alert("Failed: " + d.msg);
    }
  } catch(e) { alert("Error: " + e); }
}

function toggleSettingsPanel() {
  const panel = document.getElementById('settings-panel');
  if (panel.style.display === 'none' || !panel.style.display) {
    panel.style.display = 'block';
    loadWebSettings();
  } else {
    panel.style.display = 'none';
  }
}

async function loadWebSettings() {
  try {
    const res = await fetch('/api/settings');
    const d = await res.json();
    if (d.RISK_PER_TRADE) document.getElementById('cfg-risk').value = d.RISK_PER_TRADE;
    if (d.LEVERAGE) document.getElementById('cfg-leverage').value = d.LEVERAGE;
    if (d.MAX_HOLD_HOURS) document.getElementById('cfg-hold').value = d.MAX_HOLD_HOURS;
    if (d.MAX_OPEN_POSITIONS) document.getElementById('cfg-positions').value = d.MAX_OPEN_POSITIONS;
  } catch(e) {}
}

async function saveWebSettings() {
  const data = {
    RISK_PER_TRADE: document.getElementById('cfg-risk').value,
    LEVERAGE: document.getElementById('cfg-leverage').value,
    MAX_HOLD_HOURS: document.getElementById('cfg-hold').value,
    MAX_OPEN_POSITIONS: document.getElementById('cfg-positions').value,
  };
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const d = await res.json();
    if (d.ok) {
      alert("✅ Live Settings Saved and Applied Successfully!");
      toggleSettingsPanel();
      refreshAll();
    } else {
      alert("❌ Failed to save settings: " + d.msg);
    }
  } catch(e) { alert("Error saving settings: " + e); }
}

async function refreshAll() {
  await refreshPortfolioBar();
  await refreshQuantScanner();
  await refreshQuantLogs();
  document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

refreshAll();
setInterval(refreshAll, 15000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 8081))
    print(f"\n  Cybrox Quant Terminal Dashboard -> http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
