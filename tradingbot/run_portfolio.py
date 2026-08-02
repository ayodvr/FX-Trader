"""
Portfolio launcher — runs multiple bot instances (one per symbol) as subprocesses.

Each bot gets its own log file (logs/<SYMBOL>.log) and state file (state/<SYMBOL>.json).
The dashboard reads all state files automatically.

Usage:
    python run_portfolio.py           # start all bots
    python run_portfolio.py --stop    # stop all bots
    python run_portfolio.py --status  # show what's running

Process management uses psutil for cross-platform compatibility (Windows + Linux/macOS).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    import os, signal
    _HAS_PSUTIL = False

# ── Portfolio definition ────────────────────────────────────────────────────────
# Each entry: (symbol, fast_ema, slow_ema, risk_per_trade)
# Risk is lower per-symbol (0.5%) since multiple bots run simultaneously.
PORTFOLIO = [
    ("BTCUSDT", 100, 200, 0.005),
    ("ETHUSDT",  50, 200, 0.005),
    ("SOLUSDT",  75, 200, 0.005),
]
# ───────────────────────────────────────────────────────────────────────────────

PID_DIR = Path("state")
PID_DIR.mkdir(exist_ok=True)


def pid_file(symbol: str) -> Path:
    return PID_DIR / f"{symbol}.pid"


def is_running(symbol: str) -> bool:
    """Cross-platform: check if the saved PID is still an alive process."""
    pf = pid_file(symbol)
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
    except (ValueError, OSError):
        pf.unlink(missing_ok=True)
        return False

    if _HAS_PSUTIL:
        alive = psutil.pid_exists(pid)
    else:
        # Fallback: POSIX-only signal(0) trick
        try:
            import os, signal
            os.kill(pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False

    if not alive:
        pf.unlink(missing_ok=True)
    return alive


def kill_process(pid: int):
    """Terminate a process cross-platform."""
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
        import os, signal
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


def start_all(foreground: bool = False):
    Path("logs").mkdir(exist_ok=True)
    procs = []
    for symbol, fast, slow, risk in PORTFOLIO:
        if is_running(symbol):
            print(f"  [{symbol}] already running — skipping")
            continue

        log_path = Path("logs") / f"{symbol}.log"
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "live.run",
                "--symbol", symbol,
                "--fast-ema", str(fast),
                "--slow-ema", str(slow),
                "--risk", str(risk),
            ],
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
        )
        pid_file(symbol).write_text(str(proc.pid))
        print(f"  [{symbol}] started  PID={proc.pid}  EMA({fast}/{slow})  risk={risk*100:.1f}%  log={log_path}")
        procs.append(proc)

    if foreground:
        import time
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            stop_all()


def stop_all():
    for symbol, *_ in PORTFOLIO:
        pf = pid_file(symbol)
        if not pf.exists():
            print(f"  [{symbol}] not running")
            continue
        try:
            pid = int(pf.read_text().strip())
            kill_process(pid)
            pf.unlink(missing_ok=True)
            print(f"  [{symbol}] stopped  PID={pid}")
        except Exception as e:
            print(f"  [{symbol}] error stopping: {e}")


def show_status():
    print(f"\n{'Symbol':<12} {'Running':<12} {'EMA':<14} {'Signal':<10} {'Position':<10} {'Price':<12}")
    print("-" * 70)
    for symbol, fast, slow, risk in PORTFOLIO:
        running = is_running(symbol)
        state_file = PID_DIR / f"{symbol}.json"
        signal_val = position = price = "—"
        if state_file.exists():
            try:
                s = json.loads(state_file.read_text())
                signal_val = s.get("signal", "—")
                pos_int    = s.get("position", 0)
                position   = {1: "LONG", -1: "SHORT", 0: "Flat"}.get(pos_int, "—")
                price      = f"${s.get('price', 0):,.2f}"
            except Exception:
                pass
        status = "✓ Running" if running else "✗ Stopped"
        print(f"  {symbol:<10} {status:<12} EMA({fast}/{slow})   {signal_val:<10} {position:<10} {price}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio bot launcher")
    parser.add_argument("--stop",       action="store_true", help="Stop all bots")
    parser.add_argument("--status",     action="store_true", help="Show status of all bots")
    parser.add_argument("--foreground", "-f", action="store_true", help="Keep parent process alive in foreground (for PM2)")
    args = parser.parse_args()

    if args.stop:
        print("\nStopping all bots...")
        stop_all()
    elif args.status:
        show_status()
    else:
        print("\nStarting portfolio bots...")
        start_all(foreground=args.foreground)
        print("\nAll bots launched. Dashboard: http://localhost:8080\n")
        print("To stop:   python run_portfolio.py --stop")
        print("To status: python run_portfolio.py --status\n")

