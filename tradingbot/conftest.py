"""
conftest.py — project-root pytest configuration.

Adds the tradingbot package directory to sys.path so that all test files
can import from config, strategy, risk, backtest, etc. without needing
manual sys.path hacks inside each test module.

Run tests from any directory:
    pytest                   # from repo root
    pytest tests/ -v         # from tradingbot/
    python -m pytest tests/  # explicit
"""
import sys
from pathlib import Path

# Insert the tradingbot directory so top-level modules (config, strategy, ...)
# are importable regardless of where pytest is invoked from.
TRADINGBOT_DIR = Path(__file__).parent
if str(TRADINGBOT_DIR) not in sys.path:
    sys.path.insert(0, str(TRADINGBOT_DIR))
