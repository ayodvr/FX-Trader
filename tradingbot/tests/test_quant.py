"""
Unit tests for Institutional Quant Strategy, BTC Regime Guard, and Stagnant Timeout Engine.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from config import StrategyConfig
from strategy.quant_strategy import (
    compute_quant_indicators,
    evaluate_btc_regime,
    generate_quant_signal,
    Signal,
)
from live.run_quant_scanner import QuantScannerBot


def _make_dummy_klines(n: int = 100, base_price: float = 100.0, trend: float = 0.05) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="15min")
    prices = [base_price + i * trend + np.sin(i / 3) for i in range(n)]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [1000.0 + (i * 10) for i in range(n)],
    }, index=dates)
    return df


def test_quant_indicators_calculation():
    df = _make_dummy_klines(60)
    ind = compute_quant_indicators(df)
    
    assert "fast_ema" in ind.columns
    assert "slow_ema" in ind.columns
    assert "trend_ema" in ind.columns
    assert "atr" in ind.columns
    assert "rsi" in ind.columns
    assert "rvol" in ind.columns
    assert len(ind) == 60


def test_btc_regime_evaluator():
    df = _make_dummy_klines(60, base_price=60000.0, trend=10.0)
    regime = evaluate_btc_regime(df)
    assert regime in (1, -1, 0)


def test_quant_signal_confluence():
    cfg = StrategyConfig(fast_ema=9, slow_ema=21, long_only=False)
    df = _make_dummy_klines(100, trend=0.2)  # Uptrend
    
    out = generate_quant_signal(df, cfg, current_position=0)
    assert out.signal in (Signal.LONG, Signal.SHORT, Signal.HOLD)
    if out.signal in (Signal.LONG, Signal.SHORT):
        assert out.stop_price is not None
        assert out.tp1_price is not None
        assert out.tp2_price is not None


def test_stagnant_timeout():
    bot = QuantScannerBot()
    now = datetime.now()
    
    # Active trade opened 4 hours ago (exceeding 3.0h limit)
    old_time = now - timedelta(hours=4.0)
    bot.active_scanner_trades["SOLUSDT"] = {
        "side": "Buy",
        "entry_price": 100.0,
        "qty": 1.0,
        "entry_time": old_time,
        "stop_price": 95.0,
        "tp1": 107.5,
        "tp2": 115.0,
        "tp1_hit": False,
    }
    
    bot.check_stagnant_timeouts(now)
    assert "SOLUSDT" not in bot.active_scanner_trades
