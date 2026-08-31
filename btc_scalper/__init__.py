"""BTCUSDT 5-minute scalping research package.

A walk-forward ML framework for high-frequency (5m) BTC/USDT scalping.
It is designed around the exact Binance Vision kline file format that is
present in ``Dataset_BTCUSDT/`` (12-column OHLCV + quote/taker data).
"""

from .config import Config
from .data import load_klines, ensure_demo_dataset
from .features import add_features
from .labels import add_labels
from .model import run_walk_forward, run_walk_forward_dual
from .backtest import run_backtest
from .evaluate import evaluate_backtest

__all__ = [
    "Config",
    "load_klines",
    "ensure_demo_dataset",
    "add_features",
    "add_labels",
    "run_walk_forward",
    "run_walk_forward_dual",
    "run_backtest",
    "evaluate_backtest",
]
