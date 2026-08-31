"""Central configuration for the BTCUSDT scalping pipeline.

All tuning knobs live here so that the research pipeline, the backtester and
the model code share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    # Real BTCUSDT Binance Vision monthly archives (the format already in the repo).
    data_dir: Path = REPO_ROOT / "Dataset_BTCUSDT"
    # Kept for pair-agnostic fallback support (e.g. a demo archive).
    demo_data_dir: Path = REPO_ROOT / "Dataset_BTCUSDT"
    output_dir: Path = REPO_ROOT / "output" / "BTCUSDT"

    # --------------------------------------------------------------- labels
    label_horizon: int = 1              # bars: next-bar direction
    scalp_bars: int = 12                # max bars to hold for TP/SL label/backtest
    tp_bps: float = 50.0                # take-profit in basis points
    sl_bps: float = 20.0                # stop-loss in basis points

    # ----------------------------------------------------------------- model
    feature_start: int = 60             # drop first N bars (feature lookback)
    folds: int = 6                      # walk-forward folds
    test_frac: float = 0.15             # fraction of each fold used for test
    valid_frac: float = 0.20            # fraction of the train set used for valid
    lgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "n_estimators": 600,
            "num_leaves": 31,
            "max_depth": 6,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 0.3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
    probability_threshold: float = 0.60  # enter long/short only above this
    probability_margin: float = 0.12     # require |P(up)-P(down)| >= this too

    # ------------------------------------------------------------- backtest
    initial_capital: float = 10_000.0
    risk_pct: float = 0.01              # 1% of equity risked per trade
    notional_pct_cap: float = 1.0       # cap position at 100% of equity (no leverage)
    taker_fee_bps: float = 4.0          # Binance Futures taker fee w/ BNB discount
    slippage_bps: float = 1.0           # per-side assumed slippage
    max_hold_bars: int = 12             # aligned with scalp_bars
    cooldown_bars: int = 2              # bars to wait after an exit before re-entry

    # ------------------------------------------------------------- plotting
    figure_dpi: int = 120

    @property
    def dataset_label(self) -> str:
        return f"{self.symbol}-{self.interval}"

    def as_dict(self) -> dict:
        """JSON-serializable config snapshot (for reproducibility)."""
        d = asdict(self)
        for key in ("data_dir", "demo_data_dir", "output_dir"):
            d[key] = str(d[key])
        return d
