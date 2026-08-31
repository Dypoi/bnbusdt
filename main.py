#!/usr/bin/env python3
"""Command-line entry point for the BTCUSDT 5m scalping research pipeline.

Examples
--------
    python main.py                      # run full pipeline with defaults
    python main.py --folds 8
    python main.py --tp 18 --sl 14 --threshold 0.60
    python main.py --quick             # faster, 3 folds / 300 trees
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from btc_scalper.config import Config
from btc_scalper.pipeline import run_pipeline


@dataclass
class Overrides:
    folds: int | None = None
    label_horizon: int | None = None
    scalp_bars: int | None = None
    tp_bps: float | None = None
    sl_bps: float | None = None
    threshold: float | None = None
    margin: float | None = None
    cooldown: int | None = None
    taker_fee_bps: float | None = None
    slippage_bps: float | None = None
    risk_pct: float | None = None
    quick: bool = False


def apply_overrides(cfg: Config, o: Overrides) -> Config:
    if o.folds is not None:
        cfg.folds = o.folds
    if o.label_horizon is not None:
        cfg.label_horizon = o.label_horizon
    if o.scalp_bars is not None:
        cfg.scalp_bars = o.scalp_bars
        cfg.max_hold_bars = o.scalp_bars
    if o.tp_bps is not None:
        cfg.tp_bps = o.tp_bps
    if o.sl_bps is not None:
        cfg.sl_bps = o.sl_bps
    if o.threshold is not None:
        cfg.probability_threshold = o.threshold
    if o.margin is not None:
        cfg.probability_margin = o.margin
    if o.cooldown is not None:
        cfg.cooldown_bars = o.cooldown
    if o.taker_fee_bps is not None:
        cfg.taker_fee_bps = o.taker_fee_bps
    if o.slippage_bps is not None:
        cfg.slippage_bps = o.slippage_bps
    if o.risk_pct is not None:
        cfg.risk_pct = o.risk_pct
    if o.quick:
        cfg.folds = 3
        cfg.lgb_params["n_estimators"] = 300
        cfg.lgb_params["learning_rate"] = 0.05
        cfg.feature_start = 30
    return cfg


def parse_args(argv=None) -> Overrides:
    p = argparse.ArgumentParser(description="BTCUSDT 5m scalping research pipeline")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--label-horizon", type=int, default=None)
    p.add_argument("--scalp-bars", type=int, default=None)
    p.add_argument("--tp", type=float, dest="tp_bps", default=None)
    p.add_argument("--sl", type=float, dest="sl_bps", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--cooldown", type=int, default=None)
    p.add_argument("--taker-fee", type=float, dest="taker_fee_bps", default=None)
    p.add_argument("--slippage", type=float, dest="slippage_bps", default=None)
    p.add_argument("--risk", type=float, dest="risk_pct", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    return Overrides(**vars(args))


def main(argv=None) -> int:
    cfg = Config()
    overrides = parse_args(argv)
    cfg = apply_overrides(cfg, overrides)
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
