#!/usr/bin/env python3
"""Quick parameter sweep for the backtest (uses the last N bars for speed).

One training run is reused; only the backtest parameters (threshold, margin,
cooldown, fees, leverage cap) are swept. This is a research aid, not a full
Bayesian search.

Usage
-----
    python scripts/experiment.py --rows 80000 --folds 3
    python scripts/experiment.py --fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btc_scalper.config import Config  # noqa: E402
from btc_scalper.data import load_klines  # noqa: E402
from btc_scalper.features import add_features  # noqa: E402
from btc_scalper.labels import add_labels  # noqa: E402
from btc_scalper.model import run_walk_forward  # noqa: E402
from btc_scalper.backtest import run_backtest  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sweep backtest params on out-of-sample preds")
    p.add_argument("--rows", type=int, default=120_000, help="last N bars used")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fast", action="store_true", help="80k rows, 3 folds, 200 trees")
    args = p.parse_args(argv)
    if args.fast:
        args.rows = 80_000
        args.folds = 3

    cfg = Config()
    cfg.folds = args.folds
    cfg.lgb_params["n_estimators"] = 250
    cfg.lgb_params["learning_rate"] = 0.04

    print(f"Loading data (last {args.rows:,} bars)...")
    df, src = load_klines(cfg.data_dir, cfg.symbol, cfg.interval, cfg.demo_data_dir)
    df = add_features(df)
    df = add_labels(df, cfg)
    df = df.iloc[-args.rows:].copy()

    print("Training walk-forward model once ...")
    wf = run_walk_forward(df, cfg, label_col="label_up")
    print(f"  mean AUC = {wf.aggregate['mean_auc']:.4f}")

    grid = []
    for threshold in (0.60, 0.65, 0.70):
        for margin in (0.10, 0.20, 0.25):
            for cooldown in (2, 6, 10):
                grid.append((threshold, margin, cooldown, 1.0))

    rows = []
    for th, margin, cool, cap in grid:
        cfg.probability_threshold = th
        cfg.probability_margin = margin
        cfg.cooldown_bars = cool
        cfg.notional_pct_cap = cap
        b = run_backtest(df, wf.predictions, cfg)
        m = b.metrics
        rows.append(
            {
                "threshold": th,
                "margin": margin,
                "cooldown": cool,
                "trades": m["n_trades"],
                "return%": round(m["total_return_pct"], 2),
                "win%": round(m["win_rate"], 2),
                "pf": round(m["profit_factor"], 3),
                "maxdd%": round(m["max_drawdown"], 2),
            }
        )

    out = pd.DataFrame(rows).sort_values("return%", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print("\nSweep results (real {cfg.symbol}, last {} bars):".format(args.rows, cfg=cfg))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
