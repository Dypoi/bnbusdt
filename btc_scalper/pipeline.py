"""End-to-end research pipeline: data -> features -> labels -> ML -> backtest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import Config
from .data import ensure_demo_dataset
from .evaluate import evaluate_backtest, plot_predictions
from .features import add_features
from .labels import add_labels
from .model import run_walk_forward, run_walk_forward_dual


def run_pipeline(cfg: Config | None = None, return_results: bool = False):
    cfg = cfg or Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading data for {cfg.symbol} {cfg.interval} ...")
    df, source = ensure_demo_dataset(cfg)
    print(f"      Rows: {len(df):,}  Range: {df.index.min()} -> {df.index.max()}")
    print(f"      Source: {source}")

    print("[2/5] Engineering features ...")
    df = add_features(df)
    print(f"      Feature columns: {df.filter(like='f_').shape[1]}")

    print(f"[3/5] Generating labels (TP {cfg.tp_bps}bp / SL {cfg.sl_bps}bp / {cfg.scalp_bars} bars) ...")
    df = add_labels(df, cfg)
    for col in ("label_up", "label_down", "label_scalp_up", "label_scalp_down"):
        rate = df[col].dropna().mean()
        print(f"      {col}: base rate = {rate:.4f}")

    print(f"[4/5] Walk-forward LightGBM ({cfg.folds} folds) ...")
    # Dual long/short scalp models are the primary target for TP/SL scalping;
    # label_up is retained for comparison and for quick direction experiments.
    wf = run_walk_forward_dual(df, cfg)
    print(f"      Folds trained: {wf.trained_folds}")
    print(
        f"      Mean AUC: {wf.aggregate.get('mean_auc'):.4f} "
        f"(long {wf.aggregate.get('mean_auc_long')} / short {wf.aggregate.get('mean_auc_short')})"
    )

    print("[5/5] Backtesting out-of-sample predictions ...")
    btr = run_backtest(df, wf.predictions, cfg)
    metrics = btr.metrics
    print(
        f"      Trades={metrics['n_trades']}  Return={metrics['total_return_pct']:.2f}%  "
        f"WinRate={metrics['win_rate']:.2f}%  PF={metrics['profit_factor']:.2f}  "
        f"MaxDD={metrics['max_drawdown']:.2f}%  Sharpe={metrics['sharpe']}"
    )

    evaluate_backtest(btr, wf, cfg, cfg.output_dir, source=source)
    plot_predictions(df, wf.predictions, cfg, cfg.output_dir / "predictions.png")
    print(f"\nReport: {cfg.output_dir / 'report.html'}")
    print(f"Summary: {cfg.output_dir / 'summary.json'}")
    print(f"Equity: {cfg.output_dir / 'equity_curve.png'}")

    if return_results:
        return {"df": df, "wf": wf, "backtest": btr}
    return None


def save_training_frame(df: pd.DataFrame, cfg: Config, target: str = "label_up") -> Path:
    """Export the feature/label frame to CSV for external analysis."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / f"{cfg.dataset_label}_features.csv"
    df.dropna(subset=[target]).to_csv(path)
    return path
