#!/usr/bin/env python3
"""Full BTCUSDT research: 6-fold walk-forward + detailed backtest sweep.

Trains the dual long/short TP-event LightGBM once, then runs a detailed grid of
backtests (threshold x margin x cooldown x leverage cap x direction filter) on
out-of-sample predictions. Finally writes:
  * output/BTCUSDT/research_sweep.csv       (grid results, chosen on OOS window)
  * output/BTCUSDT/research_sweep.html      (top configs + metrics)
  * output/BTCUSDT/report_2026H1.json       (Jan-Jun 2026 period report)
  * output/BTCUSDT/2026H1/report.html       (Jan-Jun 2026 period report + plot)

The sweep uses the last ``--sweep-rows`` out-of-sample bars so it completes
quickly; the Jan-Jun 2026 report is always evaluated on the exact period.

Usage
-----
    python scripts/research_full.py --folds 6 --n-est 350
    python scripts/research_full.py --folds 6 --n-est 350 --sweep-rows 120000
    python scripts/research_full.py --folds 6 --n-est 250 --rows 250000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btc_scalper.config import Config  # noqa: E402
from btc_scalper.data import load_klines  # noqa: E402
from btc_scalper.features import add_features, feature_columns  # noqa: E402
from btc_scalper.labels import add_labels  # noqa: E402
from btc_scalper.model import run_walk_forward_dual  # noqa: E402
from btc_scalper.backtest import run_backtest  # noqa: E402


def _mask_predictions(preds: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (preds["open_time"] >= pd.Timestamp(start, tz="UTC")) & (
        preds["open_time"] <= pd.Timestamp(end, tz="UTC")
    )
    return preds.loc[mask].reset_index(drop=True)


def run_sweep(
    cfg: Config,
    df: pd.DataFrame,
    preds: pd.DataFrame,
    sweep_rows: int = 120_000,
    select_end: str | None = None,
) -> pd.DataFrame:
    """Backtest grid on the last ``sweep_rows`` out-of-sample bars.

    When ``select_end`` is supplied, configs are selected only on data up to
    that timestamp, which makes a later period report a true forward test.
    """
    if select_end:
        end_ts = pd.Timestamp(select_end, tz="UTC")
        ddf = df.loc[:end_ts].copy()
        pdf = preds[preds["open_time"].isin(ddf.index)].reset_index(drop=True)
    else:
        ddf = df
        pdf = preds.copy()
    if len(ddf) > sweep_rows:
        ddf = ddf.iloc[-sweep_rows:]
        pdf = pdf[pdf["open_time"].isin(ddf.index)].reset_index(drop=True)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    margins = [0.00, 0.10, 0.20]
    cooldowns = [2, 6, 12]
    notional_caps = [1.0, 3.0]
    directions = ["both", "long", "short"]

    rows = []
    total = len(thresholds) * len(margins) * len(cooldowns) * len(notional_caps) * len(directions)
    idx = 0
    for th in thresholds:
        for marg in margins:
            for cool in cooldowns:
                for cap in notional_caps:
                    for direction in directions:
                        idx += 1
                        p = pdf.copy()
                        if direction == "long":
                            p["prob_down"] = 0.0
                        elif direction == "short":
                            p["prob_up"] = 0.0
                        cfg.probability_threshold = th
                        cfg.probability_margin = marg
                        cfg.cooldown_bars = cool
                        cfg.notional_pct_cap = cap
                        b = run_backtest(ddf, p, cfg)
                        m = b.metrics
                        rows.append(
                            {
                                "threshold": th,
                                "margin": marg,
                                "cooldown": cool,
                                "notional_cap": cap,
                                "direction": direction,
                                "trades": m["n_trades"],
                                "return%": round(m["total_return_pct"], 3),
                                "win%": round(m["win_rate"], 3),
                                "pf": round(m["profit_factor"], 4),
                                "maxdd%": round(m["max_drawdown"], 3),
                                "sharpe": m["sharpe"],
                                "avg_bars": round(m.get("avg_bars_held", 0.0), 3),
                            }
                        )
                        if idx % 40 == 0:
                            print(f"      sweep {idx}/{total}")
    return pd.DataFrame(rows)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BTCUSDT full research + 2026H1 report")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--n-est", type=int, default=350)
    p.add_argument("--rows", type=int, default=None, help="limit training rows for speed")
    p.add_argument("--sweep-rows", type=int, default=120_000, help="OOS bars used by the sweep")
    p.add_argument("--period-start", default="2026-01-01")
    p.add_argument("--period-end", default="2026-06-30")
    p.add_argument("--select-end", default=None,
                   help="only use OOS data up to this date to choose the config "
                        "(e.g. 2025-12-31 for a true forward test)")
    args = p.parse_args(argv)

    cfg = Config()
    cfg.folds = args.folds
    # Smaller test blocks so 6 true walk-forward folds fit inside the archive.
    if args.folds >= 6:
        cfg.test_frac = 0.125
    cfg.lgb_params["n_estimators"] = args.n_est
    cfg.lgb_params["learning_rate"] = 0.04
    cfg.output_dir = ROOT / "output" / "BTCUSDT"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading real BTCUSDT data (rows={args.rows or 'all'}) ...")
    df, source = load_klines(cfg.data_dir, cfg.symbol, cfg.interval, cfg.demo_data_dir)
    print(f"      source={source} rows={len(df):,}")
    print(f"      range={df.index.min()} -> {df.index.max()}")

    print("[2/4] Feature engineering (112 backward-looking features) ...")
    df = add_features(df)
    df = add_labels(df, cfg)
    if args.rows:
        df = df.iloc[-args.rows:].copy()
    print(f"      labeled rows={len(df):,} features={len(feature_columns(df))}")

    print(f"[3/4] Training dual long/short walk-forward ({cfg.folds} folds, {cfg.lgb_params['n_estimators']} trees) ...")
    wf = run_walk_forward_dual(df, cfg)
    print(f"      aggregate={wf.aggregate}")

    print("[4/4] Detailed sweep (same OOS predictions, many backtest configs) ...")
    sweep = run_sweep(cfg, df, wf.predictions, sweep_rows=args.sweep_rows, select_end=args.select_end)
    sweep.to_csv(cfg.output_dir / "research_sweep.csv", index=False)
    # Persist OOS predictions for later reuse/analysis without retraining.
    wf.predictions.to_csv(cfg.output_dir / "predictions_oos.csv", index=False)
    print(f"      saved OOS predictions: {len(wf.predictions):,} rows")

    # Choose config on the OOS sweep, requiring a minimum number of trades.
    good = sweep[sweep["trades"] >= 50].sort_values(["pf", "return%"], ascending=False)
    if good.empty:
        good = sweep.sort_values(["pf", "return%"], ascending=False)
    top = good.iloc[0].to_dict()
    print("\nTop 10 configs by PF (OOS sweep):")
    print(good.head(10).to_string(index=False))
    print(f"\nSelected config: {top}")

    # --- Jan-Jun 2026 report using the selected config ---
    cfg2 = Config()
    cfg2.output_dir = cfg.output_dir
    cfg2.probability_threshold = top["threshold"]
    cfg2.probability_margin = top["margin"]
    cfg2.cooldown_bars = top["cooldown"]
    cfg2.notional_pct_cap = top["notional_cap"]
    p = wf.predictions.copy()
    if top["direction"] == "long":
        p["prob_down"] = 0.0
    elif top["direction"] == "short":
        p["prob_up"] = 0.0

    dfp = df.loc[pd.Timestamp(args.period_start, tz="UTC") : pd.Timestamp(args.period_end, tz="UTC")]
    pp = _mask_predictions(p, args.period_start, args.period_end)
    btr_2026 = run_backtest(dfp, pp, cfg2)
    print("\n===== Jan-Jun 2026 backtest (selected config) =====")
    for k, v in btr_2026.metrics.items():
        print(f"  {k}: {v}")

    from btc_scalper.evaluate import evaluate_backtest  # noqa: E402

    class _FakeWF:
        predictions = pp
        metrics = [{
            "fold": 0,
            "test_start": args.period_start,
            "test_end": args.period_end,
            "test_rows": len(pp),
            "base_rate_long": 0.0,
            "base_rate_short": 0.0,
            "auc_long": 0.0,
            "auc_short": 0.0,
            "acc_long": 0.0,
            "acc_short": 0.0,
        }]
        trained_folds = cfg.folds
        feature_importance = wf.feature_importance
        aggregate = wf.aggregate

    evaluate_backtest(btr_2026, _FakeWF(), cfg2, cfg.output_dir / "2026H1", source="real:BTCUSDT")
    out_json = cfg.output_dir / "report_2026H1.json"
    out_html = cfg.output_dir / "2026H1" / "report.html"
    out_json.write_text(
        json.dumps(
            {
                "source": source,
                "period": {"start": args.period_start, "end": args.period_end},
                "selection_cutoff": args.select_end,
                "model": wf.aggregate,
                "selected_config": top,
                "backtest": btr_2026.metrics,
                "report_html": str(out_html),
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {cfg.output_dir / 'research_sweep.csv'}")
    print(f"Saved: {cfg.output_dir / 'report_2026H1.json'}")
    print(f"Saved: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
