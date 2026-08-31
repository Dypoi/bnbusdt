#!/usr/bin/env python3
"""Adaptive TP/SL MFE-MAE research + strict-forward Jan-Jun 2026 comparison.

Steps
-----
1. Build MFE (max favorable excursion) and MAE (max adverse excursion) labels
   over the next ``scalp_bars`` M5 bars.
2. Train an adaptive walk-forward model:
   - binary long/short confidence (P of TP-before-SL),
   - quantile regression for long/short MFE and MAE.
3. Turn the MFE/MAE predictions + ATR into a *dynamic* TP/SL per signal bar:
      tp_bps = clip(tp_scale * pred_mfe + atr_weight * atr_pct, min_tp, max_tp)
      sl_bps = clip(sl_scale * pred_mae + atr_weight * atr_pct, min_sl, max_sl)
4. Select config on OOS data <= ``--select-end`` (default 2025-12-31), then run
   a strict forward test on Jan-Jun 2026.
5. Report the adaptive config, plus the same-signal fixed TP/SL
   configurations A/B for comparison.

Outputs
-------
output/BTCUSDT/adaptive_sweep.csv
output/BTCUSDT/adaptive_predictions.csv
output/BTCUSDT/adaptive_2026H1.json
output/BTCUSDT/adaptive_2026H1/report.html
output/BTCUSDT/adaptive_2026H1/equity_curve.png
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
from btc_scalper.labels import add_labels, add_mfe_mae_labels  # noqa: E402
from btc_scalper.model import run_walk_forward_adaptive  # noqa: E402
from btc_scalper.backtest import run_backtest, run_backtest_dynamic  # noqa: E402


def _clip(vals, lo, hi):
    return np.clip(vals, lo, hi)


def build_dynamic_levels(preds: pd.DataFrame, df: pd.DataFrame, tp_scale: float,
                         sl_scale: float, atr_weight: float,
                         min_tp: float, max_tp: float, min_sl: float, max_sl: float) -> pd.DataFrame:
    """Create per-bar tp_bps / sl_bps from predicted MFE/MAE + ATR."""
    out = preds.copy()
    atr = df["f_natr_14"].reindex(preds["open_time"])
    atr_bps = atr.fillna(0.0).to_numpy(dtype=float) * 10_000.0

    mfe_l = out["mfe_long_bps"].fillna(0.0).to_numpy(dtype=float)
    mae_l = out["mae_long_bps"].fillna(0.0).to_numpy(dtype=float)
    mfe_s = out["mfe_short_bps"].fillna(0.0).to_numpy(dtype=float)
    mae_s = out["mae_short_bps"].fillna(0.0).to_numpy(dtype=float)

    tp_long = _clip(tp_scale * mfe_l + atr_weight * atr_bps, min_tp, max_tp)
    sl_long = _clip(sl_scale * mae_l + atr_weight * atr_bps, min_sl, max_sl)
    tp_short = _clip(tp_scale * mfe_s + atr_weight * atr_bps, min_tp, max_tp)
    sl_short = _clip(sl_scale * mae_s + atr_weight * atr_bps, min_sl, max_sl)

    # Default to the long levels; the backtest uses the signal-bar value of the
    # executed side, so build combined levels that depend on the winning side.
    diff = out["prob_up"] - out["prob_down"]
    tp = tp_long.copy()
    sl = sl_long.copy()
    short_mask = (diff.to_numpy() < 0)
    tp[short_mask] = tp_short[short_mask]
    sl[short_mask] = sl_short[short_mask]

    out["tp_bps"] = tp
    out["sl_bps"] = sl

    # Risk-reward quality per bar: predicted MFE/MAE for the direction the
    # signal is leaning toward. Used as an optional filter (rr_min).
    rr_long = np.divide(mfe_l, np.where(mae_l == 0.0, np.nan, mae_l),
                        out=np.full_like(mfe_l, np.nan),
                        where=mae_l != 0.0)
    rr_short = np.divide(mfe_s, np.where(mae_s == 0.0, np.nan, mae_s),
                         out=np.full_like(mfe_s, np.nan),
                         where=mae_s != 0.0)
    diff_vals = out["prob_up"].to_numpy(dtype=float) - out["prob_down"].to_numpy(dtype=float)
    rr_quality = np.where(diff_vals >= 0, rr_long, rr_short)
    out["rr_quality"] = np.nan_to_num(rr_quality, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _mask(preds: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    if start is None:
        return preds
    m = (preds["open_time"] >= pd.Timestamp(start, tz="UTC")) & (
        preds["open_time"] <= pd.Timestamp(end, tz="UTC")
    )
    return preds.loc[m].reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--n-est", type=int, default=300)
    p.add_argument("--rows", type=int, default=None, help="limit training rows")
    p.add_argument("--select-end", default="2025-12-31")
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--min-trades-adapt", type=int, default=25,
                   help="min trades required on the selection window")
    args = p.parse_args(argv)

    cfg = Config()
    cfg.folds = args.folds
    if args.folds >= 6:
        cfg.test_frac = 0.125
    cfg.lgb_params["n_estimators"] = args.n_est
    cfg.lgb_params["learning_rate"] = 0.04
    cfg.output_dir = ROOT / "output" / "BTCUSDT"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Load BTCUSDT M5 ...")
    df, src = load_klines(cfg.data_dir, cfg.symbol, cfg.interval, cfg.demo_data_dir)
    print(f"      source={src} rows={len(df):,}")

    print("[2/5] Features + labels + MFE/MAE ...")
    df = add_features(df)
    df = add_labels(df, cfg)
    df = add_mfe_mae_labels(df, cfg, hold=cfg.scalp_bars)
    if args.rows:
        df = df.iloc[-args.rows:].copy()
    # Register ATR for the adaptive levels (already created by add_features).
    print(f"      rows={len(df):,} features={len(feature_columns(df))}")

    print(f"[3/5] Adaptive walk-forward ({cfg.folds} folds) ...")
    wf = run_walk_forward_adaptive(df, cfg)
    print(f"      aggregate={wf.aggregate}")
    preds = wf.predictions.copy()
    preds.to_csv(cfg.output_dir / "adaptive_predictions.csv", index=False)

    # Quick sanity: what is the predicted MFE / MAE distribution?
    print("      Predicted MFE/MAE (bps) distribution:")
    for c in ["mfe_long_bps", "mae_long_bps", "mfe_short_bps", "mae_short_bps"]:
        q = preds[c].quantile([0.25, 0.5, 0.75]).round(1).tolist()
        print(f"        {c}: q25/q50/q75 = {q}")

    # ---------- Adaptive config sweep (strict selection window) ----------
    print("[4/5] Sweep adaptive configs on data <= " + str(args.select_end) + " ...")
    select_end_ts = pd.Timestamp(args.select_end, tz="UTC")
    sel_preds = _mask(preds, None, args.select_end)
    sel_df = df.loc[:select_end_ts]

    grid = []
    for threshold in (0.60, 0.65):
        for margin in (0.00, 0.10):
            for cooldown in (2, 6):
                for cap in (1.0, 3.0):
                    for direction in ("long", "both"):
                        for tp_scale in (0.70, 1.00):
                            for sl_scale in (0.80, 1.00):
                                for atr_weight in (0.00, 0.20):
                                    for rr_min in (0.80, 1.00):
                                        grid.append({
                                            "threshold": threshold,
                                            "margin": margin,
                                            "cooldown": cooldown,
                                            "cap": cap,
                                            "direction": direction,
                                            "tp_scale": tp_scale,
                                            "sl_scale": sl_scale,
                                            "atr_weight": atr_weight,
                                            "rr_min": rr_min,
                                        })

    sweep_rows = []
    for idx, g in enumerate(grid):
        lv = build_dynamic_levels(sel_preds, sel_df, g["tp_scale"], g["sl_scale"],
                                  g["atr_weight"], min_tp=10, max_tp=100,
                                  min_sl=8, max_sl=80)
        # Only allow trades where predicted RR (MFE/MAE, direction-adjusted) is good.
        rr_ok = lv["rr_quality"].to_numpy(dtype=float) >= g["rr_min"]
        lv.loc[~rr_ok, ["prob_up", "prob_down"]] = 0.0
        p = lv.copy()
        if g["direction"] == "long":
            p["prob_down"] = 0.0
        elif g["direction"] == "short":
            p["prob_up"] = 0.0
        c = Config()
        c.probability_threshold = g["threshold"]
        c.probability_margin = g["margin"]
        c.cooldown_bars = g["cooldown"]
        c.notional_pct_cap = g["cap"]
        b = run_backtest_dynamic(sel_df, p, c)
        m = b.metrics
        sweep_rows.append({
            **g,
            "trades": m["n_trades"],
            "return%": round(m["total_return_pct"], 3),
            "win%": round(m["win_rate"], 3),
            "pf": round(m["profit_factor"], 4),
            "maxdd%": round(m["max_drawdown"], 3),
            "sharpe": m["sharpe"],
            "avg_tp_bps": round(m.get("avg_tp_bps", 0), 2),
            "avg_sl_bps": round(m.get("avg_sl_bps", 0), 2),
        })
        if (idx + 1) % 32 == 0:
            print(f"      adaptive sweep {idx+1}/{len(grid)}")
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(cfg.output_dir / "adaptive_sweep.csv", index=False)

    good = sweep[sweep["trades"] >= args.min_trades_adapt].sort_values(
        ["pf", "return%"], ascending=False
    )
    if good.empty:
        good = sweep.sort_values(["pf", "return%"], ascending=False)
    best = good.iloc[0].to_dict()
    print("\nTop 10 adaptive configs (selection <= %s):" % args.select_end)
    print(good.head(10)[["threshold", "margin", "cooldown", "cap", "direction", "tp_scale",
                         "sl_scale", "atr_weight", "trades", "return%", "win%", "pf", "maxdd%"]].to_string(index=False))

    # ---------- Strict-forward adaptive test Jan-Jun 2026 ----------
    start, end = args.start, args.end
    preds_test = _mask(preds, start, end)
    df_test = df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    lv = build_dynamic_levels(preds_test, df_test, best["tp_scale"], best["sl_scale"],
                              best["atr_weight"], min_tp=10, max_tp=100,
                              min_sl=8, max_sl=80)
    rr_ok = lv["rr_quality"].to_numpy(dtype=float) >= best["rr_min"]
    lv.loc[~rr_ok, ["prob_up", "prob_down"]] = 0.0
    p_test = lv.copy()
    if best["direction"] == "long":
        p_test["prob_down"] = 0.0
    elif best["direction"] == "short":
        p_test["prob_up"] = 0.0
    c2 = Config()
    c2.probability_threshold = best["threshold"]
    c2.probability_margin = best["margin"]
    c2.cooldown_bars = best["cooldown"]
    c2.notional_pct_cap = best["cap"]
    bt_adapt = run_backtest_dynamic(df_test, p_test, c2)
    print("\n===== ADAPTIVE Jan-Jun 2026 =====")
    print(bt_adapt.metrics)

    # ---------- Fixed Config A and B (same signal/model, fixed TP/SL) ----------
    fixed_results = {}
    for name, c in {
        "A_fixed": dict(threshold=0.60, margin=0.0, cooldown=6, cap=1.0, direction="long"),
        "B_fixed": dict(threshold=0.70, margin=0.0, cooldown=6, cap=3.0, direction="long"),
    }.items():
        # Fixed strategies use the same confidence predictions but cfg.TP/SL.
        lv_fixed = preds_test.copy()
        if c["direction"] == "long":
            lv_fixed["prob_down"] = 0.0
        elif c["direction"] == "short":
            lv_fixed["prob_up"] = 0.0
        c3 = Config()
        c3.tp_bps = 50.0
        c3.sl_bps = 20.0
        c3.probability_threshold = c["threshold"]
        c3.probability_margin = c["margin"]
        c3.cooldown_bars = c["cooldown"]
        c3.notional_pct_cap = c["cap"]
        fb = run_backtest(df_test, lv_fixed, c3)
        fixed_results[name] = fb.metrics
        print(f"\n===== {name} Jan-Jun 2026 (fixed 50/20) =====")
        print(fb.metrics)

    # ---------- Reports ----------
    report = {
        "source": src,
        "config": cfg.as_dict(),
        "model": wf.aggregate,
        "selected_adaptive_config": best,
        "adaptive_2026H1": bt_adapt.metrics,
        "fixed_2026H1": fixed_results,
        "selection_cutoff": args.select_end,
        "period": {"start": start, "end": end},
    }
    (cfg.output_dir / "adaptive_2026H1.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # HTML comparison report (simple, no heavy styling dependency)
    from btc_scalper.evaluate import evaluate_backtest  # noqa: E402

    class _FakeWF:
        predictions = p_test
        metrics = [{"fold": 0, "test_start": start, "test_end": end, "test_rows": len(p_test),
                    "base_rate_long": 0, "base_rate_short": 0, "auc_long": 0, "auc_short": 0,
                    "acc_long": 0, "acc_short": 0}]
        trained_folds = cfg.folds
        feature_importance = wf.feature_importance
        aggregate = wf.aggregate

    out_dir = cfg.output_dir / "adaptive_2026H1"
    evaluate_backtest(bt_adapt, _FakeWF(), c2, out_dir, source=src)

    # Append fixed comparison into the same report dir as a simple md/json.
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nSaved: {cfg.output_dir / 'adaptive_sweep.csv'}")
    print(f"Saved: {cfg.output_dir / 'adaptive_2026H1.json'}")
    print(f"Saved: {out_dir / 'report.html'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
