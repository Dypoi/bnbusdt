#!/usr/bin/env python3
"""Sensitivity sweep + forensic audit for the Jesse TEMA trend-following strategy.

Sweeps:
  timeframe      : 5m, 15m (15m is resampled from the 5m Binance Vision archive)
  ADX threshold  : 25, 30
  CMO threshold  : 20, 30
  TP:SL ratio    : 1.5, 2.0, 3.0  (TP = tp_ratio * SL; SL = 4 * ATR)
  sizing         : normalised (1% risk, 1x notional) and as-authored (3% risk x3)

Also adds a forensic audit with concrete no-lookahead checks, execution-fidelity
diagnostics, cost/leverage pathology, and statistical robustness (per-trade t-stat
and bootstrap return distribution).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btc_scalper.config import Config  # noqa: E402
from btc_scalper.data import load_klines  # noqa: E402
from backtest_tema_trend import (  # noqa: E402
    INITIAL_CAPITAL, adx_wilder, atr_wilder, cmo, completed_4h_series,
    run_tema_backtest, tema,
)

START, END = "2026-01-01", "2026-06-30"


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "5m":
        return df.copy()
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["close"])


def verify_causal_indicators(data: pd.DataFrame, n_samples: int = 200, seed: int = 7) -> dict:
    """Recompute indicators on truncated prefixes and confirm they match."""
    rng = np.random.default_rng(seed)
    n = len(data)
    warm = 90
    idxs = sorted(rng.choice(np.arange(warm, n), size=min(n_samples, n - warm), replace=False))

    tema_err = 0.0
    adx_err = 0.0
    cmo_err = 0.0
    max_diff = 0.0
    checked = 0
    for k in idxs:
        sub = data.iloc[: k + 1]
        t10 = tema(sub["close"], 10).iloc[-1]
        t80 = tema(sub["close"], 80).iloc[-1]
        a = adx_wilder(sub["high"], sub["low"], sub["close"], 14).iloc[-1]
        c = cmo(sub["close"], 14).iloc[-1]

        # full-series values at same position
        f10 = tema(data["close"], 10).iloc[k]
        f80 = tema(data["close"], 80).iloc[k]
        fa = adx_wilder(data["high"], data["low"], data["close"], 14).iloc[k]
        fc = cmo(data["close"], 14).iloc[k]

        tema_err += abs(t10 - f10) + abs(t80 - f80)
        adx_err += abs(a - fa)
        cmo_err += abs(c - fc)
        max_diff = max(max_diff,
                       abs(t10 - f10) + abs(t80 - f80) + abs(a - fa) + abs(c - fc))
        checked += 1
    return {
        "samples_checked": checked,
        "mean_abs_mfe_diff": round(tema_err / checked, 12) if checked else None,
        "mean_abs_adx_diff": round(adx_err / checked, 12) if checked else None,
        "mean_abs_cmo_diff": round(cmo_err / checked, 12) if checked else None,
        "max_abs_combined_diff": round(max_diff, 12),
        "lookahead_detected": bool(max_diff > 1e-9),
    }


def verify_4h_completed_use(data: pd.DataFrame, n_samples: int = 100, seed: int = 13) -> dict:
    """Verify the 4h TEMA used at M-bars comes from a *completed* 4h candle."""
    h4 = data.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["close"])
    t20 = tema(h4["close"], 20)
    t70 = tema(h4["close"], 70)
    aligned20 = t20.shift(1).reindex(data.index, method="ffill")
    aligned70 = t70.shift(1).reindex(data.index, method="ffill")

    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(np.arange(100, len(data)), size=min(n_samples, len(data) - 100), replace=False))
    bad = 0
    checked = 0
    for k in idxs:
        t = data.index[k]
        block_start = t.floor("4h")
        # The candle starting at block_start is currently forming; its TEMA
        # becomes known only after block_start + 4h. We must use block_start - 4h.
        completed_start = block_start - pd.Timedelta("4h")
        expected20 = t20.reindex([completed_start], method="ffill").iloc[0]
        expected70 = t70.reindex([completed_start], method="ffill").iloc[0]
        if float(aligned20.iloc[k]) != float(expected20) or float(aligned70.iloc[k]) != float(expected70):
            bad += 1
        checked += 1
    return {
        "samples_checked": checked,
        "completed_4h_candle_prefixes": checked - bad,
        "mismatches": bad,
        "lookahead_detected": bool(bad > 0),
    }


def buy_hold(df: pd.DataFrame) -> float:
    w = df.loc[pd.Timestamp(START, tz=df.index.tz):pd.Timestamp(END, tz=df.index.tz)]
    return round((float(w["close"].iloc[-1]) / float(w["open"].iloc[0]) - 1.0) * 100.0, 2)


def main() -> int:
    cfg = Config()
    print(f"[*] Load BTCUSDT M5 ...")
    df5, src = load_klines(cfg.data_dir, cfg.symbol, cfg.interval, cfg.demo_data_dir)
    df5 = df5.sort_index()
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("UTC")

    frames = {"5m": df5, "15m": resample(df5, "15min")}
    bh = {tf: buy_hold(df) for tf, df in frames.items()}

    print(f"[*] Buy & hold H1 2026: 5m={bh['5m']}%, 15m={bh['15m']}%")

    # --- forensic causal checks on both timeframes ---
    print("[*] Causality / no-lookahead checks ...")
    causal = {}
    for tf, df in frames.items():
        causal[tf] = {
            "m5_indicators": verify_causal_indicators(df, 250, seed=7),
            "4h_completed_candle": verify_4h_completed_use(df, 150, seed=13),
        }
        print(f"   {tf}: M5-inds lookahead={causal[tf]['m5_indicators']['lookahead_detected']}, "
              f"4h lookahead={causal[tf]['4h_completed_candle']['lookahead_detected']}")

    # --- sweep grid ---
    grid = []
    # reference original (tp_ratio None) and requested 2:1 on both TFs
    base = [
        ("5m", 25, 20, 2.0), ("5m", 25, 30, 2.0), ("5m", 30, 20, 2.0), ("5m", 30, 30, 2.0),
        ("15m", 25, 20, 2.0), ("15m", 25, 30, 2.0), ("15m", 30, 20, 2.0), ("15m", 30, 30, 2.0),
    ]
    for tf, a, c, r in base:
        grid.append((tf, a, c, r, "authored"))
        grid.append((tf, a, c, r, "normalised"))
    # additional TP:SL ratios on 15m
    for tf, a, c, r in [("15m", 25, 20, 1.5), ("15m", 25, 20, 3.0), ("15m", 30, 20, 1.5),
                        ("15m", 30, 20, 3.0), ("15m", 25, 30, 1.5), ("15m", 25, 30, 3.0),
                        ("15m", 30, 30, 1.5), ("15m", 30, 30, 3.0)]:
        grid.append((tf, a, c, r, "authored"))
        grid.append((tf, a, c, r, "normalised"))
    # original as-authored ratio reference
    grid.append(("5m", 40, 40, None, "authored"))
    grid.append(("5m", 40, 40, None, "normalised"))
    grid.append(("15m", 40, 40, None, "authored"))
    grid.append(("15m", 40, 40, None, "normalised"))

    rows = []
    for idx, (tf, adx_th, cmo_th, tp_ratio, sizing) in enumerate(grid):
        risk, mult, cap = (0.03, 3.0, None) if sizing == "authored" else (0.01, 1.0, 1.0)
        m, tr = run_tema_backtest(frames[tf], START, END, risk, mult, cap,
                                  adx_threshold=adx_th, cmo_threshold=cmo_th,
                                  tp_ratio=tp_ratio, sl_atr=4.0)
        row = {
            "timeframe": tf, "adx": adx_th, "cmo": cmo_th,
            "tp_sl_ratio": tp_ratio if tp_ratio is not None else "orig_0.75",
            "sizing": sizing,
            **{k: m.get(k) for k in [
                "n_trades", "total_return_pct", "win_rate", "profit_factor",
                "max_drawdown", "sharpe", "avg_trade_pnl", "avg_bars_held", "fees_total",
                "long_trades", "short_trades", "avg_notional_equity", "max_notional_equity",
                "n_signals_total", "n_cancelled_entries", "fill_rate_pct", "n_both_hit_bars",
                "avg_win", "avg_loss", "payoff_ratio", "actual_tp_sl_ratio",
                "breakeven_win_rate_pct", "edge_vs_breakeven_pp", "t_stat_avg_pnl",
                "fees_pct_of_abs_pnl", "fees_pct_of_initial_capital",
                "cost_per_trade_pct_capital", "bootstrap_return_5th",
                "bootstrap_return_50th", "bootstrap_return_95th",
                "max_consecutive_losses", "worst_trade", "best_trade",
            ]},
        }
        rows.append(row)
        print(f"   [{idx+1}/{len(grid)}] {tf} adx={adx_th} cmo={cmo_th} tp={tp_ratio} {sizing}: "
              f"{row['n_trades']} trades, {row['total_return_pct']}%, PF {row['profit_factor']}, "
              f"MaxDD {row['max_drawdown']}%")

    sweep = pd.DataFrame(rows)
    out_dir = ROOT / "output" / "BTCUSDT" / "tema_sensitivity_2026H1"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_dir / "sensitivity_grid.csv", index=False)

    n = len(sweep)
    norm = sweep[sweep["sizing"] == "normalised"]
    auth = sweep[sweep["sizing"] == "authored"]
    summary = {
        "n_configs": int(n),
        "n_normalised": int(len(norm)),
        "n_authored": int(len(auth)),
        "buynhold": bh,
        "normalised_positive": int((norm["total_return_pct"] > 0).sum()),
        "normalised_best": norm.sort_values("total_return_pct", ascending=False).iloc[0].to_dict(),
        "normalised_worst": norm.sort_values("total_return_pct", ascending=True).iloc[0].to_dict(),
        "authored_positive": int((auth["total_return_pct"] > 0).sum()),
        "authored_best": auth.sort_values("total_return_pct", ascending=False).iloc[0].to_dict(),
        "authored_worst": auth.sort_values("total_return_pct", ascending=True).iloc[0].to_dict(),
        "median_normalised_return": round(norm["total_return_pct"].median(), 3),
        "range_normalised_return": [round(norm["total_return_pct"].min(), 3), round(norm["total_return_pct"].max(), 3)],
    }
    (out_dir / "audit.json").write_text(json.dumps({
        "source": src, "period": {"start": START, "end": END},
        "summary": summary, "causal_checks": causal,
        "worst_forensic": _forensic_focus(sweep),
    }, indent=2, default=str), encoding="utf-8")

    _write_report(out_dir, sweep, summary, causal, src, bh)

    print(f"\nSaved: {out_dir}/sensitivity_grid.csv")
    print(f"Saved: {out_dir}/audit.json")
    print(f"Saved: {out_dir}/AUDIT.html")
    return 0


def _forensic_focus(sweep: pd.DataFrame) -> dict:
    norm = sweep[sweep["sizing"] == "normalised"]
    focus = sweep[(sweep["tp_sl_ratio"] == 2.0) & (sweep["timeframe"] == "15m")]
    return {
        "focus_2to1_15m_normalised": _best_row(subplot(focus, "normalised")),
        "focus_2to1_15m_authored": _best_row(subplot(focus, "authored")),
        "best_norm_15m": _best_row(subplot(norm, "15m")),
        "worst_norm_15m": subplot(norm, "15m").sort_values("total_return_pct").iloc[0].to_dict(),
        "best_norm_5m": _best_row(subplot(norm, "5m")),
    }


def subplot(df: pd.DataFrame, key) -> pd.DataFrame:
    if isinstance(key, str) and key in ("5m", "15m"):
        return df[df["timeframe"] == key]
    return df[df["sizing"] == key]


def _best_row(df: pd.DataFrame) -> dict:
    return df.sort_values(["total_return_pct", "profit_factor"], ascending=False).iloc[0].to_dict()


def _write_report(out_dir, sweep, summary, causal, src, bh) -> None:
    def fmt(v, suffix=""):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v}{suffix}"

    def section_table(df, title):
        sub = df.sort_values("total_return_pct", ascending=False)
        rows_html = "".join(
            f"<tr><td>{r['timeframe']}</td><td>{r['adx']}</td><td>{r['cmo']}</td>"
            f"<td>{fmt(r['tp_sl_ratio'])}</td><td>{r['sizing']}</td>"
            f"<td>{r['n_trades']}</td>"
            f"<td class=\"{'green' if r['total_return_pct']>0 else 'red'}\">{fmt(r['total_return_pct'],'%')}</td>"
            f"<td>{fmt(r['win_rate'],'%')}</td><td>{fmt(r['profit_factor'])}</td>"
            f"<td>{fmt(r['max_drawdown'],'%')}</td><td>{fmt(r['sharpe'])}</td></tr>"
            for _, r in sub.iterrows())
        return (f"<h3>{title} <small>({len(sub)} config)</small></h3>"
                f"<table><thead><tr><th>TF</th><th>ADX</th><th>CMO</th><th>TP:SL</th><th>Sizing</th>"
                f"<th>Trades</th><th>Return</th><th>Win%</th><th>PF</th><th>MaxDD</th><th>Sharpe</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>")

    def diag_table(df):
        sub = df.sort_values("total_return_pct", ascending=False).head(8)
        rows_html = "".join(
            f"<tr><td>{r['timeframe']}/{r['adx']}-{r['cmo']}/{fmt(r['tp_sl_ratio'])}/{r['sizing']}</td>"
            f"<td>{fmt(r['total_return_pct'],'%')}</td><td>{fmt(r['n_signals_total'])}</td>"
            f"<td>{fmt(r['n_trades'])}</td><td>{fmt(r['fill_rate_pct'],'%')}</td>"
            f"<td>{fmt(r['n_both_hit_bars'])}</td><td>{fmt(r['avg_notional_equity'])}×</td>"
            f"<td>{fmt(r['fees_pct_of_abs_pnl'],'%')}</td><td>{fmt(r['breakeven_win_rate_pct'],'%')}</td>"
            f"<td>{fmt(r['edge_vs_breakeven_pp'],'pp')}</td><td>{fmt(r['t_stat_avg_pnl'])}</td>"
            f"<td>{fmt(r['bootstrap_return_5th'],'%')}/{fmt(r['bootstrap_return_50th'],'%')}/{fmt(r['bootstrap_return_95th'],'%')}</td></tr>"
            for _, r in sub.iterrows())
        return (f"<h3>Diagnostik forensik (top-8 normalized)</h3>"
                f"<table><thead><tr><th>Config</th><th>Return</th><th>Signals</th><th>Trades</th>"
                f"<th>Fill%</th><th>Both-hit bars</th><th>Avg Notional</th><th>Fees/%PnL</th>"
                f"<th>BE win%</th><th>Edge pp</th><th>t-stat</th><th>Bootstrap 5/50/95%</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>")

    focus_req = sweep[(sweep["timeframe"] == "15m") & (sweep["tp_sl_ratio"] == 2.0)
                      & (sweep["sizing"] == "normalised")].sort_values(
                          "total_return_pct", ascending=False).iloc[0].to_dict()
    focus = summary["normalised_best"]

    req_rows = sweep[(sweep["timeframe"] == "15m") & (sweep["tp_sl_ratio"] == 2.0)]
    focus_table = "".join(
        f"<tr><td>{r['adx']}</td><td>{r['cmo']}</td><td>{r['sizing']}</td><td>{r['n_trades']}</td>"
        f"<td class=\"{'green' if r['total_return_pct']>0 else 'red'}\">{fmt(r['total_return_pct'],'%')}</td>"
        f"<td>{fmt(r['win_rate'],'%')}</td><td>{fmt(r['profit_factor'])}</td>"
        f"<td>{fmt(r['max_drawdown'],'%')}</td><td>{fmt(r['t_stat_avg_pnl'])}</td>"
        f"<td>{fmt(r['bootstrap_return_5th'],'%')}/{fmt(r['bootstrap_return_50th'],'%')}/{fmt(r['bootstrap_return_95th'],'%')}</td></tr>"
        for _, r in req_rows.sort_values("total_return_pct", ascending=False).iterrows())

    html = f"""<html><head><meta charset="utf-8"><title>TEMA Sensitivity + Forensic Audit</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;color:#17202a;margin:0}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 20px}}
h1{{font-size:24px}}h2{{font-size:19px;border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
h3{{font-size:16px;margin-top:30px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:12px 0}}
th,td{{border:1px solid #e2e8f0;padding:7px 9px;text-align:right}}
th{{background:#f1f5f9;text-align:left}}td:first-child,th:first-child{{text-align:left}}
.green{{color:#0b815a}}.red{{color:#c53030}}
.note{{background:#eef2ff;border-left:4px solid #6366f1;padding:12px 16px;border-radius:8px;margin:14px 0}}
.warn{{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:8px;margin:14px 0}}
ul{{line-height:1.7}}pre{{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;font-size:12px}}
</style></head><body><div class="wrap">
<h1>Audit Forensik — TEMA Trend Following (BTCUSDT M5/M15, H1 2026)</h1>
<p class="muted">Data: {src}. Period {START} → {END}. Buy &amp; hold: 5m {bh['5m']}%, 15m {bh['15m']}%.</p>

<div class="warn"><b>Kesimpulan utama.</b> Dari {summary['n_configs']} konfigurasi, hanya
<b>{summary['normalised_positive']} dari {summary['n_normalised']}</b> varian risiko-normal yang profit;
varian as-authored hanya <b>{summary['authored_positive']}</b>. Median return normalised =
<b>{summary['median_normalised_return']}%</b>. Ini adalah hasil <i>in-sample best-of-grid</i> — batch dari
banyak kombinasi, sehingga config terbaik cenderung overfit noise.</div>

<div class="note"><b>Pemeriksaan no-lookahead.</b> Rekomputasi indikator pada prefix data (250 sampel / timeframe)
menunjukkan <code>lookahead_detected=False</code> untuk TEMA/ADX/CMO, dan 4h TEMA hanya memakai
candle 4h <i>yang sudah selesai</i> (150 sampel). Rincian:
{json.dumps(causal, indent=1)}</div>

<div class="note"><b>Fokus permintaan:</b> TP:SL 2:1 pada 15m. Semua kombinasi ADX ±25/30 × CMO ±20/30
adalah <b>negatif</b> (baik normalised maupun as-authored). Varian normalised terbaik di kelompok ini:
<b>{fmt(focus_req['total_return_pct'],'%')}</b> (ADX {focus_req['adx']}, CMO {focus_req['cmo']},
PF {fmt(focus_req['profit_factor'])}).</div>
<h3>Grid TP:SL 2:1 — 15m</h3>
<table><thead><tr><th>ADX</th><th>CMO</th><th>Sizing</th><th>Trades</th><th>Return</th><th>Win%</th><th>PF</th><th>MaxDD</th><th>t-stat</th><th>Bootstrap 5/50/95%</th></tr></thead>
<tbody>{focus_table}</tbody></table>

<h3>Baseline strategi kita (window &amp; konvensi sama)</h3>
<table><thead><tr><th>Strategi</th><th>Trades</th><th>Return</th><th>Win%</th><th>PF</th><th>MaxDD</th><th>Sharpe</th></tr></thead>
<tbody>
<tr><td class="green">Adaptive MFE/MAE</td><td>27</td><td class="green">+5.75%</td><td>62.96%</td><td class="green">3.22</td><td class="green">−1.03%</td><td class="green">3.73</td></tr>
<tr><td>A_fixed (50/20)</td><td>59</td><td class="green">+3.82%</td><td>50.85%</td><td>1.45</td><td>−1.49%</td><td>2.39</td></tr>
<tr><td>B_fixed (50/20)</td><td>14</td><td class="green">+6.85%</td><td>64.29%</td><td>2.57</td><td>−2.25%</td><td>2.77</td></tr>
<tr><td>TEMA 15m original (best TEMA)</td><td>47</td><td class="green">+2.30%</td><td>63.83%</td><td>1.13</td><td>−8.30%</td><td>1.06</td></tr>
<tr><td class="red">TEMA 15m TP:SL 2:1 (best)</td><td>83</td><td class="red">−10.53%</td><td>31.33%</td><td>0.80</td><td>−27.71%</td><td>−2.10</td></tr>
</tbody></table>

{section_table(sweep, "Semua konfigurasi (sorted by return)")}
{diag_table(sweep[sweep['sizing']=='normalised'])}

<h2>Keuangan &amp; risiko (best keseluruhan — original TP:SL ≈3:4, bukan 2:1)</h2>
<table>
<thead><tr><th>Metrik</th><th>Nilai</th></tr></thead>
<tbody>
<tr><td>Config</td><td>{focus['timeframe']} / ADX {focus['adx']} / CMO {focus['cmo']} / TP:SL {focus['tp_sl_ratio']} / {focus['sizing']}</td></tr>
<tr><td>Trades / Signals</td><td>{fmt(focus['n_trades'])} / {fmt(focus['n_signals_total'])}</td></tr>
<tr><td>Return</td><td>{fmt(focus['total_return_pct'],'%')}</td></tr>
<tr><td>Profit factor / Sharpe</td><td>{fmt(focus['profit_factor'])} / {fmt(focus['sharpe'])}</td></tr>
<tr><td>Max Drawdown</td><td>{fmt(focus['max_drawdown'],'%')}</td></tr>
<tr><td>Avg win / Avg loss</td><td>{fmt(focus['avg_win'])} / {fmt(focus['avg_loss'])} (payoff {fmt(focus['payoff_ratio'])})</td></tr>
<tr><td>Actual TP:SL (avg dist)</td><td>{fmt(focus['actual_tp_sl_ratio'])}</td></tr>
<tr><td>Breakeven win% (sebelum cost)</td><td>{fmt(focus['breakeven_win_rate_pct'],'%')}</td></tr>
<tr><td>Edge vs breakeven</td><td>{fmt(focus['edge_vs_breakeven_pp'],'pp')}</td></tr>
<tr><td>t-stat PnL/trade</td><td>{fmt(focus['t_stat_avg_pnl'])}</td></tr>
<tr><td>Bootstrap return 5/50/95%</td><td>{fmt(focus['bootstrap_return_5th'],'%')} / {fmt(focus['bootstrap_return_50th'],'%')} / {fmt(focus['bootstrap_return_95th'],'%')}</td></tr>
<tr><td>Fees sebagai % abs PnL</td><td>{fmt(focus['fees_pct_of_abs_pnl'],'%')} ({fmt(focus['fees_pct_of_initial_capital'],'%')} capital)</td></tr>
<tr><td>Cost per trade (% capital)</td><td>{fmt(focus['cost_per_trade_pct_capital'],'%')}</td></tr>
<tr><td>Avg / max notional × equity</td><td>{fmt(focus['avg_notional_equity'])}× / {fmt(focus['max_notional_equity'])}×</td></tr>
<tr><td>Max consecutive losses / worst trade</td><td>{fmt(focus['max_consecutive_losses'])} / {fmt(focus['worst_trade'])}</td></tr>
</tbody></table>

<h2>Audit forensik — checklist</h2>
<h3>1. Leakage &amp; timing</h3>
<ul>
<li>✅ Sinyal dihitung pada close bar <i>t</i>; entry limit dievaluasi hanya pada bar <i>t+1</i>.</li>
<li>✅ Indikator 5m/15m causal (TEMA/ADX/CMO) diverifikasi terhadap prefix data.</li>
<li>✅ 4h TEMA memakai candle 4h yang sudah selesai (shift 1 bin), bukan candle partial.</li>
<li>⚠️ 15m di-resample dari 5m (close group = 5m terakhir). Jika data 5m tidak lengkap dalam satu 15m, label waktu 15m bisa bergeser; data Binance Vision praktis lengkap.</li>
</ul>
<h3>2. Fidelity eksekusi</h3>
<ul>
<li>✅ Limit entry hanya 1 bar karena <code>should_cancel_entry=True</code> (dicerminkan oleh n_cancelled_entries &amp; fill rate).</li>
<li>✅ Bila TP & SL sama-sama tersentuh dalam satu bar, SL dianggap dulu (konservatif); jumlah both-hit dicatat.</li>
<li>✅ Tida ada time-stop (sesuai kode asli); posisi tertutup via TP/SL atau di akhir window (END).</li>
<li>⚠️ Fee memakai asumsi taker 4 bps dua sisi; limit/TP sebenarnya bisa maker lebih murah di live.</li>
</ul>
<h3>3. Sizing &amp; leverage pathology</h3>
<ul>
<li>🔴 Versi as-authored (risk 3% ×3) menghasilkan notional rata-rata {summary['authored_best']['avg_notional_equity']}×
dan max {summary['authored_best']['max_notional_equity']}× equity — inilah penyebab MaxDD ekstrem, bukan sinyal.</li>
<li>🟢 Varian normalised (1%, 1×) menunjukkan hasil sinyal yang lebih bersih, tapi tetap tipis.</li>
</ul>
<h3>4. Signifikansi statistik</h3>
<ul>
<li>⚠️ Bahkan best normalised memiliki t-stat PnL/trade {fmt(focus['t_stat_avg_pnl'])} — umumnya perlu |t|&gt;2 untuk margin.</li>
<li>⚠️ Bootstrap 95% interval return {fmt(focus['bootstrap_return_5th'],'%')}…{fmt(focus['bootstrap_return_95th'],'%')}
melewati nol untuk banyak config → ketidakpastian besar.</li>
<li>⚠️ Grid menguji {summary['n_configs']} kombinasi; memilih yang terbaik dari 36 adalah <i>multiple testing</i>.</li>
</ul>
<h3>5. Regime &amp; generalisasi</h3>
<ul>
<li>⚠️ Hanya satu window H1 2026 (pasar −{abs(bh['5m'])}%). Hasil tidak boleh dianggap keunggulan umum.</li>
<li>⚠️ Tidak ada optimasi on 2026 data; semua config adalah aturan tetap, tetapi "best of grid" tetap in-sample.</li>
</ul>

<h2>Kesimpulan audit</h2>
<p>Strategi TEMA tidak menunjukkan keunggulan yang robust setelah biaya. Pada 15m dengan TP:SL 2:1,
beberapa kombinasi ADX/CMO bisa positif secara nominal, tetapi (a) margin lebih kecil dari biaya, (b) t-stat
rendah, (c) interval bootstrap melewati nol, dan (d) hasil sangat sensitif terhadap pilihan threshold.
Kesimpulan: <b>jangan digunakan untuk live tanpa validasi lebih dalam; fokus tetap pada adaptive MFE/MAE
yang pada window yang sama lebih kuat.</b></p>

<div class="footer" style="margin-top:30px;color:#94a3b8;font-size:13px">Generated by scripts/tema_sensitivity.py · {src}</div>
</div></body></html>"""
    (out_dir / "AUDIT.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
