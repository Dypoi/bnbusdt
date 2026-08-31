"""Evaluation, HTML report generation and plotting helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .model import WalkForwardResult


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        if abs(x) >= 1e6:
            return f"{x:,.0f}"
        if abs(x) >= 1e3:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    return str(x)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    html = ["<table class='data'><thead><tr>"]
    html += [f"<th>{h}</th>" for h in headers]
    html += ["</tr></thead><tbody>"]
    for row in rows:
        html.append("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>")
    html += ["</tbody></table>"]
    return "".join(html)


def evaluate_backtest(
    btr: BacktestResult,
    wf: WalkForwardResult,
    cfg,
    output_dir: Path,
    source: str | None = None,
    return_metrics: bool = True,
) -> dict:
    """Combine ML and backtest results into a markdown report + equity chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style = """
    <style>
      body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#f7f8fa;color:#1c2733}
      h1{font-size:22px} h2{font-size:18px;margin-top:28px;color:#0f172a}
      .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}
      .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;box-shadow:0 1px 2px rgba(15,23,42,.05)}
      .card .k{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#64748b}
      .card .v{font-size:24px;font-weight:700;margin-top:4px}
      table.data{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:10px 0 18px}
      table.data th,table.data td{border:1px solid #e2e8f0;padding:8px 10px;text-align:left}
      table.data th{background:#f1f5f9}
      .note{background:#eef2ff;border-left:4px solid #6366f1;padding:12px 16px;border-radius:6px;font-size:14px}
      img{max-width:100%;border:1px solid #e2e8f0;border-radius:10px}
    </style>
    """

    # ---------- equity curve plot ----------
    plot_path = output_dir / "equity_curve.png"
    _plot_equity(btr, cfg, plot_path)

    # ---------- report ----------
    m = btr.metrics
    a = wf.aggregate
    src_note = _data_source_note(cfg, wf, source)

    cards = (
        _card("Total return", f"{m['total_return_pct']:.2f}%")
        + _card("Final equity", f"{m['final_equity']:,.0f}")
        + _card("Trades", f"{m['n_trades']}")
        + _card("Win rate", f"{m['win_rate']:.1f}%")
        + _card("Profit factor", _fmt(round(m['profit_factor'], 2)))
        + _card("Max drawdown", f"{m['max_drawdown']:.2f}%")
        + _card("Sharpe", f"{m['sharpe']:.2f}")
        + _card("Mean AUC", f"{a['mean_auc']:.4f}" if a.get("mean_auc") else "—")
    )

    # Supports both single-label (label_up) and dual long/short scalp models.
    dual = any("auc_long" in r for r in wf.metrics)
    if dual:
        fold_rows = [
            [
                r["fold"],
                r["test_start"],
                r["test_end"],
                r["test_rows"],
                f"{r['base_rate_long']*100:.1f}%",
                f"{r['base_rate_short']*100:.1f}%",
                f"{r['auc_long']:.4f}",
                f"{r['auc_short']:.4f}",
                f"{r['acc_long']*100:.1f}%",
                f"{r['acc_short']*100:.1f}%",
            ]
            for r in wf.metrics
        ]
        fold_table = _table(
            [
                "Fold", "Test start (UTC)", "Test end (UTC)", "Rows",
                "Base L", "Base S", "AUC L", "AUC S", "Acc L", "Acc S",
            ],
            fold_rows,
        )
    else:
        fold_rows = [
            [
                r["fold"],
                r["test_start"],
                r["test_end"],
                r["test_rows"],
                "%" if r["base_rate"] is None else f"{r['base_rate']*100:.1f}%",
                f"{r['auc']:.4f}" if r.get("auc") is not None else "n/a",
                f"{r['acc']*100:.1f}%",
                f"{r['prec@10%']*100:.1f}%" if r.get("prec@10%") is not None else "n/a",
            ]
            for r in wf.metrics
        ]
        fold_table = _table(
            ["Fold", "Test start (UTC)", "Test end (UTC)", "Rows", "Base rate", "AUC", "Acc", "Prec@10%"],
            fold_rows,
        )

    imp = wf.feature_importance.head(20)
    imp_rows = [[i + 1, r["feature"], f"{r['gain']:,.0f}"] for i, r in imp.iterrows()]
    imp_table = _table(["#", "Feature", "Gain"], imp_rows)

    exit_counts = m.get("exit_type_counts", {})
    exit_rows = [[k, v] for k, v in exit_counts.items()]
    exit_table = _table(["Exit type", "Count"], exit_rows)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{cfg.symbol} {cfg.interval} scalping report</title>{style}</head>
<body>
<h1>{cfg.symbol} · {cfg.interval} · Scalping research report</h1>
<div class="note">{src_note}</div>
<h2>Backtest summary</h2>
<div class="cards">{cards}</div>
<h2>Equity curve</h2>
<img src="equity_curve.png" alt="equity curve">
<h2>Walk-forward folds</h2>
{fold_table}
<h2>Exit type</h2>
{exit_table}
<h2>Top 20 features</h2>
{imp_table}
<h2>Config</h2>
<pre>{json.dumps(cfg.as_dict()["lgb_params"], indent=2)}</pre>
<p>TP: {cfg.tp_bps} bps · SL: {cfg.sl_bps} bps · Max hold: {cfg.max_hold_bars} bars ·
fee: {cfg.taker_fee_bps} bps/side · slippage: {cfg.slippage_bps} bps/side ·
threshold: {cfg.probability_threshold} · margin: {cfg.probability_margin} ·
cooldown: {cfg.cooldown_bars} bars · risk: {cfg.risk_pct*100:.0f}% equity/trade ·
capital: {cfg.initial_capital:,.0f}</p>
</body></html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    # ---------- machine-readable summary ----------
    summary = {
        "config": cfg.as_dict(),
        "model": a,
        "backtest": m,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if return_metrics:
        return btr.metrics
    return summary


def _data_source_note(cfg, wf, source: str | None = None) -> str:
    n_rows = wf.predictions.shape[0]
    src = source or f"real:{cfg.symbol}"
    demo = "<b>DEMO</b> (same Binance Vision format, not the requested pair)" if src.startswith("demo:") else "real"
    return (
        f"Out-of-sample model predictions: <b>{n_rows:,}</b> bars. "
        f"Walk-forward folds: <b>{wf.trained_folds}</b>. "
        f"Dataset source: <code>{src}</code> ({demo}). "
        "This is a research harness for 5-minute scalping, not financial advice."
    )


def _card(k, v):
    return f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>"


def _plot_equity(btr: BacktestResult, cfg, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if btr.equity.empty:
        return
    eq = btr.equity.set_index("open_time")["equity"]
    dd = eq / eq.cummax() - 1.0

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, dpi=cfg.figure_dpi
    )
    axes[0].plot(eq.index, eq.values, color="#16a34a", lw=1.2)
    axes[0].axhline(cfg.initial_capital, color="#94a3b8", ls="--", lw=0.9)
    axes[0].set_ylabel("Equity (USDT)")
    axes[0].set_title(f"{cfg.symbol} {cfg.interval} scalping — walk-forward out-of-sample equity")
    axes[0].grid(alpha=0.3)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[1].plot(dd.index, dd.values * 100.0, color="#dc2626", lw=1.0)
    axes[1].fill_between(dd.index, dd.values * 100.0, 0, color="#dc2626", alpha=0.2)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(df: pd.DataFrame, preds: pd.DataFrame, cfg, path: Path) -> None:
    """Optional: plot a sliced window with model probabilities over price."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joined = preds[["open_time", "prob_up", "prob_down"]].set_index("open_time")
    joined = joined.join(df[["close"]], how="left").dropna(subset=["close"]).sort_index()
    if len(joined) < 50:
        return
    joined = joined.iloc[-2000:]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=cfg.figure_dpi)
    ax.plot(joined.index, joined["close"], color="#0f172a", lw=1.0, label="Close")
    ax2 = ax.twinx()
    ax2.plot(joined.index, joined["prob_up"], color="#16a34a", lw=0.9, alpha=0.7, label="P(up)")
    ax2.plot(joined.index, joined["prob_down"], color="#dc2626", lw=0.9, alpha=0.7, label="P(down)")
    ax2.axhline(0.5, color="#94a3b8", ls=":", lw=0.8)
    ax2.set_ylabel("Model probability")
    ax2.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title(f"{cfg.symbol} {cfg.interval} — last out-of-sample predictions")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
