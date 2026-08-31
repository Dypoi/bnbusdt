#!/usr/bin/env python3
"""Replicate the 'tiny TP + huge SL + fixed 0.05 lot' MT5 scalp style.

The MT5 screenshot shown by the user is typical of a "high win-rate" MT5
scalping account:
- TP very small (BTC ~0.31%, XAU ~0.10%, GBP ~0.04%)
- SL placed very far away (BTC ~24.7%, XAU ~24.6%, GBP ~3.0%)
- fixed 0.05 lot on a $50 account
- M5 timeframe
- many trades, mostly Buy/Sell mixed, all shown as TP/green

We cannot know the exact EA, so this script implements the *same mechanics*
on real BTCUSDT M5 data with a common high-frequency RSI mean-reversion
signal (long when RSI<30, short when RSI>70). It then reports:

1. The "pretty" summary (too many TP trades).
2. The honest net result after spread + fee (MT5 CFD spread is only shown
   as a hidden cost; here we model 5bp spread + 2bp fee).
3. What happens if the same strategy uses a sane 20bp stop instead of a
   24.7% "catastrophic" stop.
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


def _rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50).to_numpy()


def _signal(close: np.ndarray, window: int = 14) -> np.ndarray:
    """Mean-reversion RSI signal: long<30, short>70. Returns +1/-1/0."""
    rsi = _rsi(close, window)
    sig = np.zeros(len(close), dtype=int)
    sig[rsi < 30.0] = 1
    sig[rsi > 70.0] = -1
    return sig


def run_sim(df, start, end, tp_bps=31.0, sl_pct=0.247, qty=0.05, initial=50.0,
            max_hold=12, cooldown=2, spread_bps=5.0, fee_bps=2.0,
            leverage=100.0, signal="rsi") -> dict:
    d = df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")].copy()
    if len(d) < 200:
        return {"error": "period too small"}

    close = d["close"].to_numpy(float)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    sig = _signal(close) if signal == "rsi" else _signal(close)

    n = len(d)
    spread = spread_bps / 10_000.0
    fee = fee_bps / 10_000.0
    cash = float(initial)
    pos = None
    next_entry = 0
    trades = []
    eq_curve = []

    for i in range(n - 1):
        # manage position
        if pos is not None:
            entry, side, ei = pos["entry"], pos["side"], pos["entry_i"]
            tp = entry * (1 + tp_bps / 10_000.0) if side > 0 else entry * (1 - tp_bps / 10_000.0)
            sl = entry * (1 - sl_pct) if side > 0 else entry * (1 + sl_pct)
            exit_price = None
            reason = None
            if side > 0:
                if low[i] <= sl:
                    exit_price, reason = sl, "SL"
                elif high[i] >= tp:
                    exit_price, reason = tp, "TP"
            else:
                if high[i] >= sl:
                    exit_price, reason = sl, "SL"
                elif low[i] <= tp:
                    exit_price, reason = tp, "TP"
            if exit_price is None and (i - ei) < max_hold:
                continue
            if exit_price is None:
                exit_price, reason = close[i], "TIME"
            fill = exit_price * (1 - spread) if side > 0 else exit_price * (1 + spread)
            gross = (fill - entry) * qty * side
            costs = qty * entry * fee + qty * fill * fee
            net = gross - costs
            cash += net
            trades.append({
                "entry_time": str(d.index[ei]), "exit_time": str(d.index[i]),
                "side": "buy" if side > 0 else "sell",
                "entry": entry, "exit": fill, "reason": reason,
                "pnl_gross": gross, "costs": costs, "pnl_net": net,
                "margin": qty * entry / leverage,
            })
            next_entry = i + cooldown
            pos = None

        # open
        if pos is None and i >= next_entry and sig[i] != 0:
            side = int(sig[i])
            raw = close[i + 1]
            entry = raw * (1 + spread) if side > 0 else raw * (1 - spread)
            margin = qty * entry / leverage
            if margin > cash:
                continue
            pos = {"entry": entry, "side": side, "entry_i": i + 1}

        eq_curve.append({"t": str(d.index[i]), "eq": cash})
        if pos is not None:
            mark = close[i]
            eq_curve[-1]["eq"] += (mark - pos["entry"]) * qty * pos["side"]

    if pos is not None:
        entry, side, ei = pos["entry"], pos["side"], pos["entry_i"]
        fill = close[n - 1] * (1 - spread) if side > 0 else close[n - 1] * (1 + spread)
        gross = (fill - entry) * qty * side
        costs = qty * entry * fee + qty * fill * fee
        net = gross - costs
        cash += net
        trades.append({"entry_time": str(d.index[ei]), "exit_time": str(d.index[n - 1]),
                       "side": "buy" if side > 0 else "sell", "entry": entry, "exit": fill,
                       "reason": "TIME", "pnl_gross": gross, "costs": costs, "pnl_net": net,
                       "margin": qty * entry / leverage})

    tr = pd.DataFrame(trades)
    eq = pd.Series({e["t"]: e["eq"] for e in eq_curve}).sort_index()
    if tr.empty:
        return {"empty": True}

    wins = tr[tr.pnl_net > 0]
    losses = tr[tr.pnl_net < 0]
    peak = eq.cummax()
    dd = float(((eq / peak - 1.0) * 100.0).min()) if len(eq) else 0.0
    worst = tr.sort_values("pnl_net").head(5)
    return {
        "period": {"start": start, "end": end},
        "bars": int(n),
        "initial": initial,
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float((eq.iloc[-1] / initial - 1.0) * 100.0),
        "trades": int(len(tr)),
        "win_rate_pct": float((tr.pnl_net > 0).mean() * 100.0),
        "profit_factor": float(wins.pnl_net.sum() / abs(losses.pnl_net.sum())) if len(losses) and losses.pnl_net.sum() != 0 else float("inf"),
        "max_drawdown_pct": dd,
        "reasons": tr.reason.value_counts().to_dict(),
        "avg_trade_net": float(tr.pnl_net.mean()),
        "total_costs": float(tr.costs.sum()),
        "eq_gross_net": float(tr.pnl_gross.sum()),
        "blown_account": bool((eq <= 5).any()),
        "avg_margin_per_trade": float(tr.margin.mean()),
        "worst5": worst[["entry_time", "side", "reason", "pnl_net"]].to_dict("records"),
    }


def fmt(r):
    for k in ("initial", "final_equity", "total_return_pct", "win_rate_pct", "profit_factor",
              "max_drawdown_pct", "avg_trade_net", "total_costs", "eq_gross_net", "avg_margin_per_trade"):
        if k == "profit_factor":
            print(f"  {k}: {r.get(k):.3f}")
        elif k == "total_return_pct" or k == "max_drawdown_pct" or k == "win_rate_pct":
            print(f"  {k}: {r.get(k):.2f}%")
        elif k in ("final_equity", "total_costs", "eq_gross_net", "avg_margin_per_trade"):
            print(f"  {k}: {r.get(k):,.2f}")
        elif k == "avg_trade_net":
            print(f"  {k}: {r.get(k):,.2f}")
        else:
            print(f"  {k}: {r.get(k)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--tp-bps", type=float, default=31.0)
    p.add_argument("--sl-pct", type=float, default=0.247)
    p.add_argument("--qty", type=float, default=0.05)
    p.add_argument("--initial", type=float, default=50.0)
    p.add_argument("--max-hold", type=int, default=12)
    p.add_argument("--cooldown", type=int, default=2)
    p.add_argument("--spread-bps", type=float, default=5.0)
    p.add_argument("--fee-bps", type=float, default=2.0)
    p.add_argument("--leverage", type=float, default=100.0)
    args = p.parse_args(argv)

    cfg = Config()
    df, src = load_klines(cfg.data_dir, cfg.symbol, cfg.interval, cfg.demo_data_dir)
    print(f"Source: {src}, rows={len(df)}, interval=5m\n")

    print(f"=== A) MIMIC screenshot: TP {args.tp_bps}bp, SL {args.sl_pct*100:.1f}% (huge), qty {args.qty} BTC, $50 acct, M5 ===")
    print(f"    spread {args.spread_bps}bp + fee {args.fee_bps}bp/side, max hold {args.max_hold} bars, cooldown {args.cooldown}, lev {args.leverage:.0f}x\n")
    r = run_sim(df, args.start, args.end, tp_bps=args.tp_bps, sl_pct=args.sl_pct, qty=args.qty,
                initial=args.initial, max_hold=args.max_hold, cooldown=args.cooldown,
                spread_bps=args.spread_bps, fee_bps=args.fee_bps, leverage=args.leverage)
    fmt(r)

    print("\n    Worst 5 trades (the ones MT5 screenshot never shows):")
    for w in r.get("worst5", []):
        print(f"      {w}")

    print("\n=== B) Same signal, but SANE stop: SL 20bp (0.20%) ===")
    r2 = run_sim(df, args.start, args.end, tp_bps=args.tp_bps, sl_pct=0.0020, qty=args.qty,
                 initial=args.initial, max_hold=args.max_hold, cooldown=args.cooldown,
                 spread_bps=args.spread_bps, fee_bps=args.fee_bps, leverage=args.leverage)
    fmt(r2)

    print("\n=== C) Same screenshot-style (huge SL), full history 2021-2026 ===")
    r3 = run_sim(df, "2021-01-01", "2026-06-30", tp_bps=args.tp_bps, sl_pct=args.sl_pct, qty=args.qty,
                 initial=args.initial, max_hold=args.max_hold, cooldown=args.cooldown,
                 spread_bps=args.spread_bps, fee_bps=args.fee_bps, leverage=args.leverage)
    fmt(r3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
