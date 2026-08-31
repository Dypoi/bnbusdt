#!/usr/bin/env python3
"""Backtest the user's Jesse 'TemaTrendFollowing' strategy on BTCUSDT M5.

The strategy is replicated as faithfully as possible in a local event-driven
backtest using the same Binance Vision 5m data and the same fee/slippage
conventions used by the existing research backtests.

Model assumptions documented in the code and the report:
- Signal is decided on the close of bar ``t``.
- Entry is a *limit order* one ATR away from the signal close:
  long  ``close - ATR``, short ``close + ATR``.
- ``should_cancel_entry = True`` -> the limit order is only valid on bar
  ``t+1``; if it does not fill during that bar it is cancelled.
- After fill, stop-loss = 4 ATR and take-profit = 3 ATR (both measured with
  the ATR of the *fill* bar, matching ``on_open_position``).
- There is no time-stop in the original strategy, so positions may remain open
  until TP/SL (or are marked closed at the end of the test window).
- Sizing as authored: risk 3% of equity to the stop, then position is tripled
  (``qty*3``). We expose this as the high-multiples variant and also report a
  normalised 1% risk / 1x notional variant for a risk-scale comparison.
- Fees: 4 bps/side (taker, same as existing harness). TP is limit (no
  slippage), SL/END are market exits with 1 bp adverse slippage.
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

FEE = 4.0 / 10_000.0
SLIP = 1.0 / 10_000.0
INITIAL_CAPITAL = 10_000.0


def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return 3.0 * e1 - 3.0 * e2 + e3


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    sm_plus = plus_dm.ewm(alpha=1.0 / period, adjust=False).mean()
    sm_minus = minus_dm.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * sm_plus / atr
    minus_di = 100.0 * sm_minus / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def cmo(series: pd.Series, period: int = 14) -> pd.Series:
    diff = series.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    sm_gain = gain.rolling(period, min_periods=period).sum()
    sm_loss = loss.rolling(period, min_periods=period).sum()
    return 100.0 * (sm_gain - sm_loss) / (sm_gain + sm_loss)


def completed_4h_series(m5: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """TEMA(last-completed 4h close) aligned to M5 bars without lookahead."""
    h4 = m5.resample("4h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna(subset=["close"])
    t20 = tema(h4["close"], period)
    t70 = tema(h4["close"], 70)
    # shift(1): use the 4h candle that has *closed* before the M5 bar.
    return t20.shift(1).reindex(m5.index, method="ffill"), t70.shift(1).reindex(m5.index, method="ffill")


def run_tema_backtest(df: pd.DataFrame, start: str, end: str,
                      risk_pct: float, qty_mult: float, notional_cap: float | None,
                      adx_threshold: float = 40.0, cmo_threshold: float = 40.0,
                      tp_ratio: float | None = None, sl_atr: float = 4.0,
                      ) -> tuple[dict, pd.DataFrame]:
    """Run the TEMA trend-following strategy inside [start, end].

    ``tp_ratio`` = TP_distance / SL_distance. When ``None`` the original
    as-authored ratio is used (TP 3 ATR / SL 4 ATR, i.e. 0.75). When set, SL is
    ``sl_atr * ATR`` and TP is ``tp_ratio * SL``.
    """
    data = df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")].copy()
    if len(data) == 0:
        raise ValueError("No rows in test window")

    tema10 = tema(data["close"], 10)
    tema80 = tema(data["close"], 80)
    tema20_4h, tema70_4h = completed_4h_series(data, 20)
    atr = atr_wilder(data["high"], data["low"], data["close"], 14)
    adx = adx_wilder(data["high"], data["low"], data["close"], 14)
    cmo14 = cmo(data["close"], 14)

    short_trend = np.where(tema10 > tema80, 1, -1)
    long_trend = np.where(tema20_4h > tema70_4h, 1, -1)
    should_long = (short_trend == 1) & (long_trend == 1) & (adx > adx_threshold) & (cmo14 > cmo_threshold)
    should_short = (short_trend == -1) & (long_trend == -1) & (adx > adx_threshold) & (cmo14 < -cmo_threshold)

    idx = data.index.to_numpy()
    open_p = data["open"].to_numpy(float)
    high_p = data["high"].to_numpy(float)
    low_p = data["low"].to_numpy(float)
    close_p = data["close"].to_numpy(float)
    atr_p = atr.to_numpy(float)
    sig_l = should_long.to_numpy(bool)
    sig_s = should_short.to_numpy(bool)

    n = len(data)
    cash = INITIAL_CAPITAL
    trades: list[dict] = []
    next_signal = 0
    n_signals_long = 0
    n_signals_short = 0
    n_signals_skipped = 0
    n_both_hit_bars = 0
    max_notional_equity = 0.0

    i = 0
    while i < n - 1:
        if i < next_signal:
            i += 1
            continue
        long_sig = bool(sig_l[i])
        short_sig = bool(sig_s[i])
        if long_sig:
            n_signals_long += 1
        if short_sig:
            n_signals_short += 1
        if not (long_sig or short_sig):
            i += 1
            continue

        side = 1 if long_sig else -1
        j = i + 1
        if np.isnan(atr_p[i]) or atr_p[i] <= 0:
            i += 1
            continue
        limit = (close_p[i] - atr_p[i]) if side > 0 else (close_p[i] + atr_p[i])
        filled = False
        if side > 0 and low_p[j] <= limit:
            entry_price = float(min(open_p[j], limit))
            filled = True
        elif side < 0 and high_p[j] >= limit:
            entry_price = float(max(open_p[j], limit))
            filled = True
        if not filled:
            i += 1
            continue  # should_cancel_entry=True: order expires after one bar

        if np.isnan(atr_p[j]) or atr_p[j] <= 0:
            i += 1
            continue
        entry_atr = float(atr_p[j])
        if tp_ratio is None:
            sl_dist = 4.0 * entry_atr
            tp_dist = 3.0 * entry_atr
        else:
            sl_dist = sl_atr * entry_atr
            tp_dist = sl_atr * tp_ratio * entry_atr
        if side > 0:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        risk_amount = cash * risk_pct
        qty = risk_amount / max(sl_dist, 1e-12)
        qty *= qty_mult
        if notional_cap is not None:
            qty = min(qty, (cash * notional_cap) / max(entry_price, 1e-12))
        if qty <= 0:
            n_signals_skipped += 1
            i += 1
            continue
        max_notional_equity = max(max_notional_equity, qty * entry_price / max(cash, 1e-12))

        # scan forward for TP/SL (no time-stop)
        exit_j = -1
        exit_type = 0
        exit_price = np.nan
        for b in range(j + 1, n):
            if side > 0:
                both = bool(low_p[b] <= sl_price and high_p[b] >= tp_price)
                n_both_hit_bars += int(both)
                if low_p[b] <= sl_price:
                    exit_j, exit_type, exit_price = b, -1, sl_price
                    break
                if high_p[b] >= tp_price:
                    exit_j, exit_type, exit_price = b, 1, tp_price
                    break
            else:
                both = bool(high_p[b] >= sl_price and low_p[b] <= tp_price)
                n_both_hit_bars += int(both)
                if high_p[b] >= sl_price:
                    exit_j, exit_type, exit_price = b, -1, sl_price
                    break
                if low_p[b] <= tp_price:
                    exit_j, exit_type, exit_price = b, 1, tp_price
                    break
        if exit_j < 0:
            exit_j = n - 1
            exit_type = 9
            exit_price = float(close_p[exit_j])

        if exit_type == -1:
            exit_price = exit_price * (1.0 + SLIP if side > 0 else 1.0 - SLIP)  # market stop, adverse slip
        elif exit_type == 9:
            exit_price = exit_price * (1.0 - SLIP if side > 0 else 1.0 + SLIP)
        # TP is a limit order -> no adverse slippage

        entry_fee = qty * entry_price * FEE
        exit_fee = qty * exit_price * FEE
        gross = (exit_price - entry_price) * qty * side
        net = gross - entry_fee - exit_fee
        cash += net
        trades.append({
            "side": "long" if side > 0 else "short",
            "entry_time": str(pd.Timestamp(idx[j])),
            "exit_time": str(pd.Timestamp(idx[exit_j])),
            "entry_pos": int(j),
            "exit_pos": int(exit_j),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_dist_bps": sl_dist / entry_price * 10_000.0,
            "tp_dist_bps": tp_dist / entry_price * 10_000.0,
            "atr_entry_bps": entry_atr / entry_price * 10_000.0,
            "entry_gap_bps": abs(entry_price - close_p[i]) / close_p[i] * 10_000.0,
            "qty": qty,
            "notional_equity": qty * entry_price / cash,
            "exit_type": "TP" if exit_type == 1 else ("SL" if exit_type == -1 else "END"),
            "pnl_net": net,
            "pnl_pct": net / (qty * entry_price) * 100.0,
            "bars_held": exit_j - j,
            "fees": entry_fee + exit_fee,
        })

        next_signal = exit_j + 1
        i = exit_j + 1

    trades_df = pd.DataFrame(trades)

    # Equity curve marked-to-market per bar (flat between trades).
    equity_vals = np.full(n, INITIAL_CAPITAL)
    unrealized = np.zeros(n)
    if len(trades):
        exit_pos_idx = np.array([t["exit_pos"] for t in trades], dtype=int)
        pnl = np.array([t["pnl_net"] for t in trades])
        pnl_by_bar = np.zeros(n)
        np.add.at(pnl_by_bar, np.clip(exit_pos_idx, 0, n - 1), pnl)
        equity_vals = INITIAL_CAPITAL + np.cumsum(pnl_by_bar)
        for t in trades:
            e_pos = int(t["entry_pos"])
            x_pos = int(t["exit_pos"])
            direction = 1.0 if t["side"] == "long" else -1.0
            seg = np.arange(e_pos, max(e_pos, x_pos))
            unrealized[seg] += (close_p[seg] - t["entry_price"]) * t["qty"] * direction

    equity = pd.DataFrame({"open_time": idx, "equity": equity_vals + unrealized})
    metrics = _metrics(trades_df, equity)
    # Forensic / sensitivity diagnostics (not all are "performance").
    metrics["n_signals_long"] = n_signals_long
    metrics["n_signals_short"] = n_signals_short
    metrics["n_signals_total"] = n_signals_long + n_signals_short
    metrics["n_cancelled_entries"] = metrics["n_signals_total"] - len(trades) - n_signals_skipped
    metrics["fill_rate_pct"] = (len(trades) / metrics["n_signals_total"] * 100.0) if metrics["n_signals_total"] else 0.0
    metrics["n_both_hit_bars"] = n_both_hit_bars
    metrics["max_notional_equity"] = round(max_notional_equity, 3)
    metrics["avg_signals_per_day"] = round(metrics["n_signals_total"] / (len(data) / 288.0), 2)
    if len(trades_df):
        gains = trades_df.loc[trades_df["pnl_net"] > 0, "pnl_net"]
        losses = trades_df.loc[trades_df["pnl_net"] <= 0, "pnl_net"]
        avg_win = float(gains.mean()) if len(gains) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
        metrics["avg_win"] = round(avg_win, 2)
        metrics["avg_loss"] = round(avg_loss, 2)
        metrics["payoff_ratio"] = round(avg_win / abs(avg_loss), 3) if abs(avg_loss) > 0 else None
        metrics["actual_tp_sl_ratio"] = round(float(trades_df["tp_dist_bps"].mean()) / float(trades_df["sl_dist_bps"].mean()), 3) if trades_df["sl_dist_bps"].mean() > 0 else None
        # Breakeven win rate for actual average distances (before costs).
        be = float(trades_df["sl_dist_bps"].mean()) / (float(trades_df["sl_dist_bps"].mean()) + float(trades_df["tp_dist_bps"].mean()))
        metrics["breakeven_win_rate_pct"] = round(be * 100.0, 2)
        metrics["edge_vs_breakeven_pp"] = round(float((trades_df["pnl_net"] > 0).mean() * 100.0) - be * 100.0, 2)
        # Statistical t-stat on per-trade PnL (i.i.d. approximation).
        sd = float(trades_df["pnl_net"].std(ddof=1))
        se = sd / np.sqrt(len(trades_df)) if sd > 0 else 0.0
        metrics["t_stat_avg_pnl"] = round(float(trades_df["pnl_net"].mean()) / se, 2) if se > 0 else 0.0
        # Fee burden.
        gross = float(abs(trades_df["pnl_net"]).sum())
        metrics["fees_pct_of_abs_pnl"] = round(float(trades_df["fees"].sum()) / gross * 100.0, 1) if gross > 0 else None
        metrics["fees_pct_of_initial_capital"] = round(float(trades_df["fees"].sum()) / INITIAL_CAPITAL * 100.0, 2)
        metrics["cost_per_trade_pct_capital"] = round(float(trades_df["fees"].mean()) / INITIAL_CAPITAL * 100.0, 3)
        # Monte-Carlo bootstrap on trade PnL (i.i.d. *resample with replacement*,
        # 500 paths). This is a distribution of the total realised PnL (not an
        # equity-curve permutation), useful only as a rough significance check.
        rng = np.random.default_rng(42)
        pnl_arr = trades_df["pnl_net"].to_numpy()
        finals = []
        for _ in range(500):
            sample = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
            finals.append((INITIAL_CAPITAL + float(sample.sum())) / INITIAL_CAPITAL - 1.0)
        metrics["bootstrap_return_5th"] = round(float(np.percentile(finals, 5)) * 100.0, 2)
        metrics["bootstrap_return_50th"] = round(float(np.percentile(finals, 50)) * 100.0, 2)
        metrics["bootstrap_return_95th"] = round(float(np.percentile(finals, 95)) * 100.0, 2)
        metrics["max_consecutive_losses"] = int((trades_df["pnl_net"] <= 0).astype(int).groupby(
            (trades_df["pnl_net"] > 0).cumsum()).cumsum().max()) if len(trades_df) else 0
        metrics["worst_trade"] = round(float(trades_df["pnl_net"].min()), 2)
        metrics["best_trade"] = round(float(trades_df["pnl_net"].max()), 2)

    return metrics, trades_df


def _metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {
            "n_trades": 0, "total_return_pct": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
            "avg_trade_pnl": 0.0, "avg_bars_held": 0.0, "fees_total": 0.0,
            "exit_type_counts": {}, "long_trades": 0, "short_trades": 0,
            "avg_atr_entry_bps": 0.0, "avg_notional_equity": 0.0,
        }
    eq = equity.set_index("open_time")["equity"]
    rets = eq.pct_change().dropna()
    bars_per_year = 288 * 365
    sharpe = float(np.sqrt(bars_per_year) * rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else 0.0
    peak = eq.cummax()
    max_dd = float((eq / peak - 1.0).min())
    gains = trades.loc[trades["pnl_net"] > 0, "pnl_net"].sum()
    losses = -trades.loc[trades["pnl_net"] < 0, "pnl_net"].sum()
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    return {
        "n_trades": int(len(trades)),
        "total_return_pct": float(eq.iloc[-1] / INITIAL_CAPITAL - 1.0) * 100.0,
        "win_rate": float((trades["pnl_net"] > 0).mean() * 100.0),
        "profit_factor": pf,
        "max_drawdown": max_dd * 100.0,
        "sharpe": round(sharpe, 3),
        "avg_trade_pnl": float(trades["pnl_net"].mean()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "fees_total": float(trades["fees"].sum()),
        "exit_type_counts": trades["exit_type"].value_counts().to_dict(),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "avg_atr_entry_bps": float(trades["atr_entry_bps"].mean()),
        "avg_notional_equity": float(trades["notional_equity"].mean()),
    }


def main() -> int:
    cfg = Config()
    print(f"Load BTCUSDT M5 ...")
    df, src = load_klines(cfg.data_dir, cfg.symbol, cfg.interval, cfg.demo_data_dir)
    df = df.sort_index()
    # ensure UTC-aware DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    start, end = "2026-01-01", "2026-06-30"
    tz = df.index.tz
    window = df.loc[pd.Timestamp(start, tz=tz):pd.Timestamp(end, tz=tz)]
    bh_ret = (float(window["close"].iloc[-1]) / float(window["open"].iloc[0]) - 1.0) * 100.0
    print(f"   rows={len(df):,}  H1 window rows={len(window):,}")
    print(f"   Buy&hold Jan-Jun 2026 = {bh_ret:+.2f}%")

    print("\nTEMA as-authored (risk 3% x3, no notional cap) ...")
    m_auth, t_auth = run_tema_backtest(df, start, end, risk_pct=0.03, qty_mult=3.0, notional_cap=None)
    print(json.dumps(m_auth, indent=2, default=str))

    print("\nTEMA normalised (risk 1%, 1x notional cap) ...")
    m_norm, t_norm = run_tema_backtest(df, start, end, risk_pct=0.01, qty_mult=1.0, notional_cap=1.0)
    print(json.dumps(m_norm, indent=2, default=str))

    out_dir = Path("output/BTCUSDT/tema_trend_2026H1")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": src,
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "period": {"start": start, "end": end},
        "buy_hold_return_pct": round(bh_ret, 3),
        "strategy": {
            "name": "TemaTrendFollowing (Jesse, replicated)",
            "rules": {
                "entry": "limit 1 ATR from signal close (valid 1 bar, cancelled next bar)",
                "stop_loss": "4 * ATR (fill-bar ATR)",
                "take_profit": "3 * ATR (fill-bar ATR)",
                "filters": "TEMA10>TEMA80 (M5), TEMA20>TEMA70 (4h last-completed), ADX>40, CMO>±40",
                "time_stop": "none in original; END exit is only window settlement",
                "sizing": "risk% of equity to SL, then x qty_mult",
            },
            "assumptions": {
                "fee_bps": 4.0, "slip_bps": 1.0, "initial_capital": INITIAL_CAPITAL,
                "4h_indicators": "last fully completed 4h candle (no partial-candle lookahead)",
                "taker_fee_on_limit_entry": "conservative; live maker fill may be cheaper",
            },
        },
        "as_authored": m_auth,
        "normalised": m_norm,
    }
    t_auth.to_csv(out_dir / "trades_as_authored.csv", index=False)
    t_norm.to_csv(out_dir / "trades_normalised.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nSaved: {out_dir}/report.json")
    print(f"Saved: {out_dir}/trades_as_authored.csv")
    print(f"Saved: {out_dir}/trades_normalised.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
