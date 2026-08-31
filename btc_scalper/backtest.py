"""Realistic event-driven backtester for the 5m scalping model.

Execution model
---------------
* Signal is evaluated on the close of bar ``t``.
* Execution happens on the **open** of bar ``t+1`` (no same-bar look-ahead).
* Both entry and exit use market orders with taker fee + per-side slippage.
* Positions are sized by risk: ``risk_pct`` of current equity is risked to the
  stop-loss, capped at ``notional_pct_cap`` of equity (spot, no leverage).
* If a bar pierces both TP and SL, the loss (SL) is assumed first
  (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _fee_rate(cfg) -> float:
    return cfg.taker_fee_bps / 10_000.0


def _slippage(cfg) -> float:
    return cfg.slippage_bps / 10_000.0


def _open_position(df: pd.DataFrame, k: int, side: int, equity: float, cfg) -> dict:
    """Create a position at the open of bar k. side=+1 long, -1 short."""
    obar = df.iloc[k]
    raw_entry = float(obar["open"])
    slip = _slippage(cfg)
    entry = raw_entry * (1.0 + slip) if side > 0 else raw_entry * (1.0 - slip)
    tp_pct = cfg.tp_bps / 10_000.0
    sl_pct = cfg.sl_bps / 10_000.0

    if side > 0:
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)
        sl_dist = entry - sl
    else:
        tp = entry * (1.0 - tp_pct)
        sl = entry * (1.0 + sl_pct)
        sl_dist = sl - entry

    risk_amount = equity * cfg.risk_pct
    qty = risk_amount / max(sl_dist, 1e-12)
    notional_cap = equity * cfg.notional_pct_cap
    qty = min(qty, notional_cap / max(entry, 1e-12))
    if qty <= 0:
        return None

    return {
        "side": side,
        "entry_idx": k,
        "entry_time": df.index[k],
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "qty": qty,
        "bars_remaining": cfg.max_hold_bars - 1,
        "entry_fee": qty * entry * _fee_rate(cfg),
        "entry_notional": qty * entry,
    }


def _check_exit(obar: pd.Series, pos: dict, cfg) -> tuple[int, float]:
    """Return (exit_type, exit_price): 1=TP, -1=SL, 0=no-exit, 9=time-stop."""
    high = float(obar["high"])
    low = float(obar["low"])
    if pos["side"] > 0:
        # Conservative: SL first when both are hit on the same bar.
        if low <= pos["sl"]:
            return -1, pos["sl"]
        if high >= pos["tp"]:
            return 1, pos["tp"]
    else:
        if high >= pos["sl"]:
            return -1, pos["sl"]
        if low <= pos["tp"]:
            return 1, pos["tp"]
    return 0, np.nan


def _direction_at(prob_up: float, prob_down: float, thresh: float, margin: float) -> int:
    if prob_up >= thresh and (prob_up - prob_down) >= margin:
        return 1
    if prob_down >= thresh and (prob_down - prob_up) >= margin:
        return -1
    return 0


def run_backtest(df: pd.DataFrame, predictions: pd.DataFrame, cfg) -> BacktestResult:
    """Run a trade-by-trade backtest using out-of-sample predictions.

    The core simulation is written with numpy arrays and a trade loop (only
    trades are iterated, not every bar), which keeps the detailed parameter
    sweep fast while preserving the execution rules:

    - signal on close of bar ``t``, entry on the open of bar ``t+1``,
    - taker fees + per-side slippage,
    - TP/SL checked per bar with conservative SL-first ordering,
    - time-stop at ``max_hold_bars``,
    - ``cooldown`` bars between an exit and the next entry.
    """
    if "prob_up" not in predictions.columns or "prob_down" not in predictions.columns:
        raise ValueError("predictions must contain prob_up / prob_down columns")

    merged = df.copy()
    pred = predictions[["open_time", "prob_up", "prob_down"]].copy()
    pred = pred.drop_duplicates(subset="open_time").set_index("open_time")
    merged = merged.join(pred, how="left")
    merged["prob_up"] = merged["prob_up"].fillna(0.5)
    merged["prob_down"] = merged["prob_down"].fillna(0.5)

    n = len(merged)
    if n < 100:
        raise ValueError("Not enough rows to backtest.")

    idx = merged.index.to_numpy()
    open_p = merged["open"].to_numpy(dtype=float)
    high_p = merged["high"].to_numpy(dtype=float)
    low_p = merged["low"].to_numpy(dtype=float)
    close_p = merged["close"].to_numpy(dtype=float)
    prob_up = merged["prob_up"].to_numpy(dtype=float)
    prob_down = merged["prob_down"].to_numpy(dtype=float)

    thresh = cfg.probability_threshold
    margin = getattr(cfg, "probability_margin", 0.0)
    cooldown = int(getattr(cfg, "cooldown_bars", 0))
    fee = _fee_rate(cfg)
    slip = _slippage(cfg)
    max_hold = int(cfg.max_hold_bars)
    tp_pct = cfg.tp_bps / 10_000.0
    sl_pct = cfg.sl_bps / 10_000.0

    cash = float(cfg.initial_capital)
    next_entry_bar = 0
    trades: list[dict] = []
    i = 0

    while i < n - 1:
        # Find the next eligible signal bar.
        while i < n - 1 and i < next_entry_bar:
            i += 1
        if i >= n - 1:
            break

        side = _direction_at(prob_up[i], prob_down[i], thresh, margin)
        if side == 0:
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break

        raw_entry = float(open_p[entry_idx])
        entry = raw_entry * (1.0 + slip) if side > 0 else raw_entry * (1.0 - slip)
        if side > 0:
            tp = entry * (1.0 + tp_pct)
            sl = entry * (1.0 - sl_pct)
            sl_dist = entry - sl
        else:
            tp = entry * (1.0 - tp_pct)
            sl = entry * (1.0 + sl_pct)
            sl_dist = sl - entry

        risk_amount = cash * cfg.risk_pct
        qty = risk_amount / max(sl_dist, 1e-12)
        notional_cap = cash * cfg.notional_pct_cap
        qty = min(qty, notional_cap / max(entry, 1e-12))
        if qty <= 0:
            i += 1
            continue

        entry_fee = qty * entry * fee

        # Find exit within the holding window.
        window_end = min(n - 1, entry_idx + max_hold - 1)
        exit_j = -1
        exit_type = 0
        exit_price = np.nan
        for j in range(entry_idx, window_end + 1):
            if side > 0:
                if low_p[j] <= sl:
                    exit_j, exit_type, exit_price = j, -1, sl
                    break
                if high_p[j] >= tp:
                    exit_j, exit_type, exit_price = j, 1, tp
                    break
            else:
                if high_p[j] >= sl:
                    exit_j, exit_type, exit_price = j, -1, sl
                    break
                if low_p[j] <= tp:
                    exit_j, exit_type, exit_price = j, 1, tp
                    break
        if exit_j < 0:
            exit_j = window_end
            exit_type = 9
            exit_price = float(close_p[window_end])

        if side > 0:
            exit_price = exit_price * (1.0 - slip)
        else:
            exit_price = exit_price * (1.0 + slip)
        exit_fee = qty * exit_price * fee
        gross = (exit_price - entry) * qty * side
        net = gross - entry_fee - exit_fee
        cash += net

        type_label = "TP" if exit_type == 1 else ("SL" if exit_type == -1 else "TIME")
        trades.append(
            {
                "side": "long" if side > 0 else "short",
                "entry_time": pd.Timestamp(idx[entry_idx]),
                "exit_time": pd.Timestamp(idx[exit_j]),
                "entry_pos": entry_idx,
                "exit_pos": exit_j,
                "entry_price": entry,
                "exit_price": exit_price,
                "qty": qty,
                "exit_type": type_label,
                "pnl_net": net,
                "pnl_pct": net / (qty * entry) * 100.0,
                "bars_held": exit_j - entry_idx,
                "fees": entry_fee + exit_fee,
                "equity_after": cash,
            }
        )

        next_entry_bar = exit_j + cooldown
        i = exit_j + 1

    trades_df = pd.DataFrame(trades)

    # Build marked-to-market equity per bar.
    cash_series = np.full(n, float(cfg.initial_capital))
    unrealized = np.zeros(n)
    if len(trades):
        exit_idx = np.array([t["exit_pos"] for t in trades], dtype=int)
        pnl = np.array([t["pnl_net"] for t in trades])
        pnl_by_bar = np.zeros(n)
        np.add.at(pnl_by_bar, exit_idx, pnl)
        cash_series = float(cfg.initial_capital) + np.cumsum(pnl_by_bar)
        for t in trades:
            entry_pos = int(t["entry_pos"])
            exit_pos = int(t["exit_pos"])
            direction = 1.0 if t["side"] == "long" else -1.0
            qty_t = t["qty"]
            entry_t = t["entry_price"]
            end = max(entry_pos, exit_pos)
            segment = np.arange(entry_pos, end)
            unrealized[segment] += (close_p[segment] - entry_t) * qty_t * direction

    equity_vals = cash_series + unrealized
    equity_df = pd.DataFrame({"open_time": pd.Index(idx, name="open_time"), "equity": equity_vals})
    equity_df["open_time"] = pd.to_datetime(equity_df["open_time"], utc=True)

    metrics = _performance_metrics(trades_df, equity_df, cfg)
    return BacktestResult(trades=trades_df, equity=equity_df, metrics=metrics)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _performance_metrics(trades: pd.DataFrame, equity: pd.DataFrame, cfg) -> dict:
    if len(trades) == 0 or len(equity) == 0:
        return {
            "n_trades": 0,
            "final_equity": cfg.initial_capital,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_bars_held": 0.0,
            "fees_total": 0.0,
            "exit_type_counts": {},
        }

    gains = trades.loc[trades["pnl_net"] > 0, "pnl_net"].sum()
    losses = -trades.loc[trades["pnl_net"] < 0, "pnl_net"].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

    eq = equity.set_index("open_time")["equity"]
    rets = eq.pct_change().dropna()
    # Annualisation from 5-minute bars: 288 bars/day, 365 days/year.
    bars_per_year = 288 * 365
    sharpe = float(np.sqrt(bars_per_year) * rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else 0.0
    downside = rets[rets < 0].std() if (rets < 0).any() else np.nan
    sortino = float(np.sqrt(bars_per_year) * rets.mean() / downside) if downside and downside > 0 else 0.0
    max_dd = _max_drawdown(eq)
    total_return = float(eq.iloc[-1] / cfg.initial_capital - 1.0)

    return {
        "n_trades": int(len(trades)),
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": total_return * 100.0,
        "win_rate": float((trades["pnl_net"] > 0).mean() * 100.0),
        "profit_factor": profit_factor,
        "max_drawdown": max_dd * 100.0,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "avg_trade_pnl": float(trades["pnl_net"].mean()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "fees_total": float(trades["fees"].sum()),
        "exit_type_counts": trades["exit_type"].value_counts().to_dict() if len(trades) else {},
    }
