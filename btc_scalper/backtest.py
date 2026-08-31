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


def run_backtest(df: pd.DataFrame, predictions: pd.DataFrame, cfg) -> BacktestResult:
    """Run a trade-by-trade backtest using out-of-sample predictions.

    Parameters
    ----------
    df : full kline + feature frame indexed by ``open_time``.
    predictions : DataFrame with ``open_time``, ``prob_up``, ``prob_down``.
    """
    if "prob_up" not in predictions.columns:
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

    thresh = cfg.probability_threshold
    margin = getattr(cfg, "probability_margin", 0.0)
    cooldown = getattr(cfg, "cooldown_bars", 0)
    cash = cfg.initial_capital
    pos = None
    next_entry_bar = 0
    trades: list[dict] = []
    eq_points: list[dict] = []

    for k in range(n - 1):
        bar = merged.iloc[k]

        # ---- 1. Manage an existing position on bar k ----------------------
        if pos is not None:
            exit_type, exit_price = _check_exit(bar, pos, cfg)
            exit_fee = 0.0
            if exit_type == 0 and pos["bars_remaining"] > 0:
                pos["bars_remaining"] -= 1
            elif exit_type == 0:  # time stop at the close
                exit_type = 9
                exit_price = float(bar["close"])
            if exit_type != 0:
                slip = _slippage(cfg)
                exit_price = exit_price * (1.0 - slip) if pos["side"] > 0 else exit_price * (1.0 + slip)
                exit_fee = pos["qty"] * exit_price * _fee_rate(cfg)
                direction = pos["side"]
                gross = (exit_price - pos["entry"]) * pos["qty"] * direction
                net = gross - pos["entry_fee"] - exit_fee
                cash += net
                trades.append(
                    {
                        "side": "long" if direction > 0 else "short",
                        "entry_time": pos["entry_time"],
                        "exit_time": bar.name,
                        "entry_price": pos["entry"],
                        "exit_price": exit_price,
                        "qty": pos["qty"],
                        "exit_type": "TP" if exit_type == 1 else ("SL" if exit_type == -1 else "TIME"),
                        "pnl_net": net,
                        "pnl_pct": net / (pos["qty"] * pos["entry"]) * 100.0,
                        "bars_held": int((bar.name - pos["entry_time"]) // pd.Timedelta(minutes=5)),
                        "fees": pos["entry_fee"] + exit_fee,
                        "equity_after": cash,
                    }
                )
                # Wait ``cooldown`` bars before re-entering on a fresh signal.
                next_entry_bar = k + cooldown
                pos = None

        # ---- 2. Open a new position on the next bar open ------------------
        if pos is None and k >= next_entry_bar and k + 1 < n:
            p_up = float(bar["prob_up"])
            p_down = float(bar["prob_down"])
            side = 0
            if p_up >= thresh and (p_up - p_down) >= margin:
                side = 1
            elif p_down >= thresh and (p_down - p_up) >= margin:
                side = -1
            if side != 0:
                pos = _open_position(merged, k + 1, side, cash, cfg)

        # Record marked-to-market equity snapshot at the close of bar k.
        unrealized = 0.0
        if pos is not None:
            mark = float(bar["close"])
            unrealized = (mark - pos["entry"]) * pos["qty"] * pos["side"]
        eq_points.append({"open_time": bar.name, "equity": cash + unrealized})

    # Close any dangling position on the last bar.
    if pos is not None:
        bar = merged.iloc[n - 1]
        exit_price = float(bar["close"]) * (1.0 - _slippage(cfg)) if pos["side"] > 0 else float(bar["close"]) * (1.0 + _slippage(cfg))
        exit_fee = pos["qty"] * exit_price * _fee_rate(cfg)
        gross = (exit_price - pos["entry"]) * pos["qty"] * pos["side"]
        net = gross - pos["entry_fee"] - exit_fee
        cash += net
        trades.append(
            {
                "side": "long" if pos["side"] > 0 else "short",
                "entry_time": pos["entry_time"],
                "exit_time": bar.name,
                "entry_price": pos["entry"],
                "exit_price": exit_price,
                "qty": pos["qty"],
                "exit_type": "END",
                "pnl_net": net,
                "pnl_pct": net / (pos["qty"] * pos["entry"]) * 100.0,
                "bars_held": int((bar.name - pos["entry_time"]) // pd.Timedelta(minutes=5)),
                "fees": pos["entry_fee"] + exit_fee,
                "equity_after": cash,
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(eq_points)
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
            "avg_trade_pnl": 0.0,
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
