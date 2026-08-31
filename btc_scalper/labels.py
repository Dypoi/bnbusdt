"""Label generation for scalping targets.

Two label families are provided:

1. ``label_up`` / ``label_down`` – next ``label_horizon`` bar direction
   (1 if price closes higher after ``h`` bars else 0).
2. ``label_scalp_up`` / ``label_scalp_down`` – whether a TP/SL scalp is
   resolved in the correct direction within ``scalp_bars`` bars. A bar is left
   as NaN (dropped by the model) when the outcome is ambiguous (e.g. the same
   bar pierces both TP and SL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _first_touch_label(
    df: pd.DataFrame,
    n_bars: int,
    tp_price: pd.Series,
    sl_price: pd.Series,
    side: int,
) -> pd.Series:
    """Label whether TP or SL is touched first within the next ``n_bars``.

    Returns 1 for TP, 0 for SL, NaN when the outcome is ambiguous (e.g. the
    same bar pierces both levels) or when no level is hit.
    """
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    tp = tp_price.to_numpy()
    sl = sl_price.to_numpy()
    out = np.full(n, np.nan)
    decided = np.zeros(n, dtype=bool)
    for lag in range(1, n_bars + 1):
        if side > 0:
            tp_hit = high[lag:] >= tp[:-lag]
            sl_hit = low[lag:] <= sl[:-lag]
        else:
            tp_hit = low[lag:] <= tp[:-lag]
            sl_hit = high[lag:] >= sl[:-lag]
        # Align shifts back to the originating bar index.
        hit_tp = np.zeros(n, dtype=bool)
        hit_sl = np.zeros(n, dtype=bool)
        hit_tp[:-lag] = tp_hit
        hit_sl[:-lag] = sl_hit
        both = hit_tp & hit_sl & ~decided
        tp_only = hit_tp & ~hit_sl & ~decided
        sl_only = hit_sl & ~hit_tp & ~decided
        out[both] = np.nan        # ambiguous bar
        out[tp_only] = 1.0
        out[sl_only] = 0.0
        decided |= hit_tp | hit_sl
    return pd.Series(out, index=df.index)


def add_labels(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add target columns for the configured label/TP/SL settings."""
    out = df.copy()
    close = out["close"]

    # 1) next-bar / next-h bars direction ----------------------------------
    up = close.shift(-cfg.label_horizon) > close
    out["label_up"] = up.astype(float)
    out["label_down"] = (1.0 - up).astype(float)
    # Mark the last ``label_horizon`` bars as unknown (no future available).
    out.loc[out["label_up"].isna(), ["label_up", "label_down"]] = np.nan

    # 2) TP/SL scalp labels ------------------------------------------------
    tp_up = close * (1.0 + cfg.tp_bps / 10_000.0)
    sl_up = close * (1.0 - cfg.sl_bps / 10_000.0)
    out["label_scalp_up"] = _first_touch_label(out, cfg.scalp_bars, tp_up, sl_up, side=1)

    tp_dn = close * (1.0 - cfg.tp_bps / 10_000.0)
    sl_dn = close * (1.0 + cfg.sl_bps / 10_000.0)
    out["label_scalp_down"] = _first_touch_label(out, cfg.scalp_bars, tp_dn, sl_dn, side=-1)

    return out
