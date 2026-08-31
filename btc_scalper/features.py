"""Feature engineering for 5-minute klines.

Every feature is strictly backward-looking (uses information available at the
close of the current candle), so there is no look-ahead leakage into the labels
or into the walk-forward training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RET_WINDOWS = [1, 2, 3, 5, 10, 15, 30, 60]
SMA_WINDOWS = [10, 20, 50]
EMA_WINDOWS = [9, 21]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there is no loss, RSI is effectively 100.
    out = out.where(loss > 0, 100.0 * (gain > 0))
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scalar/quant features to a Binance-style kline DataFrame.

    Parameters
    ----------
    df: DataFrame indexed by ``open_time`` with at least columns
        open, high, low, close, volume, quote_volume, trades,
        taker_base, taker_quote.

    Returns
    -------
    A new DataFrame with the input plus ``f_*`` feature columns.
    """
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]
    quote = out["quote_volume"].fillna(out["volume"] * close)
    trades = out["trades"].fillna(0.0)
    taker_base = out["taker_base"].fillna(out["volume"] / 2.0)
    taker_quote = out["taker_quote"].fillna(quote / 2.0)

    # ------------------------------------------------------------ returns & momentum
    prev = [1] + RET_WINDOWS[1:]
    for n in RET_WINDOWS:
        out[f"f_ret_{n}"] = close.pct_change(n)

    for n in (5, 10, 20):
        out[f"f_mom_{n}"] = close / close.shift(n) - 1.0

    # ------------------------------------------------------------ moving averages
    for n in SMA_WINDOWS:
        sma = close.rolling(n, min_periods=max(2, n // 3)).mean()
        out[f"f_sma_{n}"] = sma
        out[f"f_dist_sma_{n}"] = close / sma - 1.0
    for n in EMA_WINDOWS:
        ema = _ema(close, n)
        out[f"f_ema_{n}"] = ema
        out[f"f_dist_ema_{n}"] = close / ema - 1.0

    # ------------------------------------------------------------ oscillators
    out["f_rsi_14"] = _rsi(close, 14)
    out["f_rsi_28"] = _rsi(close, 28)
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    out["f_macd"] = macd
    out["f_macd_signal"] = signal
    out["f_macd_hist"] = macd - signal
    out["f_macd_pct"] = macd / close

    # ------------------------------------------------------------ volatility / ATR
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    for n in (14, 21):
        atr = true_range.rolling(n, min_periods=n // 2).mean()
        out[f"f_atr_{n}"] = atr
        out[f"f_natr_{n}"] = atr / close

    for n in (10, 30):
        out[f"f_rvol_{n}"] = close.pct_change().rolling(n, min_periods=n // 2).std()

    # ------------------------------------------------------------ candle shape
    body = (close - out["open"]).abs()
    rng = (high - low).replace(0.0, np.nan)
    out["f_range_pct"] = rng / close
    out["f_body_pct"] = body / close
    out["f_body_frac"] = body / rng
    oc_max = pd.concat([out["open"], close], axis=1).max(axis=1)
    oc_min = pd.concat([out["open"], close], axis=1).min(axis=1)
    out["f_upper_wick"] = high - oc_max
    out["f_lower_wick"] = oc_min - low
    out["f_upper_wick_frac"] = out["f_upper_wick"] / rng
    out["f_lower_wick_frac"] = out["f_lower_wick"] / rng
    out["f_cpos"] = (close - low) / rng
    out["f_gap"] = out["open"] / close.shift() - 1.0
    out["f_body_sign"] = np.sign(body * np.sign(close - out["open"]))

    # ------------------------------------------------------------ volume & order-flow
    out["f_log_volume"] = np.log1p(volume)
    for n in (5, 20):
        out[f"f_vol_ratio_{n}"] = volume / volume.rolling(n, min_periods=max(2, n // 3)).mean()
        out[f"f_quote_ratio_{n}"] = quote / quote.rolling(n, min_periods=max(2, n // 3)).mean()
        out[f"f_trades_ratio_{n}"] = trades / trades.rolling(n, min_periods=max(2, n // 3)).mean()

    out["f_taker_ratio_base"] = taker_base / volume.replace(0.0, np.nan)
    out["f_taker_ratio_quote"] = taker_quote / quote.replace(0.0, np.nan)
    delta_base = 2.0 * taker_base - volume
    delta_quote = 2.0 * taker_quote - quote
    out["f_flow_base"] = delta_base / volume.replace(0.0, np.nan)
    out["f_flow_quote"] = delta_quote / quote.replace(0.0, np.nan)
    for n in (5, 20):
        flow_mean = out[f"f_flow_base"].rolling(n, min_periods=max(2, n // 3)).mean()
        flow_std = out[f"f_flow_base"].rolling(n, min_periods=max(2, n // 3)).std()
        out[f"f_flow_z_{n}"] = (out[f"f_flow_base"] - flow_mean) / flow_std.replace(0.0, np.nan)

    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    out["f_obv_slope_10"] = obv.diff(10) / obv.rolling(10, min_periods=3).std().replace(0.0, np.nan)

    # ------------------------------------------------------------ trend context
    out["f_hi_shr"] = high.rolling(20, min_periods=5).max() / close - 1.0
    out["f_lo_shr"] = close / low.rolling(20, min_periods=5).min() - 1.0
    out["f_streak"] = np.sign(close.diff()).fillna(0.0).groupby(
        np.sign(close.diff()).fillna(0.0).ne(
            np.sign(close.diff()).fillna(0.0).shift()
        ).cumsum()
    ).cumcount() + 1.0

    # ------------------------------------------------------------ time-of-day
    idx = out.index
    out["f_hour"] = idx.hour
    out["f_dow"] = idx.dayofweek
    out["f_minute"] = idx.minute
    # UTC session flags
    out["f_session_asia"] = (idx.hour >= 1) & (idx.hour < 7)
    out["f_session_euro"] = (idx.hour >= 7) & (idx.hour < 13)
    out["f_session_us"] = (idx.hour >= 13) & (idx.hour < 21)

    # Epoch seconds: harmless for the model but helps capture long-horizon drift.
    out["f_time_sec"] = idx.astype("int64") // 10**9

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]
