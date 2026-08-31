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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ADX (directional strength), 0..100."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Money Flow Index, 0..100."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    positive = mf.where(tp > tp.shift(), 0.0)
    negative = mf.where(tp < tp.shift(), 0.0)
    pos_sum = positive.rolling(window, min_periods=window // 2).sum()
    neg_sum = negative.rolling(window, min_periods=window // 2).sum()
    return 100.0 - 100.0 / (1.0 + pos_sum / neg_sum.replace(0.0, np.nan))


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Fast relative position of the latest value inside a rolling window (0..1).

    Min/max normalization is used instead of an exact percentile rank because
    the exact rolling rank is prohibitively slow on 500k+ 5m bars.
    """
    lo = series.rolling(window, min_periods=max(2, window // 3)).min()
    hi = series.rolling(window, min_periods=max(2, window // 3)).max()
    return (series - lo) / (hi - lo).replace(0.0, np.nan)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between two aligned series."""
    return a.rolling(window, min_periods=max(3, window // 4)).corr(b)


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

    # ------------------------------------------------------ advanced order-flow
    # Buy-vs-sell aggression pressure, normalized to [~-1, ~+1].
    sell_base = volume - taker_base
    sell_quote = quote - taker_quote
    aggr_base = (taker_base - sell_base) / volume.replace(0.0, np.nan)
    aggr_quote = (taker_quote - sell_quote) / quote.replace(0.0, np.nan)
    out["f_aggr_base"] = aggr_base
    out["f_aggr_quote"] = aggr_quote
    for span in (5, 12):
        out[f"f_aggr_ema_{span}"] = _ema(aggr_quote, span)
        out[f"f_sell_ratio_ema_{span}"] = _ema(sell_quote / quote.replace(0.0, np.nan), span)
        out[f"f_flow_quote_ema_{span}"] = _ema(out["f_flow_quote"], span)

    # Cumulative (signed) delta: rolling window and full-history normalized slope.
    cdelta = (taker_base - sell_base).cumsum()
    for n in (20, 60):
        cd_z = (cdelta - cdelta.rolling(n, min_periods=n // 3).mean()) / cdelta.rolling(
            n, min_periods=n // 3
        ).std().replace(0.0, np.nan)
        out[f"f_cdelta_z_{n}"] = cd_z
        out[f"f_cdelta_slope_{n}"] = cdelta.diff(n) / cdelta.rolling(n, min_periods=n // 3).std().replace(0.0, np.nan)

    # Order-flow vs price divergence (is volume supporting the move?).
    pch = close.pct_change()
    flow_pct = out["f_flow_quote"]
    for n in (5, 20):
        out[f"f_flow_price_corr_{n}"] = _rolling_corr(pch, flow_pct, n)

    # Execution microstructure: average trade size + trade-count momentum.
    out["f_avg_trade"] = quote / trades.replace(0.0, np.nan)
    out["f_avg_trade_log"] = np.log1p(out["f_avg_trade"])
    for n in (10, 30):
        avg_ts = out["f_avg_trade"]
        out[f"f_avg_trade_ratio_{n}"] = avg_ts / avg_ts.rolling(n, min_periods=n // 3).mean()
        out[f"f_trade_log_z_{n}"] = (np.log1p(trades) - np.log1p(trades).rolling(n, min_periods=n // 3).mean()) / np.log1p(trades).rolling(n, min_periods=n // 3).std().replace(0.0, np.nan)

    # Liquidity / participation regimes.
    for n in (20, 60):
        out[f"f_volume_rank_{n}"] = _rolling_rank(volume, n)
        out[f"f_quote_rank_{n}"] = _rolling_rank(quote, n)
        out[f"f_trades_rank_{n}"] = _rolling_rank(trades, n)
        out[f"f_vol_z_{n}"] = (volume - volume.rolling(n, min_periods=n // 3).mean()) / volume.rolling(n, min_periods=n // 3).std().replace(0.0, np.nan)

    # ------------------------------------------------------------ regime features
    # Volatility regime (rolling percentile) and range expansion/contraction.
    for n in (30, 60):
        rvol = close.pct_change().rolling(n, min_periods=n // 3).std()
        out[f"f_rvol_rank_{n}"] = _rolling_rank(rvol.fillna(0.0), n)
        out[f"f_atr_rank_{n}"] = _rolling_rank(true_range.fillna(0.0), n)
    ravg = true_range.rolling(20, min_periods=5).mean()
    out["f_range_expand"] = true_range / ravg.replace(0.0, np.nan)
    out["f_range_expand_60"] = true_range / true_range.rolling(60, min_periods=10).mean().replace(0.0, np.nan)

    # Trend strength / regime.
    adx = _adx(high, low, close, 14)
    out["f_adx_14"] = adx
    out["f_di_plus_14"] = 100.0 * (high.diff().clip(lower=0.0)).ewm(alpha=1 / 14, adjust=False).mean() / true_range.ewm(alpha=1 / 14, adjust=False).mean().replace(0.0, np.nan)
    out["f_di_minus_14"] = 100.0 * (-low.diff().clip(lower=0.0)).ewm(alpha=1 / 14, adjust=False).mean() / true_range.ewm(alpha=1 / 14, adjust=False).mean().replace(0.0, np.nan)
    out["f_dx_44"] = _adx(high, low, close, 44)

    # Donchian channel position (normalized inside the N-bar range).
    for n in (20, 60):
        rng_hi = high.rolling(n, min_periods=n // 3).max()
        rng_lo = low.rolling(n, min_periods=n // 3).min()
        spread = (rng_hi - rng_lo).replace(0.0, np.nan)
        out[f"f_donchian_pos_{n}"] = (close - rng_lo) / spread

    # Money-flow style measures.
    mfi14 = _mfi(high, low, close, volume, 14)
    out["f_mfi_14"] = mfi14
    out["f_cmf_20"] = ((close - low) - (high - close)) / (high - low).replace(0.0, np.nan) * volume
    out["f_cmf_z_20"] = (out["f_cmf_20"] - out["f_cmf_20"].rolling(20, min_periods=5).mean()) / out["f_cmf_20"].rolling(20, min_periods=5).std().replace(0.0, np.nan)

    # Wick balance (continuous): positive = more lower-wick (buyer support).
    out["f_wick_skew"] = (out["f_lower_wick"] - out["f_upper_wick"]) / rng
    out["f_wick_skew_ema"] = _ema(out["f_wick_skew"], 12)

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
