"""Data access for the Binance Vision kline archive.

The files in ``Dataset_BTCUSDT/`` are monthly ZIP archives in Binance Vision
format. Each ZIP contains one CSV with 12 columns and NO header:

    open_time, open, high, low, close, volume,
    close_time, quote_volume, number_of_trades,
    taker_buy_base_volume, taker_buy_quote_volume, ignore

This module loads that exact format for BTCUSDT (or any symbol/interval), so
the same parser also works for a ``Dataset_BNBUSDT/`` folder if the pair ever
changes.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]
FLOAT_COLS = [c for c in KLINE_COLUMNS if c not in ("open_time", "close_time")]
FILE_RE = re.compile(r"(?P<symbol>[A-Z0-9]+)-(?P<interval>[0-9a-z]+)-(?P<ts>[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)\.(?P<ext>zip|csv)")


def _parse_csv(text: str, symbol: str) -> pd.DataFrame:
    first = text.splitlines()[0]
    has_header = "open_time" in first.lower() or "Open time" in first
    names = list(KLINE_COLUMNS) if not has_header else None
    df = pd.read_csv(__import__("io").StringIO(text), names=names)
    # Keep the full columns if a header was present (defensive).
    if has_header:
        df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
        for col in KLINE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[[c for c in KLINE_COLUMNS if c in df.columns]]
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol
    return df


def _read_archive(path: Path, symbol: str) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = zf.namelist()[0]
            text = zf.read(member).decode("utf-8")
            df = _parse_csv(text, symbol)
    elif path.suffix == ".csv":
        df = _parse_csv(path.read_text(encoding="utf-8"), symbol)
    else:
        raise ValueError(f"Unsupported file: {path}")
    return _normalize_archive_timestamps(df)


def _normalize_archive_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Binance Vision changed the kline epoch unit from ms to microseconds
    around recent years. Normalise every archive to **milliseconds**."""
    ts = df["open_time"].astype("int64")
    if ts.median() > 1e14:  # microseconds (e.g. 1735689600000000 -> 1735689600000)
        df["open_time"] = ts // 1000
        df["close_time"] = df["close_time"].astype("int64") // 1000
    return df


def _discover(data_dir: Path, symbol: str, interval: str) -> list[Path]:
    if not data_dir.exists():
        return []
    paths = sorted(data_dir.glob(f"{symbol}-{interval}-*.zip")) + sorted(
        data_dir.glob(f"{symbol}-{interval}-*.csv")
    )
    return paths


def load_klines(
    data_dir: Path,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    demo_data_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Load monthly ZIP/CSV kline files for ``symbol``.

    Returns ``(df, source)`` where ``source`` is ``"real:{symbol}"`` if the
    requested symbol is found, otherwise ``"demo:<found_symbol>"`` when a
    format-compatible demo dataset exists in ``demo_data_dir``.
    """
    paths = _discover(data_dir, symbol, interval)
    source = f"real:{symbol}"
    if not paths and demo_data_dir is not None and demo_data_dir.exists():
        # Fall back to any format-compatible dataset (e.g. BTCUSDT).
        available = sorted(demo_data_dir.glob(f"*-{interval}-*.zip")) + sorted(
            demo_data_dir.glob(f"*-{interval}-*.csv")
        )
        if not available:
            raise FileNotFoundError(
                f"No `{symbol}` data in {data_dir} and no demo data in {demo_data_dir}."
            )
        probe = parse_file_match(available[0])
        if probe is None:
            raise ValueError(f"Cannot parse filename: {available[0]}")
        paths = _discover(demo_data_dir, probe["symbol"], interval)
        source = f"demo:{probe['symbol']}"

    parts = []
    for p in paths:
        regex = parse_file_match(p)
        if regex is None:
            continue
        try:
            parts.append(_read_archive(p, regex["symbol"]))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[data] skip {p.name}: {exc}")

    if not parts:
        raise FileNotFoundError(
            f"No kline data found for `{symbol}` in {data_dir} (demo dir: {demo_data_dir})."
        )

    df = pd.concat(parts, ignore_index=True)
    # Drop corrupt/implausible timestamps before converting.
    # Binance Vision timestamps are UTC epoch milliseconds; sane archives live
    # between 2015-01-01 and 2035-01-01.
    ts = df["open_time"].astype("int64")
    df = df[(ts >= 1_420_070_400_000) & (ts <= 2_051_222_400_000)].copy()
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("open_time")
    return df, source


def parse_file_match(path: Path) -> Optional[dict[str, str]]:
    m = FILE_RE.match(path.name)
    if not m:
        return None
    return {
        "symbol": m.group("symbol"),
        "interval": m.group("interval"),
        "period": m.group("ts"),
    }


def dataset_status(data_dir: Path, symbol: str, interval: str) -> tuple[int, str]:
    """Return (number of files found, short human-readable status)."""
    paths = _discover(data_dir, symbol, interval)
    if not paths:
        return 0, "NOT_FOUND"
    return len(paths), "OK"


def ensure_demo_dataset(config) -> tuple[pd.DataFrame, str]:
    """Convenience wrapper used by the CLI/pipeline.

    Loads the configured symbol when available, otherwise loads the same-format
    demo data and clearly labels it as ``demo``.
    """
    return load_klines(
        config.data_dir,
        config.symbol,
        config.interval,
        demo_data_dir=config.demo_data_dir,
    )
