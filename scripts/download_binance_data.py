#!/usr/bin/env python3
"""Download BTCUSDT kline data from the Binance Vision public archive.

The data format matches the existing ``Dataset_BTCUSDT/`` ZIP files produced by
``data.binance.vision``:

    open_time, open, high, low, close, volume,
    close_time, quote_volume, trades,
    taker_base, taker_quote, ignore

Usage
-----
    python scripts/download_binance_data.py --symbol BTCUSDT --start 2025-01 --end 2025-06
    python scripts/download_binance_data.py --interval 5m --symbol BTCUSDT --all

Notes
-----
* If your sandbox/network cannot reach ``data.binance.vision`` (common in some
  restricted environments) the script tells you which host failed; you can then
  put any same-format monthly ZIP under ``Dataset_<SYMBOL>/`` and the pipeline
  will use it automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

BASE_MONTHLY = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "Dataset_BTCUSDT"


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    out = []
    while s <= e:
        out.append((s.year, s.month))
        y, m = (s.year + 1, 1) if s.month == 12 else (s.year, s.month + 1)
        s = datetime(y, m, 1)
    return out


def download_month(symbol: str, interval: str, year: int, month: int, out_dir: Path, retries: int = 3) -> Path:
    url = BASE_MONTHLY.format(symbol=symbol, interval=interval, year=year, month=month)
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[skip] {filename} already exists")
        return dest

    for attempt in range(1, retries + 1):
        try:
            print(f"[get ] {url} (attempt {attempt})")
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 404:
                print(f"[404 ] {filename} not found (maybe the month is not archived yet)")
                return dest
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(dest) as zf:
                zf.testzip()
            print(f"[ok  ] {filename} ({dest.stat().st_size/1024:.0f} KiB)")
            return dest
        except Exception as exc:
            print(f"[err ] attempt {attempt}: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)
    print(f"[fail] {filename}: could not download from data.binance.vision")
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download Binance Vision kline archives")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="5m")
    p.add_argument("--start", default="2021-01", help="YYYY-MM inclusive")
    p.add_argument("--end", default=None, help="YYYY-MM inclusive; default=current month")
    p.add_argument("--all", action="store_true", help="download every month from start")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    if args.end is None:
        args.end = datetime.utcnow().strftime("%Y-%m")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start, args.end)
    print(f"Downloading {args.symbol} {args.interval} for {len(months)} months -> {out_dir}")

    ok = 0
    for year, month in months:
        path = download_month(args.symbol, args.interval, year, month, out_dir)
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
    print(f"\nDone: {ok}/{len(months)} months available.")
    if ok == 0:
        print(
            "Reachability check: if this environment blocks binance.vision, "
            "copy the monthly ZIPs from your own Binance Vision download into "
            f"{out_dir}/ and re-run the pipeline."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
