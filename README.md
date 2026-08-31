# BTCUSDT Scalping Model (5m)

A complete, reproducible research pipeline for a **5-minute BTC/USDT scalping
model**, built with a quant/scalping mindset and a full-stack/data-engineering
workflow:

```
Dataset → Feature Engineering → Direction Labels → Walk-Forward LightGBM → Realistic Backtest → Report/Plots
```

The repository **already contains real BTCUSDT historical data** in the standard
Binance Vision format (`Dataset_BTCUSDT/`, monthly ZIPs from 2021-01 to
2026-06).

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fast validation run (3 folds, lighter model)
python main.py --quick

# Full run (6 folds, 600 trees)
python main.py --folds 6

# Parameter overrides
python main.py --tp 30 --sl 25 --threshold 0.60 --risk 0.01
python main.py --folds 8 --scalp-bars 6 --taker-fee 4
```

Reports land in `output/BTCUSDT/`:

| File | Description |
|---|---|
| `report.html` | Interactive summary: metrics, folds, features, config |
| `summary.json` | Machine-readable results |
| `equity_curve.png` | Out-of-sample equity curve + drawdown |
| `predictions.png` | Last 2000 bars with P(up) / P(down) |

---

## What the model does

1. **Data** – loads Binance Vision klines (open, high, low, close, volume,
   quote volume, trades, taker buy volume/quote). Timestamps are normalized to
   milliseconds UTC (Binance Vision changed to microseconds in newer years; the
   parser handles both).
2. **Features** – momentum, rolling/EMA distances, RSI, MACD, ATR/NATR,
   candle shape, wick fractions, volume ratios, taker flow imbalance, OBV slope,
   trend context and UTC session features. All features are strictly
   backward-looking, so they contain no leakage.
3. **Labels** –
   - `label_up` / `label_down`: next-bar direction (the main target),
   - `label_scalp_up` / `label_scalp_down`: resolved TP/SL outcome within
     `scalp_bars` bars (both-touched bars are excluded as ambiguous).
4. **Model** – expanding-window **LightGBM** time-series walk-forward. It never
   shuffles train/test, and every test fold is strictly future data relative to
   its training window.
5. **Backtest** – event-driven:
   - signal on close of bar `t`, entry at the **open of bar `t+1`**,
   - taker fees + per-side slippage,
   - risk-based position sizing (`risk_pct` of equity to the stop),
   - spot-like notional cap (no leverage by default),
   - conservative SL-first rule when a bar touches both TP and SL,
   - time-stop at `max_hold_bars`,
   - optional `cooldown_bars` to avoid churning after exits.

---

## Data

The repo already has real BTCUSDT data:

```
Dataset_BTCUSDT/BTCUSDT-5m-YYYY-MM.zip
```

If you need to refresh or extend it, use `scripts/download_binance_data.py`:

```bash
python scripts/download_binance_data.py --symbol BTCUSDT --start 2021-01 --end 2026-06
```

The parser will read any Binance Vision symbol directory automatically when the
config symbol changes (e.g. from BTCUSDT to another Binance Vision symbol).

---

## Project structure

```
btc_scalper/
├── config.py        # all tunable parameters
├── data.py          # Binance Vision ZIP parser + ms/µs timestamp handling
├── features.py      # feature engineering
├── labels.py        # label generation (direction + TP/SL scalp labels)
├── model.py         # expanding-window LightGBM walk-forward
├── backtest.py      # realistic event-driven backtester
├── evaluate.py      # plots + HTML/JSON report
└── pipeline.py      # end-to-end orchestration
scripts/download_binance_data.py
scripts/experiment.py
main.py              # CLI entry point
requirements.txt
Dataset_BTCUSDT/     # real BTCUSDT Binance Vision monthly archives
```

---

## Walk-forward / anti overfitting

The framework is deliberately conservative:

- no random train/test split,
- no cross-validation shuffle,
- no scaling or transforms fitted on the test set,
- a value in the test window can never influence its own inputs,
- TP/SL labels that are ambiguous are marked `NaN` and excluded from training,
- the backtest charges Binance-style taker fees and assumes slippage,
- the default model uses only out-of-sample predictions for the equity report.

---

## Typical parameter notes

| Parameter | Default | Why |
|---|---|---|
| `tp_bps` | 50.0 | ~0.50% target |
| `sl_bps` | 20.0 | ~0.20% stop |
| `scalp_bars` | 12 | hold at most 12 × 5m = 60m |
| `taker_fee_bps` | 4.0 | Binance Futures taker fee |
| `slippage_bps` | 1.0 | per side |
| `probability_threshold` | 0.60 | probability must be at least 0.60 |
| `probability_margin` | 0.12 | and |P(up)−P(down)| must be ≥ 0.12 |
| `cooldown_bars` | 2 | wait 2 bars after an exit |
| `risk_pct` | 0.01 | 1% equity risked per trade |

You should **optimize these with your own walk-forward protocol** — never tune
directly on the test fold that is used to report performance.

`scripts/experiment.py` sweeps threshold / margin / cooldown while reusing one
model training run (train once, backtest many times):

```bash
python scripts/experiment.py --rows 80000 --folds 3
```

---

## Honest baseline note

The default pipeline charges costs and uses only out-of-sample predictions, so
a baseline with small edge (AUC ~0.52–0.53) typically loses money after
5-minute taker fees + slippage. That is intentional: the framework is a bridge
between pure ML metrics and a cost-aware live-ready backtest. The useful next
steps are finding higher-conviction subsets (higher threshold/margin), better
features/labels, cheaper execution assumptions, or a higher TP/SL regime that
fits the noise level of 5-minute BTC bars.

---

## License / disclaimer

Research and educational use only. Not financial advice. Scalping with leverage
carries substantial risk of loss; always test on out-of-sample and paper-trading
data before committing capital.
