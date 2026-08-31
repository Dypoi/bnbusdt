# BTCUSDT 5m Research Results — January–June 2026

> **Important naming note:** “H1 2026” in this document means **first half of
> the year 2026 (January–June)**, *not* a 1-hour timeframe. The model and the
> backtest are executed entirely on the **M5 (5-minute) candle**:
> `interval = "5m"`, `max_hold_bars = 12 bars = 60 minutes`.

Run date: 2026-08-31  
Dataset: `Dataset_BTCUSDT/` (Binance Vision 5m, 2021-01 → 2026-06, real data)  
Symbol / interval: **BTCUSDT / M5 (5m)**

## What was run

1. **Full 6-fold expanding-window walk-forward** on the whole archive
   (577,803 bars), using a **dual long/short LightGBM** over TP/SL scalp events.
2. **112 backward-looking features** (added advanced order-flow + regime block):
   - taker-aggression pressure, cumulative delta, flow-vs-price correlation,
   - average trade size, trade-count z-scores,
   - volume/quote rank, realized-vol/ATR rank, range expansion,
   - ADX/DX, Donchian position, CMF, MFI, wick skew etc.
3. **Detailed parameter sweep** (270 backtest configs) over
   threshold × margin × cooldown × notional cap × direction filter.
4. **Strict forward test**: configs were selected only on OOS predictions
   **up to 2025-12-31**, then applied to **2026-01-01 → 2026-06-30**.

## Model quality (6 folds, out-of-sample)

| Metric | Value |
|---|---|
| Folds | 6 |
| Mean AUC | 0.5411 |
| AUC long | 0.5436 |
| AUC short | 0.5386 |
| Acc long / short | 68.4% / 67.4% |

Top contributing features were regime / order-flow oriented:
`f_dx_44`, `f_time_sec`, `f_hour`, `f_rvol_30`, `f_natr_21`, `f_adx_14`,
`f_donchian_pos_60`, `f_flow_price_corr_20`, `f_cdelta_slope_60`,
`f_cdelta_z_60`, `f_wick_skew_ema`.

## Strict forward-test backtest: Jan–Jun 2026

The config was chosen **without looking at 2026 data**:
- threshold `0.60`, margin `0.00`, cooldown `6`, notional cap `1.0`, `long-only`
- TP `50 bps`, SL `20 bps`, max hold `12 bars`, fee `4 bps/side`, slippage `1 bps/side`

| Metric | Value |
|---|---|
| Trades | 59 |
| Total return | **+3.82%** |
| Final equity (start 10,000) | 10,381.77 |
| Win rate | 50.85% |
| Profit factor | 1.446 |
| Max drawdown | −1.48% |
| Sharpe | 2.38 |
| Exits | TP 29 / SL 29 / TIME 1 |

This is a **modestly positive, cost-aware, out-of-sample** result for H1 2026.

## Alternative high-conviction scenario (same strict selection basis)

Using `threshold 0.70, long-only, cooldown 6, cap 3x` — the high-PF cluster found
in the ≤2025-12-31 sweep — 2026 H1 looks stronger, but with far fewer trades:

| Metric | Value |
|---|---|
| Trades | 17 |
| Total return | **+8.54%** |
| Win rate | 64.71% |
| Profit factor | 2.636 |
| Max drawdown | −3.32% |
| Sharpe | 3.02 |
| Exits | TP 11 / SL 6 |

## Adaptive MFE/MAE TP/SL (new, same strict-forward basis)

As a second step we built **MFE/MAE labels** (max favorable / adverse excursion,
in basis points, over the next 12 M5 bars, using the next bar's open as entry to
avoid same-bar look-ahead), trained **quantile-regression + directional
confidence** models, and turned predicted MFE/MAE into a **per-bar TP/SL**:
`tp = clip(1.0 × pred_MFE, min/max)`, `sl = clip(0.8 × pred_MAE, min/max)`,
with an optional `atr_weight` term and an `rr_min` risk-reward filter.
The adaptive config was swept **only on OOS predictions ≤ 2025-12-31**, then
tested strict-forward on Jan–Jun 2026 against two fixed TP/SL baselines using
the **same signal model**.

| Config | Trades | Return | Win% | PF | MaxDD | Sharpe |
|---|---|---|---|---|---|---|
| **Adaptive** (0.65, cool 6, cap 1, both, tp×1.0, sl×0.8, rr≥1.0) | 27 | **+5.75%** | 62.96% | **3.22** | **−1.03%** | **3.73** |
| A_fixed (0.60/6/1.0, 50/20) | 59 | +3.82% | 50.85% | 1.45 | −1.49% | 2.39 |
| B_fixed (0.70/6/3.0, 50/20) | 14 | +6.85% | 64.29% | 2.57 | −2.25% | 2.77 |

Adaptive H1 2026 trades: TP 12 / SL 9 / TIME 6, avg TP 64.1 bps, avg SL 26.3 bps.
On the selection window the chosen adaptive config traded 754 times (+11.6%,
PF 1.08, MaxDD −13.0%), so its edge there is modest; the H1 result is a single
6-month forward sample and should be treated as one regime, not proof.

> **MFE/MAE label check:** `mfe_short_bps` equals `mae_long_bps` and vice-versa.
> This is **expected**, not a bug: for a short entry, "maximum favorable" is the
> same price move that is "maximum adverse" for a long (entry → low/entry → high).

### Full OOS sanity check (same selected adaptive config, 2021-11 → 2026-06)

Running the same adaptive config over the whole out-of-sample prediction stack
(all walk-forward test folds) vs fixed A/B on the same prediction base:

| Config | Trades | Return | Win% | PF | MaxDD | Sharpe |
|---|---|---|---|---|---|---|
| Adaptive | 781 | +18.01% | 47.12% | 1.124 | −13.01% | 0.671 |
| A_fixed | 1,229 | −20.27% | 39.30% | 0.883 | −31.17% | −0.898 |
| B_fixed | 355 | +29.07% | 45.07% | 1.161 | −11.13% | 0.679 |

Adaptive is profitable over the full OOS period and beats fixed A, but its edge
is thin (PF 1.12, Sharpe 0.67, MaxDD −13%) and it does **not** beat the
high-conviction fixed B over the whole history (+29%, PF 1.16) — it only wins on
the narrow Jan–Jun 2026 forward window.

## Jesse TEMA Trend-Following comparison (Jan–Jun 2026)

We also replicated the user's provided Jesse `TemaTrendFollowing` strategy on the
same M5 data/window: limit entry 1 ATR from signal close (valid one bar),
SL 4 ATR / TP 3 ATR (fill-bar ATR), no time-stop, ADX>40 + CMO(±40) + TEMA(10/80)
on M5 + TEMA(20/70) on the last-completed 4h candle.

Market context: BTC was **−31.23%** over Jan–Jun 2026 (buy & hold).

| Strategy | Trades | Return | Win% | PF | MaxDD | Sharpe |
|---|---|---|---|---|---|---|
| TEMA as-authored (risk 3% ×3) | 133 | −71.91% | 58.65% | 0.84 | −84.26% | −1.28 |
| TEMA normalised (risk 1%, 1×) | 133 | −0.92% | 58.65% | 0.98 | −10.16% | −0.10 |
| Adaptive MFE/MAE | 27 | +5.75% | 62.96% | 3.22 | −1.03% | 3.73 |
| A_fixed | 59 | +3.82% | 50.85% | 1.45 | −1.49% | 2.39 |
| B_fixed | 14 | +6.85% | 64.29% | 2.57 | −2.25% | 2.77 |

The TEMA strategy is **not profitable** here: as-authored the `qty*3` + 3% risk
creates ~8–13× notional and wipes the account (−71.9%); even at normalised risk
it only reaches −0.92% (PF 0.98). Its 58.65% win rate is just above the ~57.1%
breakeven for a 3:4 TP:SL ratio, so the edge does not survive M5 fees. It is
also a fair test despite the falling market: it traded more shorts (76) than longs (57).
See `output/BTCUSDT/COMPARISON_2026H1.html` / `.md`.

## TEMA sensitivity + forensic audit (5m & 15m, H1 2026)

A 36-config sweep (timeframe 5m/15m × ADX 25/30 × CMO 20/30 × TP:SL 1.5/2.0/3.0
× sizing normalised/as-authored) plus a forensic audit produced:

- **Only 1 of 18 normalised configs was profitable** (median normalised return −8.14%).
  That one is NOT the requested 2:1 — it is the original 15m TEMA (TP/SL = 3:4,
  ADX>40, CMO>±40): +2.30%, PF 1.135, MaxDD −8.30%, but **t-stat 0.41** and
  bootstrap 5th percentile −7.83% → not statistically significant.
- **All TP:SL 2:1 configs on 15m were negative** (best normalised −10.53%,
  worst −25.54%; authored much worse due to leverage). Reason: 2:1 needs 33.33%
  breakeven win-rate, but actual win-rate only 25.7%–31.3%.
- No-lookahead checks passed (indicator prefix recomputation = 0 diff; 4h TEMA
  used only completed 4h candles). Audit is documented in
  `output/BTCUSDT/tema_sensitivity_2026H1/AUDIT.html` / `.md`.

## Honest caveats

- The 6-fold model AUC (`~0.54`) is only a small edge. At 5-minute frequency,
  the edge is close to the cost floor, so a config can flip from positive to
  negative in a different regime.
- The **strict forward** 2026 H1 result is positive (+3.82%), but the same
  selected config across the **entire OOS history** is negative
  (−30% over 2021–2026 for that exact config). In other words, the strategy is
  **regime-dependent**; it did well in the last 6 months on a high-conviction
  long subset, but not consistently across all market regimes.
- The 0.70 high-conviction variant is even more regime-sensitive (13 prior OOS
  trades before 2026).
- These are research results, **not financial advice**, and should be confirmed
  on live paper trading before any capital is committed.

## Files

| File | Description |
|---|---|
| `output/BTCUSDT/report_2026H1.json` | Strict-forward selected config + H1 metrics |
| `output/BTCUSDT/2026H1/report.html` | H1 backtest report + equity curve |
| `output/BTCUSDT/2026H1_highconf/report.html` | High-conviction alternative |
| `output/BTCUSDT/research_sweep.csv` | 270-config grid |
| `output/BTCUSDT/predictions_oos.csv` | Out-of-sample predictions |
| `output/BTCUSDT/adaptive_sweep.csv` | Adaptive TP/SL config sweep (512 configs) |
| `output/BTCUSDT/adaptive_predictions.csv` | Adaptive MFE/MAE + confidence predictions |
| `output/BTCUSDT/adaptive_2026H1.json` | Adaptive vs fixed strict-forward comparison |
| `output/BTCUSDT/adaptive_2026H1/report.html` | Adaptive H1 backtest report |
| `output/BTCUSDT/adaptive_2026H1/comparison.json` | Same comparison in report dir |
| `output/BTCUSDT/adaptive_full_oos.json` | Full-OOS adaptive vs fixed A/B (2021-11 → 2026-06) |
| `output/BTCUSDT/adaptive_2026H1/COMPARISON.md` | Human-readable comparison + full-OOS sanity check |
| `output/BTCUSDT/COMPARISON_2026H1.html` / `.md` | Full H1 2026 comparison incl. Jesse TEMA strategy |
| `output/BTCUSDT/tema_trend_2026H1/report.json` | TEMA as-authored + normalised results |
| `output/BTCUSDT/tema_trend_2026H1/trades_*.csv` | TEMA trade logs |
| `output/BTCUSDT/tema_sensitivity_2026H1/AUDIT.html` / `.md` | TEMA sensitivity + forensic audit |
| `output/BTCUSDT/tema_sensitivity_2026H1/sensitivity_grid.csv` | 36-config sweep |

## Reproduce

```bash
# 6-fold + strict forward H1 2026 report
python scripts/research_full.py --folds 6 --n-est 350 --sweep-rows 120000 --select-end 2025-12-31

# same but select config on the whole OOS window (weaker, more optimistic)
python scripts/research_full.py --folds 6 --n-est 350 --sweep-rows 120000

# adaptive MFE/MAE comparison (6-fold, sweep ≤2025-12-31, strict forward 2026 H1)
python scripts/adaptive_tp_sl.py --folds 6 --n-est 300 \
  --select-end 2025-12-31 --start 2026-01-01 --end 2026-06-30 --min-trades-adapt 25

# Jesse TEMA trend-following comparison (H1 2026, as-authored + normalised)
python scripts/backtest_tema_trend.py

# TEMA sensitivity + forensic audit (ADX/CMO/TP:SL grid, 5m & 15m)
python scripts/tema_sensitivity.py
```
