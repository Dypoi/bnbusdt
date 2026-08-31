"""Walk-forward LightGBM training for the BTCUSDT scalping model.

The model is trained on expanding time windows (no shuffling, no random CV)
and evaluated on strictly out-of-sample future data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import feature_columns


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame                     # out-of-sample predictions
    metrics: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    trained_folds: int = 0


def _roc_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _topk_precision(y: np.ndarray, p: np.ndarray, top_frac: float = 0.10) -> Optional[float]:
    if len(y) == 0:
        return None
    top_n = max(1, int(len(y) * top_frac))
    idx = np.argsort(-p)[:top_n]
    return float(np.mean(y[idx] == 1))


def _base_rate(y: np.ndarray) -> float:
    return float(np.mean(y == 1)) if len(y) else float("nan")


def _build_folds(index: pd.Index, n_folds: int, test_frac: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (train_start, test_start, test_end) tuples per fold."""
    n = len(index)
    n_test = max(1, int(n * test_frac))
    # Adjust so the first fold has some minimum train history.
    min_train = max(2_000, int(n * 0.25))
    folds = []
    for i in range(n_folds):
        test_end = n - n_test * (n_folds - 1 - i)
        test_start = max(min_train, test_end - n_test)
        if test_start <= 0 or test_end > n or test_end - test_start < 100:
            continue
        folds.append((index[test_start], index[test_start], index[test_end - 1]))
    return folds


def _train_fold(
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    Xva: pd.DataFrame,
    yva: pd.Series,
    Xte: pd.DataFrame,
    yte: pd.Series,
    params: dict,
    training_rounds: int,
) -> tuple[np.ndarray, lgb.Booster]:
    train = lgb.Dataset(Xtr, ytr)
    valid = lgb.Dataset(Xva, yva, reference=train)
    booster = lgb.train(
        params,
        train,
        num_boost_round=training_rounds,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else None
    preds = booster.predict(Xte, num_iteration=best_iter)
    return preds.astype(float), booster


def run_walk_forward(
    df: pd.DataFrame,
    cfg,
    label_col: str = "label_up",
    return_probabilities: tuple[str, str] = ("label_up", "label_down"),
) -> WalkForwardResult:
    """Run expanding-window LightGBM classification.

    The returned predictions contain one row per out-of-sample bar with:
    ``open_time``, ``y``, ``prob_up`` and ``prob_down``.
    """
    data = df.iloc[cfg.feature_start:].copy()
    cols = feature_columns(data)
    if not cols:
        raise ValueError("No feature columns found; call add_features() first.")
    data = data.dropna(subset=[label_col])
    if len(data) < 5_000:
        raise ValueError(f"Not enough labeled rows for walk-forward ({len(data)}).")

    X = data[cols]
    y = data[label_col].astype(float)

    folds = _build_folds(data.index, cfg.folds, cfg.test_frac)
    if not folds:
        raise ValueError("Walk-forward folds could not be built; reduce folds/test_frac.")

    predictions = []
    metrics = []
    importance = []
    booster_folds: list[lgb.Booster] = []

    params = dict(cfg.lgb_params)
    n_rounds = params.pop("n_estimators", 600)
    for fold_i, (_train_start, t_test_start, t_test_end) in enumerate(folds):
        mask_tr = data.index < t_test_start
        mask_te = (data.index >= t_test_start) & (data.index <= t_test_end)
        tr_idx = data.index[mask_tr]
        te_idx = data.index[mask_te]
        if len(tr_idx) < 2_000 or len(te_idx) < 200:
            continue
        # Validation = last valid_frac of the train set (time-ordered).
        n_va = max(500, int(len(tr_idx) * cfg.valid_frac))
        va_idx = tr_idx[-n_va:]

        Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
        Xva, yva = X.loc[va_idx], y.loc[va_idx]
        Xte, yte = X.loc[te_idx], y.loc[te_idx]

        preds, booster = _train_fold(Xtr, ytr, Xva, yva, Xte, yte, params, n_rounds)
        booster_folds.append(booster)

        auc = _roc_auc(yte.to_numpy(), preds)
        top10 = _topk_precision(yte.to_numpy(), preds, 0.10)
        acc = float(np.mean((preds >= 0.5).astype(int) == yte.to_numpy()))
        metrics.append(
            {
                "fold": fold_i + 1,
                "test_start": str(te_idx[0]),
                "test_end": str(te_idx[-1]),
                "train_rows": int(len(tr_idx)),
                "test_rows": int(len(te_idx)),
                "base_rate": _base_rate(yte.to_numpy()),
                "auc": auc,
                "acc": acc,
                "prec@10%": top10,
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "open_time": te_idx,
                    "y": yte.to_numpy(),
                    "prob_up": preds,
                    "prob_down": 1.0 - preds,
                }
            )
        )
        imp = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance(importance_type="gain"),
            }
        )
        importance.append(imp)

    if not predictions:
        raise ValueError("No folds were trained; check data size/configuration.")

    preds_df = pd.concat(predictions, ignore_index=False).sort_values("open_time").reset_index(drop=True)

    importance_avg = (
        pd.concat(importance, ignore_index=True)
        .groupby("feature", as_index=False)["gain"]
        .mean()
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    aggregates = _aggregate_metrics(metrics)

    return WalkForwardResult(
        predictions=preds_df,
        metrics=metrics,
        aggregate=aggregates,
        feature_importance=importance_avg,
        trained_folds=len(metrics),
    )


def run_walk_forward_dual(
    df: pd.DataFrame,
    cfg,
    long_label: str = "label_scalp_up",
    short_label: str = "label_scalp_down",
) -> WalkForwardResult:
    """Train a separate LightGBM model for long and short scalp outcomes.

    * ``long model`` learns P(TP before SL) for a long entry.
    * ``short model`` learns P(TP before SL) for a short entry.

    This is the more cost-relevant target for TP/SL scalping than plain
    next-bar direction. The returned predictions expose ``prob_up`` /
    ``prob_down`` as those two TP-event probabilities, which the backtester
    uses directly.
    """
    data = df.iloc[cfg.feature_start:].copy()
    cols = feature_columns(data)
    if not cols:
        raise ValueError("No feature columns found; call add_features() first.")
    data = data.dropna(subset=[long_label, short_label])
    if len(data) < 5_000:
        raise ValueError(f"Not enough labeled rows for dual scalp walk-forward ({len(data)}).")

    X = data[cols]
    y_long = data[long_label].astype(float)
    y_short = data[short_label].astype(float)

    folds = _build_folds(data.index, cfg.folds, cfg.test_frac)
    if not folds:
        raise ValueError("Walk-forward folds could not be built; reduce folds/test_frac.")

    predictions = []
    metrics = []
    importance = []

    params = dict(cfg.lgb_params)
    n_rounds = params.pop("n_estimators", 600)
    for fold_i, (_train_start, t_test_start, t_test_end) in enumerate(folds):
        mask_tr = data.index < t_test_start
        mask_te = (data.index >= t_test_start) & (data.index <= t_test_end)
        tr_idx = data.index[mask_tr]
        te_idx = data.index[mask_te]
        if len(tr_idx) < 2_000 or len(te_idx) < 200:
            continue
        n_va = max(500, int(len(tr_idx) * cfg.valid_frac))
        va_idx = tr_idx[-n_va:]

        results = {}
        for direction, y in (("up", y_long), ("down", y_short)):
            Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
            Xva, yva = X.loc[va_idx], y.loc[va_idx]
            Xte, yte = X.loc[te_idx], y.loc[te_idx]
            preds, booster = _train_fold(Xtr, ytr, Xva, yva, Xte, yte, params, n_rounds)
            results[direction] = (
                preds.astype(float),
                booster,
                yte.to_numpy(),
                _roc_auc(yte.to_numpy(), preds),
            )
            imp = pd.DataFrame(
                {
                    "feature": booster.feature_name(),
                    "gain": booster.feature_importance(importance_type="gain"),
                }
            )
            importance.append(imp)

        p_up, _, y_up_te, auc_up = results["up"]
        p_down, _, y_down_te, auc_down = results["down"]
        acc_up = float(np.mean((p_up >= 0.5).astype(int) == y_up_te))
        acc_down = float(np.mean((p_down >= 0.5).astype(int) == y_down_te))
        metrics.append(
            {
                "fold": fold_i + 1,
                "test_start": str(te_idx[0]),
                "test_end": str(te_idx[-1]),
                "test_rows": int(len(te_idx)),
                "base_rate_long": float(y_up_te.mean()),
                "base_rate_short": float(y_down_te.mean()),
                "auc_long": auc_up,
                "auc_short": auc_down,
                "acc_long": acc_up,
                "acc_short": acc_down,
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "open_time": te_idx,
                    "y_up": y_up_te,
                    "y_down": y_down_te,
                    "prob_up": p_up,
                    "prob_down": p_down,
                }
            )
        )

    if not predictions:
        raise ValueError("No folds were trained; check data size/configuration.")

    preds_df = pd.concat(predictions, ignore_index=False).sort_values("open_time").reset_index(drop=True)

    importance_avg = (
        pd.concat(importance, ignore_index=True)
        .groupby("feature", as_index=False)["gain"]
        .mean()
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    aucs_long = [m["auc_long"] for m in metrics if m.get("auc_long") is not None]
    aucs_short = [m["auc_short"] for m in metrics if m.get("auc_short") is not None]

    aggregate = {
        "folds": len(metrics),
        "mean_auc": float(np.mean(aucs_long + aucs_short)) if (aucs_long + aucs_short) else None,
        "mean_auc_long": float(np.mean(aucs_long)) if aucs_long else None,
        "mean_auc_short": float(np.mean(aucs_short)) if aucs_short else None,
        "mean_acc_long": float(np.mean([m["acc_long"] for m in metrics])),
        "mean_acc_short": float(np.mean([m["acc_short"] for m in metrics])),
    }

    return WalkForwardResult(
        predictions=preds_df,
        metrics=metrics,
        aggregate=aggregate,
        feature_importance=importance_avg,
        trained_folds=len(metrics),
    )


def _aggregate_metrics(fold_metrics: list[dict]) -> dict:
    aucs = [m["auc"] for m in fold_metrics if m.get("auc") is not None]
    accs = [m["acc"] for m in fold_metrics if m.get("acc") is not None]
    tops = [m["prec@10%"] for m in fold_metrics if m.get("prec@10%") is not None]
    return {
        "folds": len(fold_metrics),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "mean_prec@10%": float(np.mean(tops)) if tops else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }
