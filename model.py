
# -*- coding: utf-8 -*-
"""
Model utilities: training, evaluation helpers, and persistence.
Includes robust handling for single-class classification by falling back to
DummyClassifier and recording a clear note in the returned report.
"""

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    r2_score,
)
from sklearn.exceptions import UndefinedMetricWarning

# Suppress AUC warnings when only one class is present in a fold
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def _to_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric when possible; non-convertible become NaN."""
    Xn = pd.DataFrame(index=X.index)
    for c in X.columns:
        Xn[c] = pd.to_numeric(X[c], errors="coerce")
    return Xn


def _align_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Align X and y on index intersection without dropping in caller unexpectedly."""
    Xi, yi = X.align(y, join="inner", axis=0)
    # Replace +/-inf which can appear after indicator construction
    Xi = Xi.replace([np.inf, -np.inf], np.nan)
    return Xi, yi


def fit_classifier(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train a binary classifier with TimeSeries CV summary.
    If y has only one class after cleaning, fall back to DummyClassifier.
    Returns {"model": pipeline, "report": {...}, "features": [...], "cont_cols": [...]}.
    """
    X, y = _align_xy(_to_numeric_df(X), y)

    # mask nan in y; ensure int labels
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    # Identify continuous columns (numeric) for scaling; keep original order
    cont_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    # Simple pipeline shared pieces
    def build_pipe(est):
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if cont_cols else "passthrough"),
            ("clf", est),
        ])

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    classes = np.unique(y)

    # Single-class fallback
    if len(classes) < 2:
        majority = 1 if n_pos >= n_neg else 0
        pipe = build_pipe(DummyClassifier(strategy="constant", constant=majority))
        pipe.fit(X, y)
        report = {
            "cv_auc_mean": np.nan,
            "cv_auc_std": np.nan,
            "cv_acc_mean": float((y == majority).mean()) if len(y) else np.nan,
            "cv_acc_std": 0.0,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "note": "single-class; fallback=DummyClassifier",
        }
        return {
            "model": pipe,
            "report": report,
            "features": list(X.columns),
            "cont_cols": cont_cols,
        }

    # Normal classifier
    pipe = build_pipe(LogisticRegression(solver="lbfgs", max_iter=2000))

    # TimeSeries CV evaluation
    cv = TimeSeriesSplit(n_splits=5)
    aucs, accs = [], []
    for tr_idx, te_idx in cv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(Xtr, ytr)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(Xte)[:, 1]
        elif hasattr(pipe, "decision_function"):
            s = pipe.decision_function(Xte)
            proba = 1.0 / (1.0 + np.exp(-s))
        else:
            # Shouldn't happen with LogisticRegression, but guard anyway
            raw = pipe.predict(Xte)
            proba = 1.0 / (1.0 + np.exp(-raw))

        pred = (proba >= 0.5).astype(int)
        # AUC may be undefined if this fold has single class in yte
        try:
            aucs.append(roc_auc_score(yte, proba))
        except Exception:
            aucs.append(np.nan)
        accs.append(accuracy_score(yte, pred))

    # Fit on full data to produce final model
    pipe.fit(X, y)

    report = {
        "cv_auc_mean": float(np.nanmean(aucs)) if len(aucs) else np.nan,
        "cv_auc_std": float(np.nanstd(aucs)) if len(aucs) else np.nan,
        "cv_acc_mean": float(np.nanmean(accs)) if len(accs) else np.nan,
        "cv_acc_std": float(np.nanstd(accs)) if len(accs) else np.nan,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }

    return {
        "model": pipe,
        "report": report,
        "features": list(X.columns),
        "cont_cols": cont_cols,
    }


def fit_regressor(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train a simple regression model with TimeSeries CV summary (R^2).
    Returns {"model": pipeline, "report": {...}, "features": [...], "cont_cols": [...]}
    """
    X, y = _align_xy(_to_numeric_df(X), y)

    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    cont_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler() if cont_cols else "passthrough"),
        ("reg", LinearRegression()),
    ])

    cv = TimeSeriesSplit(n_splits=5)
    r2s = []
    for tr_idx, te_idx in cv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        try:
            r2s.append(r2_score(yte, pred))
        except Exception:
            r2s.append(np.nan)

    pipe.fit(X, y)

    report = {
        "cv_r2_mean": float(np.nanmean(r2s)) if len(r2s) else np.nan,
        "cv_r2_std": float(np.nanstd(r2s)) if len(r2s) else np.nan,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }

    return {
        "model": pipe,
        "report": report,
        "features": list(X.columns),
        "cont_cols": cont_cols,
    }


def save_model(obj: Dict[str, Any], out_path: str) -> None:
    """Persist trained object to disk."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
# === inference helpers (for predict_model.py) ===
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def _prepare_infer_X(obj, X: pd.DataFrame) -> pd.DataFrame:
    feats = obj.get("features", list(X.columns))
    cont_cols = obj.get("cont_cols", [])
    mdl = obj["model"]
    scaler = mdl.named_steps.get("scaler", None) if hasattr(mdl, "named_steps") else None
    if scaler == "passthrough":
        scaler = None
    Xb = X.copy()
    for c in feats:
        if c not in Xb.columns:
            Xb[c] = 0.0
    Xb = Xb[feats]
    if scaler is not None and cont_cols:
        Xb.loc[:, cont_cols] = scaler.transform(Xb[cont_cols])
    return Xb

def predict_latest_clf(obj, X: pd.DataFrame):
    Xb = _prepare_infer_X(obj, X)
    xb_last = Xb.iloc[[-1]]
    mdl = obj["model"]
    if hasattr(mdl, "predict_proba"):
        proba = float(mdl.predict_proba(xb_last)[:, 1][0])
    elif hasattr(mdl, "decision_function"):
        s = float(mdl.decision_function(xb_last)[0])
        proba = 1.0 / (1.0 + np.exp(-s))
    else:
        r = float(mdl.predict(xb_last)[0])
        proba = 1.0 / (1.0 + np.exp(-r))
    score = int(proba * 100)
    return proba, score

def predict_latest_reg(obj, X: pd.DataFrame):
    Xb = _prepare_infer_X(obj, X)
    xb_last = Xb.iloc[[-1]]
    yhat = float(obj["model"].predict(xb_last)[0])  # 未来收益预测（比例）
    score = int(np.clip(round(50 + yhat * 200), 0, 100))
    return yhat, score
