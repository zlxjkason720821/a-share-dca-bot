# -*- coding: utf-8 -*-
"""
tools/predict_model.py — 自动收集标的 + 逐票或池化模型推理
兼容训练产物（features / cont_cols / DummyClassifier）。
"""
import os, json, glob, argparse
from datetime import datetime

import numpy as np
import pandas as pd

from data_loader import get_price_df
from features import make_features
from model import load_model


def _collect_symbols(args):
    if args.symbols is not None:
        return list(args.symbols)
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    if args.symbols_from_data:
        syms = []
        for p in glob.glob(os.path.join("data", "*.csv")):
            base = os.path.basename(p)
            sym = os.path.splitext(base)[0]
            syms.append(sym)
        return sorted(syms)
    raise ValueError("请用 --symbols / --symbols_file / --symbols_from_data 三选一提供标的列表")


def _prepare_infer_X(obj, X: pd.DataFrame) -> pd.DataFrame:
    """按训练时的列顺序对齐，不再手动缩放。"""
    feats = obj.get("features", list(X.columns))
    Xb = X.copy()

    for c in feats:
        if c not in Xb.columns:
            Xb[c] = 0.0
    Xb = Xb[feats].astype(float)

    return Xb


def _predict_last(obj, X: pd.DataFrame):
    """
    返回 (raw, score, mode)
    - 分类器: raw=概率[0,1], score=0~100
    - 回归器: raw=未来收益比例, score=映射到0~100
    """
    Xb = _prepare_infer_X(obj, X)
    if len(Xb) == 0:
        raise RuntimeError("no feature rows after alignment")
    xb_last = Xb.iloc[[-1]]
    mdl = obj["model"]

    is_clf = hasattr(mdl, "predict_proba") or hasattr(mdl, "decision_function")
    if is_clf:
        if hasattr(mdl, "predict_proba"):
            proba = float(mdl.predict_proba(xb_last)[:, 1][0])
        elif hasattr(mdl, "decision_function"):
            s = float(mdl.decision_function(xb_last)[0])
            proba = 1.0 / (1.0 + np.exp(-s))
        else:
            r = float(mdl.predict(xb_last)[0])
            proba = 1.0 / (1.0 + np.exp(-r))
        # 防止极端0/100，做个轻微夹逼
        proba = np.clip(proba, 0.01, 0.99)
        score = int(round(proba * 100))
        return proba, score, "clf"
    else:
        yhat = float(mdl.predict(xb_last)[0])
        score = int(np.clip(round(50 + yhat * 200), 0, 100))
        return yhat, score, "reg"


def _model_path(args, symbol: str):
    if args.pooled:
        cand = [
            os.path.join(args.model_dir, f"{args.model_name}.pkl"),
            os.path.join(args.model_dir, f"{args.model_name}")
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        return cand[0]
    else:
        return os.path.join(args.model_dir, f"{args.model_name}_{symbol}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--symbols_from_data", action="store_true")
    ap.add_argument("--symbols_file", type=str, default=None)

    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pooled", action="store_true")

    ap.add_argument("--out_json", default=os.path.join("logs", "model_scores.json"))
    ap.add_argument("--out_csv", default=os.path.join("logs", "model_scores.csv"))
    ap.add_argument("--timeout", type=int, default=10)

    args = ap.parse_args()
    symbols = _collect_symbols(args)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    pooled_obj = None
    if args.pooled:
        p = _model_path(args, symbol=None)
        if os.path.exists(p):
            pooled_obj = load_model(p)
        else:
            print(f"[WARN] pooled model not found: {p}")

    out_rows = []
    for sym in symbols:
        try:
            obj = pooled_obj if args.pooled else (
                load_model(_model_path(args, sym))
                if os.path.exists(_model_path(args, sym)) else None
            )
            if obj is None:
                out_rows.append({"symbol": sym, "raw": 0.0, "score": 50, "mode": None, "err": "model not found"})
                continue

            df = get_price_df(sym, timeout_sec=args.timeout, use_cache=True)
            X = make_features(df)

            raw, score, mode = _predict_last(obj, X)
            out_rows.append({"symbol": sym, "raw": raw, "score": score, "mode": mode, "err": None})
        except Exception as e:
            out_rows.append({"symbol": sym, "raw": 0.0, "score": 50, "mode": None, "err": f"predict error: {e}"})

    result = {"time": datetime.now().isoformat(), "details": out_rows}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
