# -*- coding: utf-8 -*-
"""
滚动验证（walk-forward）：时间序列切片，多窗评估AUC/F1/Precision@TopK，以及基于阈值/TopK的
简易收益回测（按close买入、K日后卖出，不考虑手续费/滑点；用于相对比较）。
"""

import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, f1_score, precision_score

def walk_splits(df: pd.DataFrame, train_len=600, test_len=120, step=60):
    idx = np.arange(len(df))
    i = train_len
    while i + test_len <= len(df):
        tr = idx[i-train_len:i]
        te = idx[i:i+test_len]
        yield tr, te
        i += step

def topk_precision(y_true: np.ndarray, y_score: np.ndarray, k_ratio=0.2) -> float:
    k = max(1, int(len(y_true) * k_ratio))
    order = np.argsort(-y_score)
    pick = order[:k]
    return float(np.mean(y_true[pick]))

def simple_kday_backtest(close: pd.Series, y_score: np.ndarray, k: int = 10, pick_ratio=0.2):
    # 每天选TopK%，按close买入，K日后按close卖出
    n = len(close)
    k = min(k, n-1)
    ret = []
    for t in range(n - k):
        scores = y_score[:t+1]
        cur_idx = t
        # 只在当日做一次选择
        order = np.argsort(-scores)[:max(1, int(len(scores)*pick_ratio))]
        if cur_idx in order:
            r = float(close.iloc[t+k]/close.iloc[t] - 1.0)
            ret.append(r)
    if not ret: return 0.0, 0.0
    arr = np.array(ret)
    ann = (1+arr.mean())**(252/k) - 1
    mdd = float(np.max(np.maximum.accumulate(arr) - arr)) if len(arr)>1 else 0.0
    return float(ann), float(mdd)
