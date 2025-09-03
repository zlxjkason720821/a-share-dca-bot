# -*- coding: utf-8 -*-
"""
标签生成：二分类(未来K日收益>=阈值)、三分类、回归（未来K日收益）。
"""

import pandas as pd
import numpy as np


def _future_return_series(df: pd.DataFrame, k: int) -> pd.Series:
    """未来k日收益率序列，对齐到起点"""
    close = pd.Series(df["close"].values, index=df.index)
    fut = (close.shift(-k) / close - 1.0)
    return fut

def label_binary_future_return(df: pd.DataFrame, k: int = 10, thr: float = 0.05) -> pd.Series:
    """
    原规则：future_ret > thr 记为 1，否则为 0。
    兜底：若单一类别，自动降阈值；仍单一则改用分位数(0.6 / 0.55)。
    返回的 index 与 df 对齐（末尾k行为NaN）。
    """
    fut = _future_return_series(df, k)
    y = (fut > thr).astype("float32")

    def _is_single_class(s: pd.Series) -> bool:
        s = s.dropna().astype(int)
        return s.nunique() < 2

    # 1) 原阈值
    if not _is_single_class(y):
        return y

    # 2) 逐步降阈值
    for rate in (0.5, 0.25, 0.1):
        y_try = (fut > (thr * rate)).astype("float32")
        if not _is_single_class(y_try):
            return y_try

    # 3) 分位数法兜底：60/40 -> 55/45
    fut_clean = fut.dropna()
    if len(fut_clean) >= 50:  # 样本太少也没意义
        for q in (0.60, 0.55):
            cut = fut_clean.quantile(q)
            y_try = (fut >= cut).astype("float32")
            if not _is_single_class(y_try):
                return y_try

    # 4) 仍然单一类别：返回原结果（让上层感知并跳过）
    return y


def label_ternary_future_return(df, k: int = 10, thr_pos: float = 0.05, thr_neg: float = -0.05) -> pd.Series:
    """未来k日收益 >thr_pos->+1, <thr_neg->-1, 否则0"""
    c = df['close']
    fut = c.shift(-k) / c - 1.0
    y = pd.Series(0, index=df.index, dtype=int)
    y[fut >= thr_pos] = 1
    y[fut <= thr_neg] = -1
    return y

def target_reg_future_return(df, k: int = 10) -> pd.Series:
    """回归目标：未来k日累计收益"""
    c = df['close']
    fut = c.shift(-k) / c - 1.0
    return fut
