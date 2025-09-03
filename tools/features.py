# -*- coding: utf-8 -*-
# tools/features.py — 从K线构造通用特征（适配分类/回归模型）
import numpy as np
import pandas as pd

def _rsi(series: pd.Series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：包含列 date, open, high, low, close, volume 的DataFrame（按时间升序）
    输出：与输入索引对齐的特征DataFrame（不丢行，不重置索引）
    """
    x = df.copy()

    # 收益率与均线
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_5"] = x["close"].pct_change(5)
    x["ret_10"] = x["close"].pct_change(10)

    x["ma_5"] = x["close"].rolling(5, min_periods=5).mean()
    x["ma_10"] = x["close"].rolling(10, min_periods=10).mean()
    x["ma_20"] = x["close"].rolling(20, min_periods=20).mean()

    # 与均线的偏离（避免除零）
    eps = 1e-12
    x["ma_5_gap"] = x["close"] / (x["ma_5"] + eps) - 1
    x["ma_10_gap"] = x["close"] / (x["ma_10"] + eps) - 1
    x["ma_20_gap"] = x["close"] / (x["ma_20"] + eps) - 1

    # 波动率
    ret = x["close"].pct_change()
    x["vol_5"] = ret.rolling(5, min_periods=5).std()
    x["vol_20"] = ret.rolling(20, min_periods=20).std()

    # RSI / MACD
    x["rsi_14"] = _rsi(x["close"], 14)
    macd, macd_sig, macd_hist = _macd(x["close"])
    x["macd"] = macd
    x["macd_sig"] = macd_sig
    x["macd_hist"] = macd_hist

    # 量价
    if "volume" in x.columns:
        x["vol_chg_5"] = x["volume"].pct_change(5)

    # 删除非特征列（若存在）
    if "date" in x.columns:
        x = x.drop(columns=["date"])

    # 统一清洗：将 inf/-inf 置为 NaN（由 Pipeline 的 SimpleImputer 处理）
    x = x.replace([np.inf, -np.inf], np.nan)

    return x

