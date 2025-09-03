# -*- coding: utf-8 -*-
# tools/data_loader.py — 本地CSV优先 → 失败时可扩展到 akshare/yfinance（此处默认关闭外网）
import os
import pandas as pd

CN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}
EN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in df.columns:
        if c in CN_MAP:
            cols[c] = CN_MAP[c]
        elif c in EN_MAP:
            cols[c] = EN_MAP[c]
        else:
            cols[c] = c
    df = df.rename(columns=cols).copy()
    # 只保留必需列
    keep = ["date", "open", "high", "low", "close", "volume"]
    have = [c for c in keep if c in df.columns]
    df = df[have].copy()
    # 规范类型
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
    return df

def get_price_df(symbol: str, timeout_sec: int = 8, use_cache: bool = True) -> pd.DataFrame:
    """
    读取 data/<symbol>.csv 为主（本地优先）。如需外网兜底，可自行加 akshare/yfinance。
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"{symbol}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"local csv not found: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = _normalize_columns(df)
    # 去重
    if "date" in df.columns:
        df = df.drop_duplicates(subset=["date"])
    return df
