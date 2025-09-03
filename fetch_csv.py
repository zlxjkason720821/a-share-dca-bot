# -*- coding: utf-8 -*-
# fetch_csv.py — 全量首建 + 增量更新（国外网络稳：yfinance优先，akshare兜底）
# 功能：
# 1) 第一次无CSV时抓全历史；之后每天只抓增量（上次最后日期+1 → 今天）
# 2) 数据源：默认 yfinance（海外更稳），失败再兜底 akshare（需可访问国内站点）
# 3) 统一列名：日期, 开盘, 最高, 最低, 收盘, 成交量（UTF-8-SIG）
# 4) 股票池来源：优先 tools/universe/focus_a_100.yaml；否则并集 model_config/budget/signals
# 5) 日志：logs/fetch_csv.log；数据：data/<SYMBOL>.csv
# 兼容：Python 3.9+

import os
import traceback
from typing import Optional
from datetime import datetime
from pathlib import Path

import pandas as pd

def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

ak = _safe_import("akshare")
yf = _safe_import("yfinance")

# --- 目录与常量 ---
ROOT: Path = Path(__file__).resolve().parent           # 项目根（本文件所在目录）
DATA_DIR: Path = ROOT / "data"
LOGS_DIR: Path = ROOT / "logs"
UNIVERSE_FILE: Path = ROOT / "tools" / "universe" / "focus_a_100.yaml"
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE: Path = LOGS_DIR / "fetch_csv.log"

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{ts}] {msg}"
    print(text)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass

def _fmt_cols_cn(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为同花顺风格，并规范类型"""
    mapping = {
        "date": "日期", "Date": "日期",
        "open": "开盘", "Open": "开盘",
        "high": "最高", "High": "最高",
        "low":  "最低", "Low":  "最低",
        "close": "收盘", "Close": "收盘",
        "volume": "成交量", "Volume": "成交量",
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols).copy()

    keep = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    exist_keep = [c for c in keep if c in df.columns]
    df = df[exist_keep].copy()

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["开盘", "最高", "最低", "收盘", "成交量"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["收盘"])
    return df

# ---------------- 数据源封装 ----------------
def fetch_ak_full(sym: str) -> Optional[pd.DataFrame]:
    """akshare 全量（前复权日线）"""
    if ak is None:
        return None
    try:
        s = ("sz" + sym.split(".")[0]) if sym.endswith(".SZ") else ("sh" + sym.split(".")[0])
        df = ak.stock_zh_a_daily(symbol=s, adjust="qfq")
        if df is None or df.empty:
            return None
        return _fmt_cols_cn(df)
    except Exception as e:
        log(f"akshare full {sym} 异常: {e}")
        log(traceback.format_exc())
        return None

def fetch_yf_range(sym: str, start_date: Optional[str], end_date: Optional[str]) -> Optional[pd.DataFrame]:
    """yfinance 全量（period=max）或区间（start/end）"""
    if yf is None:
        return None
    try:
        yfsym = sym.replace(".SH", ".SS").replace(".SZ", ".SZ")
        tkr = yf.Ticker(yfsym)
        if start_date is None:
            df = tkr.history(period="max", interval="1d", auto_adjust=False)
        else:
            df = tkr.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        return _fmt_cols_cn(df)
    except Exception as e:
        log(f"yfinance {sym} 异常: {e}")
        log(traceback.format_exc())
        return None

def fetch_full(sym: str) -> Optional[pd.DataFrame]:
    """首建全量：yfinance 优先，失败回退 akshare"""
    df = fetch_yf_range(sym, start_date=None, end_date=None)
    if df is not None and not df.empty:
        return df
    return fetch_ak_full(sym)

def fetch_increment(sym: str, last_date: str) -> Optional[pd.DataFrame]:
    """增量：抓 last_date+1 ~ 今天；优先 yfinance 区间，失败兜底 akshare 全量后过滤。"""
    start_dt = (pd.to_datetime(last_date, errors="coerce") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_dt = datetime.now().strftime("%Y-%m-%d")
    inc = fetch_yf_range(sym, start_date=start_dt, end_date=end_dt)
    if inc is not None and not inc.empty:
        return inc
    full = fetch_ak_full(sym)
    if full is None or full.empty:
        return None
    mask = pd.to_datetime(full["日期"], errors="coerce") > pd.to_datetime(last_date, errors="coerce")
    inc2 = full.loc[mask].copy()
    return inc2 if not inc2.empty else None

# ---------------- 股票池加载 ----------------
def load_symbols_from_universe() -> list:
    """
    优先从 tools/universe/focus_a_100.yaml 读取；若不存在或为空，回退到：
    - tools/model_config.yaml: symbols.keys()
    - budget.yaml: weights.keys()
    - signals_config.yaml: symbols.keys()（dict）
    """
    import yaml
    syms = set()

    # 1) 独立清单（推荐）
    if UNIVERSE_FILE.exists():
        try:
            u = yaml.safe_load(open(UNIVERSE_FILE, "r", encoding="utf-8"))
            if isinstance(u, dict) and "symbols" in u:
                if isinstance(u["symbols"], list):
                    got = [s for s in u["symbols"] if isinstance(s, str) and "." in s]
                    return sorted(set(got))
                elif isinstance(u["symbols"], dict):
                    got = [k for k in u["symbols"].keys() if isinstance(k, str) and "." in k]
                    return sorted(set(got))
        except Exception as e:
            log(f"[WARN] read {UNIVERSE_FILE} failed: {e}")

    # 2) 回退：model_config.yaml / budget.yaml / signals_config.yaml
    try:
        mc = ROOT / "tools" / "model_config.yaml"
        if mc.exists():
            m = yaml.safe_load(open(mc, "r", encoding="utf-8"))
            if isinstance(m.get("symbols"), dict):
                syms |= set(m["symbols"].keys())
    except Exception as e:
        log(f"[WARN] read model_config.yaml failed: {e}")

    try:
        by = ROOT / "budget.yaml"
        if by.exists():
            b = yaml.safe_load(open(by, "r", encoding="utf-8"))
            if isinstance(b.get("weights"), dict):
                syms |= set(b["weights"].keys())
    except Exception as e:
        log(f"[WARN] read budget.yaml failed: {e}")

    try:
        sc = ROOT / "signals_config.yaml"
        if sc.exists():
            s = yaml.safe_load(open(sc, "r", encoding="utf-8"))
            if isinstance(s.get("symbols"), dict):
                syms |= set(s["symbols"].keys())
    except Exception as e:
        log(f"[WARN] read signals_config.yaml failed: {e}")

    return sorted([x for x in syms if isinstance(x, str) and "." in x])

def main() -> None:
    log("===== CSV 抓取开始 =====")
    log(f"Project ROOT   : {ROOT}")
    log(f"Data directory : {DATA_DIR}")
    log(f"Universe file  : {UNIVERSE_FILE} ({'exists' if UNIVERSE_FILE.exists() else 'missing'})")

    symbols = load_symbols_from_universe()
    log(f"Symbols loaded : {len(symbols)}")
    if not symbols:
        log("[ERROR] 没有加载到任何股票代码，请检查 tools/universe/focus_a_100.yaml 或三份 YAML。退出。")
        return

    any_ok = False
    for sym in symbols:
        fn: Path = DATA_DIR / f"{sym}.csv"
        try:
            if not fn.exists():
                log(f"[{sym}] 首建：抓全历史 …")
                df = fetch_full(sym)
                if df is None or df.empty:
                    log(f"[{sym}] 全量抓取失败")
                else:
                    df = df.sort_values("日期").drop_duplicates(subset=["日期"]).reset_index(drop=True)
                    df.to_csv(fn, index=False, encoding="utf-8-sig")
                    log(f"[{sym}] 全量保存：{fn}  共{len(df)}行")
                    any_ok = True
            else:
                old = pd.read_csv(fn, encoding="utf-8-sig")
                old = _fmt_cols_cn(old).sort_values("日期").drop_duplicates(subset=["日期"]).reset_index(drop=True)
                if old.empty:
                    log(f"[{sym}] 现有文件为空，转为首建 …")
                    df = fetch_full(sym)
                    if df is None or df.empty:
                        log(f"[{sym}] 全量抓取失败")
                    else:
                        df = df.sort_values("日期").drop_duplicates(subset=["日期"]).reset_index(drop=True)
                        df.to_csv(fn, index=False, encoding="utf-8-sig")
                        log(f"[{sym}] 全量覆盖：{fn}  共{len(df)}行")
                        any_ok = True
                else:
                    last_date = str(old["日期"].iloc[-1])
                    log(f"[{sym}] 增量：从 {last_date} + 1 抓到今天 …")
                    inc = fetch_increment(sym, last_date)
                    if inc is None or inc.empty:
                        log(f"[{sym}] 无需更新（可能非交易日或源无新数据）")
                    else:
                        new = pd.concat([old, inc], ignore_index=True)
                        new = new.sort_values("日期").drop_duplicates(subset=["日期"]).reset_index(drop=True)
                        new.to_csv(fn, index=False, encoding="utf-8-sig")
                        log(f"[{sym}] 增量追加 {len(inc)} 行 → 总 {len(new)} 行")
                        any_ok = True
        except Exception as e:
            log(f"[{sym}] 处理异常：{e}")
            log(traceback.format_exc())

    if not any_ok:
        log("[FAIL] 没有任何标的更新成功，请检查网络/依赖/日志")
    log("===== CSV 抓取结束 =====")

if __name__ == "__main__":
    main()
