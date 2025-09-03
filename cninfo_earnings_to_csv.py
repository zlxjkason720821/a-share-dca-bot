
# cninfo_earnings_to_csv.py
# 目标：优先用 AKShare 的“巨潮资讯”财务指标接口（如果可用）；不可用时回退到东方财富接口；
#      统一落地为 data/{symbol}earn.csv（或 {symbol}_earn.csv），最近 4 期 + 同比自动回填。
#
# 用法：
#   pip install --upgrade akshare pandas python-dateutil
#   python cninfo_earnings_to_csv.py --symbols 600519,000001.SZ --sleep 1.0
#   python cninfo_earnings_to_csv.py --symbols-file symbols.txt --sleep 1.0
#
# 说明：
# - 支持代码形式：600519 / 600519.SH / 000001.SZ（内部会标准化为6位数用于匹配）
# - 字段对齐：report_date,revenue,net_income,eps,yoy_revenue,yoy_profit
# - 同比回填规则：同一“月-日”，年份相差1，(本期-去年同期)/|去年同期|
# - 每只股票仅保留最近 4 期（可通过 --keep-all 保留全部）

import os, re, time, math, sys, csv, traceback
from pathlib import Path
from dateutil import parser as dtparse

import pandas as pd
import akshare as ak

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def norm_symbol(sym: str) -> str:
    s = sym.strip().upper()
    s = s.replace(".SH","").replace(".SZ","").replace("SH","").replace("SZ","")
    return re.sub(r"\D","",s)

def pick_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    for c in df.columns:
        for name in candidates:
            if name in str(c):
                return c
    return None

def to_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return None
    for unit, mul in [("万", 1e4), ("亿", 1e8)]:
        if unit in s:
            try:
                return float(s.replace(unit, "")) * mul
            except:
                return None
    try:
        return float(s)
    except:
        return None
def fetch_lrb_df():
    """利润表（东财），无参返回全量，再本地按代码过滤。"""
    try:
        import akshare as ak
        df = ak.stock_lrb_em()  # 注意：不带 symbol 参数
        return df if df is not None and not df.empty else None
    except Exception:
        return None

def rows_from_lrb(df, sym6: str):
    if df is None or df.empty:
        return None
    # 可能的代码列
    code_cols = ["股票代码","证券代码","代码","code","Code"]
    code_col = next((c for c in code_cols if c in df.columns), None)
    if not code_col:
        # 模糊找一下
        code_col = next((c for c in df.columns if "代码" in str(c)), None)
    if code_col:
        dff = df[df[code_col].astype(str).str.contains(sym6)]
    else:
        dff = df
    if dff is None or dff.empty:
        return None

    # 可能的字段名
    def pick(cols, cands):
        for n in cands:
            if n in cols: return n
        for c in cols:
            for n in cands:
                if n in str(c): return c
        return None

    date_col = pick(dff.columns, ["公告日期","报告期","报告日期","REPORT_DATE","日期"])
    rev_col  = pick(dff.columns, ["营业总收入","营业收入"])
    ni_col   = pick(dff.columns, ["归属于母公司股东的净利润","归母净利润","净利润"])
    eps_col  = pick(dff.columns, ["基本每股收益","每股收益","EPS"])

    rows = []
    for _, r in dff.iterrows():
        rows.append({
            "report_date": to_date(r.get(date_col)) if date_col else None,
            "revenue":     to_float(r.get(rev_col)) if rev_col else None,
            "net_income":  to_float(r.get(ni_col)) if ni_col else None,
            "eps":         to_float(r.get(eps_col)) if eps_col else None,
            "yoy_revenue": None,
            "yoy_profit":  None,
        })
    rows = [x for x in rows if x["report_date"]]
    return rows if rows else None

def to_date(x):
    if pd.isna(x):
        return None
    try:
        return dtparse.parse(str(x)).date().isoformat()
    except:
        return None

def safe_div(num, den):
    try:
        den = float(den)
        if abs(den) < 1e-12: return None
        return float(num) / den
    except:
        return None

def compute_yoy(rows):
    if not rows: return rows
    buckets = {}
    for r in rows:
        try:
            d = pd.to_datetime(r["report_date"])
            key = (d.month, d.day)
            buckets.setdefault(key, []).append((d.year, r))
        except Exception:
            continue
    for key, lst in buckets.items():
        lst.sort(key=lambda x: x[0])
        for i in range(1, len(lst)):
            y_cur, cur = lst[i]
            y_prev, prev = lst[i-1]
            if y_cur - y_prev == 1:
                if cur.get("yoy_revenue") in (None, ""):
                    cur["yoy_revenue"] = safe_div((cur.get("revenue") or 0) - (prev.get("revenue") or 0), abs(prev.get("revenue") or 0))
                if cur.get("yoy_profit") in (None, ""):
                    cur["yoy_profit"] = safe_div((cur.get("net_income") or 0) - (prev.get("net_income") or 0), abs(prev.get("net_income") or 0))
    return rows

def save_rows(symbol, rows, underscore=False, keep_all=False):
    rows = [r for r in rows if r.get("report_date")]
    rows = sorted(rows, key=lambda r: r["report_date"])
    rows = compute_yoy(rows)
    if not keep_all:
        rows = rows[-4:]
    out = DATA_DIR / (f"{symbol}_earn.csv" if underscore else f"{symbol}earn.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["report_date","revenue","net_income","eps","yoy_revenue","yoy_profit"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] {symbol} -> {out} ({len(rows)} rows)")

# ----------------- 抓取：CNINFO 优先 -----------------
def fetch_cninfo(sym6: str):
    """
    巨潮优先：不同 AKShare 版本里，巨潮接口可能叫不同名字，且有的需要带交易所后缀。
    这里对“函数名 × 符号格式”做穷举尝试：
      - 函数：stock_financial_analysis_indicator_cninfo / stock_financial_analysis_indicator
      - 符号：'600000' / '600000.SH' / 'sh600000'
    只要任意一种返回了非空 DataFrame，就视为 CNINFO 命中。
    """
    # 候选函数（全是“巨潮”系）
    func_candidates = []
    if hasattr(ak, "stock_financial_analysis_indicator_cninfo"):
        func_candidates.append(ak.stock_financial_analysis_indicator_cninfo)
    if hasattr(ak, "stock_financial_analysis_indicator"):
        # 某些版本这个函数也是走巨潮数据源
        func_candidates.append(ak.stock_financial_analysis_indicator)

    # 符号候选（不同版本要求不同格式）
    sym_candidates = [
        sym6,
        f"{sym6}.SH" if sym6.startswith("6") else f"{sym6}.SZ",
        f"sh{sym6}" if sym6.startswith("6") else f"sz{sym6}",
    ]

    for fn in func_candidates:
        for s in sym_candidates:
            # 先尝试 keyword 形式
            try:
                df = fn(symbol=s)
                if df is not None and not df.empty:
                    return df
            except TypeError:
                # 少数旧版用位置参数
                try:
                    df = fn(s)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    pass
            except Exception:
                # 某些票/时段可能抛异常，继续尝试其他组合
                pass

    return None


def rows_from_cninfo_df(df):
    date_col = pick_col(df, ["报告期","报告日期","公告日期","report_date","REPORT_DATE","日期"])
    rev_col  = pick_col(df, ["营业总收入","营业收入","REVENUE"])
    ni_col   = pick_col(df, ["归母净利润","归属于母公司股东的净利润","NETPROFIT_PARENT","净利润"])
    eps_col  = pick_col(df, ["基本每股收益","每股收益","EPS","BASIC_EPS"])
    yoyr_col = pick_col(df, ["营业总收入同比增长","营收同比","YOY_REVENUE"])
    yoyp_col = pick_col(df, ["归母净利润同比增长","净利润同比","YOY_NETPROFIT"])

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "report_date": to_date(r.get(date_col)) if date_col else None,
            "revenue":     to_float(r.get(rev_col)) if rev_col else None,
            "net_income":  to_float(r.get(ni_col)) if ni_col else None,
            "eps":         to_float(r.get(eps_col)) if eps_col else None,
            "yoy_revenue": to_float(r.get(yoyr_col)) if yoyr_col else None,
            "yoy_profit":  to_float(r.get(yoyp_col)) if yoyp_col else None,
        })
    rows = [x for x in rows if x["report_date"]]
    return rows if rows else None

# ----------------- 回退：东财接口 -----------------
def fetch_eastmoney(sym6: str):
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=sym6)
        if df is not None and not df.empty:
            return df
    except Exception:
        return None
    return None

def rows_from_em_df(df):
    return rows_from_cninfo_df(df)

def fetch_one(symbol: str, underscore=False, keep_all=False, pause=1.0):
    sym6 = norm_symbol(symbol)

    # 1) CNINFO 指标
    df = fetch_cninfo(sym6)
    rows = rows_from_cninfo_df(df) if df is not None else None

    # 2) EM 指标
    if not rows:
        df2 = fetch_eastmoney(sym6)
        rows = rows_from_em_df(df2) if df2 is not None else None

    # 3) EM 利润表 兜底
    if not rows:
        df3 = fetch_lrb_df()
        rows = rows_from_lrb(df3, sym6)

    if not rows:
        # 不再 raise，中断一条但不中断全局
        print(f"[WARN] {symbol}: no rows from CNINFO/EM endpoints, skip.")
        return

    save_rows(symbol, rows, underscore=underscore, keep_all=keep_all)
    if pause > 0:
        time.sleep(pause)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="600519,000001.SZ 多个用逗号分隔")
    ap.add_argument("--symbols-file", default="", help="每行一个代码（可含 .SH/.SZ）")
    ap.add_argument("--underscore", action="store_true", help="输出文件名使用 {symbol}_earn.csv")
    ap.add_argument("--keep-all", action="store_true", help="不裁剪期数，全部输出")
    ap.add_argument("--sleep", type=float, default=1.0, help="请求间隔秒，建议 >= 0.8")
    args = ap.parse_args()

    symbols = []
    if args.symbols:
        symbols += [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    symbols.append(s)

    if not symbols:
        print("请用 --symbols 或 --symbols-file 指定股票列表")
        sys.exit(2)

    for s in symbols:
        try:
            fetch_one(s, underscore=args.underscore, keep_all=args.keep_all, pause=args.sleep)
        except Exception as e:
            print(f"[ERR] {s}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
