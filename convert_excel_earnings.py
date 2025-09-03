#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd, re, math, csv, os, sys, traceback
from pathlib import Path
from datetime import datetime
from dateutil import parser as dtparse

DATA_DIR = Path("data")
RAW_DIR = Path("raw")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _norm_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).upper().strip()
    s = s.replace(".SH","").replace(".SZ","").replace("SH","").replace("SZ","")
    m = re.search(r"(\d{6})", s)
    return m.group(1) if m else ""

def _to_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan","none","null","-"}:
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

def _to_date(x):
    if pd.isna(x):
        return None
    try:
        d = dtparse.parse(str(x)).date()
        return d.isoformat()
    except:
        return None

def _safe_div(num, den):
    try:
        den = float(den)
        if abs(den) < 1e-12:
            return None
        return float(num) / den
    except:
        return None

def _compute_yoy(rows):
    """同一月-日，年差1，计算同比"""
    if not rows:
        return rows
    by_key = {}
    for r in rows:
        try:
            dt = pd.to_datetime(r["report_date"])
            key = (dt.month, dt.day)
            by_key.setdefault(key, []).append((dt.year, r))
        except Exception:
            continue
    for key, lst in by_key.items():
        lst.sort(key=lambda x: x[0])
        for i in range(1, len(lst)):
            y_cur, rc = lst[i]
            y_prev, rp = lst[i-1]
            if y_cur - y_prev == 1:
                if rc.get("yoy_revenue") in (None, ""):
                    rc["yoy_revenue"] = _safe_div((rc.get("revenue") or 0) - (rp.get("revenue") or 0), abs(rp.get("revenue") or 0))
                if rc.get("yoy_profit") in (None, ""):
                    rc["yoy_profit"] = _safe_div((rc.get("net_income") or 0) - (rp.get("net_income") or 0), abs(rp.get("net_income") or 0))
    return rows

# 可能的列名池（中英兼容，含模糊匹配）
CAND = {
    "report_date": ["报告期","报告日期","公告日期","report_date","REPORT_DATE","日期"],
    "revenue": ["营业总收入","营业收入","revenue","REVENUE"],
    "net_income": ["归属于母公司股东的净利润","归母净利润","净利润","net_income","NETPROFIT_PARENT","NET_PROFIT_PARENT"],
    "eps": ["基本每股收益","每股收益","EPS","BASIC_EPS"],
    "yoy_revenue": ["营业总收入同比增长","营收同比","yoy_revenue","YOY_REVENUE"],
    "yoy_profit": ["归母净利润同比增长","净利润同比","yoy_profit","YOY_NETPROFIT"],
    "code": ["股票代码","证券代码","代码","code","Code","SECURITY_CODE"],
}

def _pick_col(df, names):
    for n in names:
        if n in df.columns: return n
    for c in df.columns:
        for n in names:
            if n.lower() in str(c).lower():
                return c
    return None

def convert_one(path: Path, out_dir: Path, keep_all=False):
    if path.suffix.lower() in [".xlsx",".xls"]:
        df = pandas.read_excel(path)
    elif path.suffix.lower() in [".csv"]:
        df = pandas.read_csv(path)
    else:
        print(f"[SKIP] 不支持的文件类型: {path.name}")
        return

    sym = _norm_symbol(path.stem)
    code_col = _pick_col(df, CAND["code"]) if not sym else None
    if not sym and code_col:
        vals = df[code_col].dropna().astype(str).tolist()
        sym = _norm_symbol(vals[0]) if vals else ""
    if not sym:
        print(f"[ERR] 无法识别股票代码: {path.name}")
        return

    col_date = _pick_col(df, CAND["report_date"])
    col_rev  = _pick_col(df, CAND["revenue"])
    col_net  = _pick_col(df, CAND["net_income"])
    col_eps  = _pick_col(df, CAND["eps"])
    col_yoyr = _pick_col(df, CAND["yoy_revenue"])
    col_yoyp = _pick_col(df, CAND["yoy_profit"])

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "report_date": _to_date(r.get(col_date)) if col_date else None,
            "revenue":     _to_float(r.get(col_rev)) if col_rev else None,
            "net_income":  _to_float(r.get(col_net)) if col_net else None,
            "eps":         _to_float(r.get(col_eps)) if col_eps else None,
            "yoy_revenue": _to_float(r.get(col_yoyr)) if col_yoyr else None,
            "yoy_profit":  _to_float(r.get(col_yoyp)) if col_yoyp else None,
        })
    rows = [x for x in rows if x["report_date"]]
    rows = sorted(rows, key=lambda r: r["report_date"])
    rows = _compute_yoy(rows)
    if not keep_all:
        rows = rows[-4:]

    _ensure_dir(out_dir)
    out = out_dir / f"{sym}earn.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["report_date","revenue","net_income","eps","yoy_revenue","yoy_profit"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] {path.name} -> {out} ({len(rows)} rows)")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="将手动下载的财报 Excel/CSV 转为 data/{symbol}earn.csv")
    ap.add_argument("--raw-dir", default=str(RAW_DIR), help="原始文件目录（含 Excel/CSV）")
    ap.add_argument("--out-dir", default=str(DATA_DIR), help="输出目录（默认 data/）")
    ap.add_argument("--keep-all", action="store_true", help="不裁剪期数，全部输出")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    files = []
    for suf in ("*.xlsx","*.xls","*.csv"):
        files += list(raw_dir.glob(suf))
    if not files:
        print(f"[INFO] {raw_dir} 下未找到 Excel/CSV")
        return

    for p in files:
        try:
            convert_one(p, out_dir, keep_all=args.keep_all)
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
