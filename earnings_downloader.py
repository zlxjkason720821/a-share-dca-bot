#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, json, time, sys, os, csv, re, math, traceback
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = (SCRIPT_DIR / "data")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize_num(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"null", "nan", "none", "-"}:
        return None
    s = s.replace(",", "").replace(" ", "")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)(万|億|亿)?$", s)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit in ("万",):
            return val * 1e4
        if unit in ("億", "亿"):
            return val * 1e8
        return val
    try:
        return float(s)
    except Exception:
        return None

def _get(d: Any, path: str) -> Any:
    cur = d
    for part in re.split(r"\.", path) if path else []:
        if part == "":
            continue
        m = re.match(r"^([^\[]+)(\[\d+\])*$", part)
        if not m:
            return None
        key = m.group(1)
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            return None
        idx_parts = re.findall(r"\[(\d+)\]", part)
        for idx_str in idx_parts:
            try:
                i = int(idx_str)
                if isinstance(cur, list) and 0 <= i < len(cur):
                    cur = cur[i]
                else:
                    return None
            except Exception:
                return None
    return cur

def _parse_headers(header_list: List[str]) -> Dict[str, str]:
    out = {}
    for h in header_list or []:
        if ":" in h:
            k, v = h.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def _request_json(url: str, timeout: int, proxies: Dict[str, str], headers: Dict[str, str], cookies: str, retries: int) -> Any:
    sess = requests.Session()
    if cookies:
        headers = headers.copy()
        headers["Cookie"] = cookies
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, timeout=timeout, proxies=proxies or None, headers=headers or None)
            r.raise_for_status()
            text = r.text.strip()
            if re.match(r"^\w+\(", text) and text.endswith(")"):
                text = text[text.find("(")+1:-1]
            return json.loads(text)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(min(1.0 * attempt, 5.0))
            else:
                raise
    raise last_err

def download_http_json(symbol: str, url_template: str, items_path: str, fields: Dict[str, str],
                       timeout: int, proxies: Dict[str, str], headers: Dict[str, str], cookies: str, retries: int) -> pd.DataFrame:
    url = url_template.format(symbol=symbol)
    data = _request_json(url, timeout, proxies, headers, cookies, retries)
    items = _get(data, items_path) if items_path else data
    if not isinstance(items, list):
        items = [items]
    rows = []
    for it in items:
        row = {}
        for out_key, path in fields.items():
            val = _get(it, path) if path else None
            if out_key == "report_date":
                try:
                    row[out_key] = pd.to_datetime(val, errors="coerce").date().isoformat() if val is not None else None
                except Exception:
                    row[out_key] = None
            else:
                row[out_key] = _normalize_num(val)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def transform_local_file(path: Path, sheet: Optional[str], mapping: Dict[str, str]) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def pick(colname: str) -> Optional[str]:
        lc = colname.lower()
        return cols.get(lc, None)
    out = {}
    for out_key, src_key in mapping.items():
        src_col = pick(src_key) or pick(src_key.replace(" ", "")) or pick(src_key.replace("_", "")) or cols.get(src_key, None)
        if src_col is None:
            out[out_key] = None
        else:
            out[out_key] = df[src_col]
    out_df = pd.DataFrame(out)
    if "report_date" in out_df.columns:
        out_df["report_date"] = pd.to_datetime(out_df["report_date"], errors="coerce").dt.date.astype("string")
    for k in ["revenue", "net_income", "eps", "yoy_revenue", "yoy_profit"]:
        if k in out_df.columns:
            out_df[k] = out_df[k].map(_normalize_num)
    return out_df

def save_earnings_csv(df: pd.DataFrame, out_path: Path):
    keep = ["report_date", "revenue", "net_income", "eps", "yoy_revenue", "yoy_profit"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep]
    df = df.dropna(how="all")
    if "report_date" in df.columns:
        try:
            df["_dt"] = pd.to_datetime(df["report_date"], errors="coerce")
            df = df.sort_values("_dt").drop(columns=["_dt"])
        except Exception:
            pass
    _ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Download/transform earnings to data/{symbol}earn.csv")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., 600519.SH,000001.SZ or 600519,000001")
    ap.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR), help="Directory to place CSVs (default: ./data)")
    ap.add_argument("--underscore", action="store_true", help="Use {symbol}_earn.csv naming instead of {symbol}earn.csv")

    sub = ap.add_subparsers(dest="mode", required=True)

    s1 = sub.add_parser("http-json", help="Fetch from a JSON API via GET")
    s1.add_argument("--url-template", required=True, help="URL template with {symbol}")
    s1.add_argument("--items-path", default="", help="Dot-path to list of records in JSON (e.g., 'data.list')")
    s1.add_argument("--field-report-date", required=True, help="Dot-path to report date within each item")
    s1.add_argument("--field-revenue", default="", help="Dot-path to revenue")
    s1.add_argument("--field-net-income", default="", help="Dot-path to net income")
    s1.add_argument("--field-eps", default="", help="Dot-path to EPS")
    s1.add_argument("--field-yoy-revenue", default="", help="Dot-path to YoY revenue growth (decimal or %)")
    s1.add_argument("--field-yoy-profit", default="", help="Dot-path to YoY profit growth (decimal or %)")
    s1.add_argument("--timeout", type=int, default=12)
    s1.add_argument("--retries", type=int, default=3)
    s1.add_argument("--http-proxy", default="")
    s1.add_argument("--https-proxy", default="")
    s1.add_argument("--cookie", default="")
    s1.add_argument("--header", action="append", default=[], help="Custom headers, e.g., 'User-Agent: Mozilla/5.0'")

    s2 = sub.add_parser("transform", help="Transform a local CSV/XLSX to the target schema")
    s2.add_argument("--input-path", required=True, help="Path to the source CSV/XLSX")
    s2.add_argument("--sheet", default=None, help="Sheet name or index for Excel")
    s2.add_argument("--map-report-date", required=True, help="Source column for report_date (e.g., '报告期' or 'report_date')")
    s2.add_argument("--map-revenue", default="", help="Source column for revenue")
    s2.add_argument("--map-net-income", default="", help="Source column for net_income")
    s2.add_argument("--map-eps", default="", help="Source column for eps")
    s2.add_argument("--map-yoy-revenue", default="", help="Source column for yoy_revenue")
    s2.add_argument("--map-yoy-profit", default="", help="Source column for yoy_profit")

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.mode == "http-json":
        headers = _parse_headers(args.header)
        proxies = {}
        if args.http_proxy: proxies["http"] = args.http_proxy
        if args.https_proxy: proxies["https"] = args.https_proxy
        fields = {
            "report_date": args.field_report_date,
            "revenue": args.field_revenue,
            "net_income": args.field_net_income,
            "eps": args.field_eps,
            "yoy_revenue": args.field_yoy_revenue,
            "yoy_profit": args.field_yoy_profit,
        }
        for sym in symbols:
            try:
                df = download_http_json(sym, args.url_template, args.items_path, fields,
                                        args.timeout, proxies, headers, args.cookie, args.retries)
                out_path = out_dir / (f"{sym}_earn.csv" if args.underscore else f"{sym}earn.csv")
                save_earnings_csv(df, out_path)
                print(f"[OK] {sym} -> {out_path}")
            except Exception as e:
                print(f"[ERR] {sym}: {e}", file=sys.stderr)
                traceback.print_exc()

    elif args.mode == "transform":
        mapping = {
            "report_date": args.map_report_date,
            "revenue": args.map_revenue,
            "net_income": args.map_net_income,
            "eps": args.map_eps,
            "yoy_revenue": args.map_yoy_revenue,
            "yoy_profit": args.map_yoy_profit,
        }
        src_path = Path(args.input_path)
        df = transform_local_file(src_path, args.sheet, mapping)
        for sym in symbols:
            out_path = out_dir / (f"{sym}_earn.csv" if args.underscore else f"{sym}earn.csv")
            save_earnings_csv(df, out_path)
            print(f"[OK] {sym} -> {out_path} (transform from {src_path})")

if __name__ == "__main__":
    main()
