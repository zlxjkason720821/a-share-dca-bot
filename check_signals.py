# -*- coding: utf-8 -*-
"""
check_signals.py — 综合打分器（增强版，Py3.9兼容）
来源：价格趋势 + 财报多指标（东财优先，新浪兜底）+ 新闻情绪（RSS优先，HTML兜底）+ 模型分
特性：
- 代理自动注入+可达性探测（从 signals_config.yaml 读取 proxy）
- 新闻：Google News RSS→失败再用 Bing HTML（eastmoney/sina/jrj/cs/10jqka/stcn/yicai）
- 财报：Eastmoney 优先，Sina 兜底；退避重试+当日缓存
- 运行进度打印到 stderr；最终 JSON 仅打印到 stdout
"""

import os
import sys
import json
import time
import math
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 可选第三方依赖（不存在也能跑，相关模块会回退）
try:
    import akshare as ak
except Exception:
    ak = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# ---------- 路径 ----------
THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_JSON_DEFAULT = os.path.join(LOG_DIR, "model_scores.json")
EARN_CACHE_PATH = os.path.join(LOG_DIR, "earnings_cache.json")
EARN_CACHE_LOCK = threading.Lock()


# ---------- 工具 ----------

# ---------- 巨潮资讯 ----------
def _cninfo_column_by_symbol(sym: str) -> str:
    """
    '000001.SZ' -> 'szse' ; '600519.SH' -> 'sse' ; 若无后缀，按首位 0/3 归深、6 归沪简单推断
    """
    try:
        if sym.endswith(".SZ"): return "szse"
        if sym.endswith(".SH"): return "sse"
        code = sym.split(".")[0]
        if code.startswith(("0","3")): return "szse"
        if code.startswith("6"): return "sse"
    except Exception:
        pass
    return "szse"

def _fetch_cninfo_titles(sym: str, lookback_days: int = 7, limit: int = 30):
    """
    从巨潮资讯拉取近 lookback_days 天公告标题。使用 POST 接口，不需要单独依赖。
    """
    try:
        import requests  # ensure local scope
    except Exception:
        return []
    col = _cninfo_column_by_symbol(sym)
    code = sym.split(".")[0]
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = {
        "stock": f"{code},",
        "pageNum": 1,
        "pageSize": int(limit),
        "column": col,
        "tabName": "fulltext",
        "plate": "",
        "seDate": "",
        "searchkey": "",
        "isHLtitle": "true",
    }
    proxies = {
        "http": os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("HTTPS_PROXY"),
    }
    try:
        r = requests.post(url, headers=headers, data=data, timeout=12, proxies=proxies)
        r.raise_for_status()
        js = r.json() or {}
        anns = js.get("announcements") or []
    except Exception:
        return []
    cutoff = datetime.utcnow() - timedelta(days=int(lookback_days))
    out = []
    for a in anns:
        tt = (a.get("announcementTitle") or "").strip()
        tms = a.get("announcementTime")
        dt = None
        # announcementTime 常为毫秒时间戳
        try:
            tms_int = int(str(tms))
            dt = datetime.utcfromtimestamp(tms_int / 1000.0)
        except Exception:
            try:
                dt = pd.to_datetime(tms, utc=True).to_pydatetime()
            except Exception:
                dt = None
        if dt and dt >= cutoff and tt:
            out.append(tt)
    return out


def _clip(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo

def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})

def symbols_dict(conf: dict):
    syms = conf.get("symbols")
    if isinstance(syms, dict):
        return syms
    if isinstance(syms, list):
        return {s: {} for s in syms}
    return {}

def run_with_timeout(fn, timeout_sec: int, *args, **kwargs) -> Tuple[float, Optional[str]]:
    """
    统一返回 (score, err)：
      - 正常：如果 fn 返回单值 -> (val, None)；若返回 (score, err) -> 原样返回
      - 超时：返回 (50.0, "timeout")
      - 异常：返回 (50.0, "<exception>")
    """
    ret = {"val": None}
    err = {"e": None}

    def _runner():
        try:
            ret["val"] = fn(*args, **kwargs)
        except Exception as e:
            err["e"] = e

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout_sec)

    if th.is_alive():
        return 50.0, "timeout"

    if err["e"] is not None:
        return 50.0, str(err["e"])

    val = ret["val"]
    if isinstance(val, tuple) and len(val) == 2:
        return val  # (score, err)
    try:
        return (float(val), None) if (val is not None) else (50.0, None)
    except Exception:
        return 50.0, None

# ---------- 本地CSV ----------
def load_csv_local(symbol: str, data_dir: str = DATA_DIR) -> Optional[pd.DataFrame]:
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_earnings_local(symbol: str, data_dir: str = DATA_DIR) -> Optional[pd.DataFrame]:
    """
    读取本地财报 CSV：优先 data/{symbol}earn.csv（或 {symbol}_earn.csv）。
    日期列可为 'date'/'report_date'/'日期'；营收 'revenue'/'营业收入'；
    净利 'net_income'/'净利润'；EPS 可选 'eps'。
    """
    candidates = [
        os.path.join(data_dir, f"{symbol}earn.csv"),
        os.path.join(data_dir, f"{symbol}_earn.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception:
                return None
    return None

def score_earnings_local(symbol: str) -> Tuple[float, Optional[str]]:
    """
    从本地 CSV 计算财报趋势分：
      - 按报告期排序，取最近 4 期（不够则尽量多）
      - 指标：YoY 增长 + QoQ 增长 + 线性趋势的斜率 组合
      - tanh 映射到 0..100，中位 50
    支持列名（不分大小写）:
      日期：date / report_date / 日期
      营收：revenue / 营业收入
      净利：net_income / 净利润
      EPS ：eps（可选）
      YoY ：yoy_revenue / yoy_profit / yoy_net（可选）
    """
    df = load_earnings_local(symbol)
    if df is None or df.empty:
        return 50.0, "本地财报缺失"

    df_cols = {c.lower(): c for c in df.columns}
    def _pick(*names):
        for n in names:
            if n in df_cols: 
                return df_cols[n]
        return None

    col_date = _pick("date", "report_date", "日期")
    col_rev  = _pick("revenue", "营业收入")
    col_net  = _pick("net_income", "净利润")
    col_eps  = _pick("eps")
    col_yoyr = _pick("yoy_revenue")
    col_yoyn = _pick("yoy_profit", "yoy_net")

    try:
        if col_date is not None:
            dt = pd.to_datetime(df[col_date], errors="coerce")
        else:
            dt = pd.Series(pd.RangeIndex(len(df)), index=df.index)
        rev = pd.to_numeric(df[col_rev], errors="coerce") if col_rev else None
        net = pd.to_numeric(df[col_net], errors="coerce") if col_net else None
        eps = pd.to_numeric(df[col_eps], errors="coerce") if col_eps else None
        yoyr = pd.to_numeric(df[col_yoyr], errors="coerce") if col_yoyr else None
        yovn = pd.to_numeric(df[col_yoyn], errors="coerce") if col_yoyn else None
    except Exception:
        return 50.0, "列解析失败"

    keep = dt.notna()
    if rev is not None: keep &= rev.notna()
    elif net is not None: keep &= net.notna()
    elif eps is not None: keep &= eps.notna()
    if not keep.any():
        return 50.0, "有效记录不足"

    sdf = pd.DataFrame({"dt": dt, "rev": rev, "net": net, "eps": eps, "yoyr": yoyr, "yoyn": yovn})[keep].sort_values("dt")
    if len(sdf) < 2:
        return 50.0, "样本过少"

    sdf = sdf.tail(4)

    base = sdf["rev"]
    if base.isna().all():
        base = sdf["net"]
    if base is None or base.isna().all():
        base = sdf["eps"]
    if base is None or base.isna().all():
        return 50.0, "指标缺失"

    try:
        qoq = (base.iloc[-1] - base.iloc[-2]) / abs(base.iloc[-2]) if abs(base.iloc[-2]) > 1e-12 else 0.0
    except Exception:
        qoq = 0.0

    yoy = None
    if not sdf["yoyr"].isna().all():
        yoy = float(sdf["yoyr"].iloc[-1])
        if yoy > 5: yoy = yoy / 100.0
    elif not sdf["yoyn"].isna().all():
        yoy = float(sdf["yoyn"].iloc[-1])
        if yoy > 5: yoy = yoy / 100.0
    else:
        yoy = None

    try:
        x = pd.RangeIndex(len(base))
        y = base.reset_index(drop=True).astype(float)
        xbar = x.mean(); ybar = y.mean()
        num = ((x - xbar) * (y - ybar)).sum()
        den = ((x - xbar) ** 2).sum()
        slope = float(num / den) if den != 0 else 0.0
        slope_rel = slope / (abs(ybar) + 1e-12)
    except Exception:
        slope_rel = 0.0

    parts = []
    parts.append(qoq if qoq is not None else 0.0)
    parts.append(yoy if yoy is not None else 0.0)
    parts.append(slope_rel)

    composite = sum(parts) / len(parts)
    score = _score_from_return(composite)
    return float(score), None
# ---------- 价格打分 ----------
def _score_from_return(ret: float) -> float:
    # tanh 柔化，-∞..+∞ → 0..100, 中心50
    return float(_clip(50.0 + 50.0 * math.tanh(ret * 3.0), 0.0, 100.0))


def score_price(symbol: str, lookback_days=7):
    """
    仅使用本地 data/{symbol}.csv 中“过去 lookback_days 个自然日”的收盘价来计算趋势：
    score = f( (last_close - first_close_in_window) / first_close_in_window )
    - 仅本地 CSV，不联网
    - 日期优先使用「日期」或「Date」列；否则取第一列尝试解析
    - 收盘价优先使用「收盘」或「Close」列；否则用最后一列
    """
    try:
        df = load_csv_local(symbol)
        if df is None or df.empty:
            return 50.0, "数据不足"

        # 选日期列
        cols = df.columns.tolist()
        date_col = "日期" if "日期" in cols else ("Date" if "Date" in cols else cols[0])
        # 选收盘列
        close_col = "收盘" if "收盘" in cols else ("Close" if "Close" in cols else cols[-1])

        # 解析日期 & 收盘
        try:
            dts = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        except Exception:
            return 50.0, "日期列无法解析"
        px = pd.to_numeric(df[close_col], errors="coerce")

        mask = dts.notna() & px.notna()
        dts = dts[mask]
        px = px[mask]

        if len(px) < 2:
            return 50.0, "收盘价数据不足"

        # 取最近 lookback_days 个自然日窗口（包含今天）
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
        in_win = (dts >= cutoff)
        if not in_win.any():
            return 50.0, "近7天无数据"

        win_px = px[in_win]
        if len(win_px) < 2:
            return 50.0, "近7天有效样本不足"

        # 按日期排序后取窗口首尾价
        win_df = pd.DataFrame({"dt": dts[in_win], "px": win_px}).sort_values("dt")
        first = float(win_df["px"].iloc[0])
        last = float(win_df["px"].iloc[-1])
        if first == 0.0:
            ret = 0.0
        else:
            ret = (last - first) / first

        return _score_from_return(ret), None
    except Exception as e:
        return 50.0, f"异常: {e}"

def _cache_load():
    try:
        if os.path.exists(EARN_CACHE_PATH):
            with open(EARN_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _cache_save(obj):
    try:
        with EARN_CACHE_LOCK:
            with open(EARN_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass

def _find_first_float(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] not in (None, "", "-"):
            try:
                return float(d[k])
            except Exception:
                continue
    return None

def _score_growth_pct(x: Optional[float]) -> float:
    if x is None:
        return 50.0
    return float(_clip(50.0 + 40.0 * math.tanh((x)/30.0), 0.0, 100.0))

def _score_gross_margin(x: Optional[float]) -> float:
    if x is None:
        return 50.0
    return float(_clip((x/50.0)*100.0, 0.0, 100.0))

def _score_roe(x: Optional[float]) -> float:
    if x is None:
        return 50.0
    return float(_clip(50.0 + 40.0 * math.tanh((x-10.0)/10.0), 0.0, 100.0))

def _score_debt_ratio(x: Optional[float]) -> float:
    if x is None:
        return 50.0
    return float(_clip(100.0 - (_clip(x, 0.0, 80.0)/80.0 * 100.0), 0.0, 100.0))

def _to_em_code(symbol: str) -> str:
    """'600000.SH' -> 'SH600000'  /  '000001.SZ' -> 'SZ000001' """
    code, ex = symbol.split(".")
    return (ex.upper() + code)

def _fetch_from_eastmoney(symbol: str) -> Dict[str, Optional[float]]:
    """
    使用 akshare 的东财财务指标接口抓取近一期（取第一行）。
    不同版本函数名/字段名可能不同，这里做了多重兼容。
    """
    if ak is None:
        raise RuntimeError("akshare未安装")

    em_code = _to_em_code(symbol)
    df = None
    # 兼容不同版本函数名
    for fn_name in ("stock_financial_analysis_indicator_em", "stock_financial_analysis_indicator"):
        try:
            fn = getattr(ak, fn_name)
            df = fn(em_code)
            break
        except Exception:
            continue
    if df is None or len(df) == 0:
        raise RuntimeError("eastmoney_no_data")

    d0 = df.iloc[0].to_dict()
    # 尝试多种字段名
    return {
        "net_profit_yoy": _find_first_float(d0, ["净利润同比增长率", "净利润增长率", "归母净利润同比", "净利润同比", "净利润同比(%)"]),
        "revenue_yoy":    _find_first_float(d0, ["营业收入同比增长率", "营业收入同比", "营收同比", "营业收入同比(%)"]),
        "gross_margin":   _find_first_float(d0, ["毛利率", "销售毛利率", "综合毛利率"]),
        "roe":            _find_first_float(d0, ["净资产收益率", "ROE", "加权净资产收益率", "ROE加权"]),
        "debt_ratio":     _find_first_float(d0, ["资产负债率", "负债率"]),
    }

def _fetch_from_sina(symbol: str) -> Dict[str, Optional[float]]:
    """使用新浪摘要（你之前的老逻辑）"""
    if ak is None:
        raise RuntimeError("akshare未安装")
    s = ("sz" + symbol.split(".")[0]) if symbol.endswith(".SZ") else ("sh" + symbol.split(".")[0])
    df = ak.stock_financial_abstract(symbol=s)
    if df is None or len(df) == 0:
        raise RuntimeError("sina_no_data")
    d0 = df.iloc[0].to_dict()
    return {
        "net_profit_yoy": _find_first_float(d0, ["净利润同比增长率", "净利润增长率", "归母净利润同比", "净利润同比"]),
        "revenue_yoy":    _find_first_float(d0, ["营业收入同比增长率", "营业收入同比", "营收同比"]),
        "gross_margin":   _find_first_float(d0, ["销售毛利率", "毛利率", "综合毛利率"]),
        "roe":            _find_first_float(d0, ["净资产收益率", "ROE", "加权净资产收益率", "ROE加权"]),
        "debt_ratio":     _find_first_float(d0, ["资产负债率", "负债率"]),
    }

def score_earnings_multi(symbol: str, metric_weights: Dict[str, float]):
    """
    东财优先，新浪兜底；退避重试=3，指数退避；当日缓存
    """
    # —— 当日缓存命中 —— 
    cache = _cache_load()
    key = f"{symbol}:{datetime.now().strftime('%Y-%m-%d')}"
    if key in cache:
        d = cache[key]
        w = metric_weights or {}
        w_net  = float(w.get("net_profit_yoy", 0.35))
        w_rev  = float(w.get("revenue_yoy",    0.20))
        w_gm   = float(w.get("gross_margin",   0.20))
        w_roe  = float(w.get("roe",            0.20))
        w_debt = float(w.get("debt_ratio",     0.05))
        w_sum  = max(1e-9, w_net+w_rev+w_gm+w_roe+w_debt)
        total = (d["s_net"]*w_net + d["s_rev"]*w_rev + d["s_gm"]*w_gm + d["s_roe"]*w_roe + d["s_debt"]*w_debt) / w_sum
        return float(_clip(total, 0.0, 100.0)), None

    if ak is None:
        return 50.0, "akshare未安装"

    last_err = None
    for attempt in range(3):
        try:
            # 1) Eastmoney 优先
            try:
                em = _fetch_from_eastmoney(symbol)
                # 映射为 0~100
                s_net  = _score_growth_pct(em["net_profit_yoy"])
                s_rev  = _score_growth_pct(em["revenue_yoy"])
                s_gm   = _score_gross_margin(em["gross_margin"])
                s_roe  = _score_roe(em["roe"])
                s_debt = _score_debt_ratio(em["debt_ratio"])
            except Exception as e_em:
                last_err = f"eastmoney_err:{e_em}"
                # 2) Sina 兜底
                si = _fetch_from_sina(symbol)
                s_net  = _score_growth_pct(si["net_profit_yoy"])
                s_rev  = _score_growth_pct(si["revenue_yoy"])
                s_gm   = _score_gross_margin(si["gross_margin"])
                s_roe  = _score_roe(si["roe"])
                s_debt = _score_debt_ratio(si["debt_ratio"])

            w = metric_weights or {}
            w_net  = float(w.get("net_profit_yoy", 0.35))
            w_rev  = float(w.get("revenue_yoy",    0.20))
            w_gm   = float(w.get("gross_margin",   0.20))
            w_roe  = float(w.get("roe",            0.20))
            w_debt = float(w.get("debt_ratio",     0.05))
            w_sum  = max(1e-9, w_net+w_rev+w_gm+w_roe+w_debt)
            score = (s_net*w_net + s_rev*w_rev + s_gm*w_gm + s_roe*w_roe + s_debt*w_debt) / w_sum
            score = float(_clip(score, 0.0, 100.0))

            # 当日缓存
            cache[key] = {"s_net": s_net, "s_rev": s_rev, "s_gm": s_gm, "s_roe": s_roe, "s_debt": s_debt}
            _cache_save(cache)
            return score, None

        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(0.6 * (2 ** attempt))
            else:
                break

    return 50.0, f"earnings_timeout: {last_err or 'unknown'}"

# ---------- 新闻（RSS 优先，HTML 兜底） ----------
def _pick_query_terms(sym: str, meta: Dict[str, Any]) -> List[str]:
    if isinstance(meta, dict):
        kws = meta.get("news_keywords")
        if isinstance(kws, str) and kws.strip():
            return [kws.strip()]
        if isinstance(kws, list) and kws:
            terms = [str(x).strip() for x in kws if isinstance(x, (str,int))]
            return [t for t in terms if t][:3]  # 取前3个
        name = meta.get("name")
        if isinstance(name, str) and name.strip():
            return [name.strip()]
    return [sym]

def _fetch_google_news_rss(query: str,
                           timeout_sec: int = 10,
                           retries: int = 2,
                           backoff: float = 1.8,
                           locales: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """
    Google News RSS 抓取，支持：
      - 超时配置（timeout_sec）
      - 重试+指数退避（retries, backoff）
      - 多地区参数兜底（locales: [{'hl':'zh-CN','gl':'CN','ceid':'CN:zh'}, ...]）
    任一 URL 成功即返回文本；全部失败返回 None。
    """
    if requests is None:
        return None

    from urllib.parse import quote_plus
    import random

    # 默认地区候选：CN中文 → US中文 → US英文
    locales = locales or [
        {"hl": "zh-CN", "gl": "CN", "ceid": "CN:zh"},
        {"hl": "zh-CN", "gl": "US", "ceid": "US:zh-Hans"},
        {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    ]

    proxies = {
        "http": os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("HTTPS_PROXY"),
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    for loc in locales:
        base = (f"https://news.google.com/rss/search?"
                f"hl={loc.get('hl','zh-CN')}&gl={loc.get('gl','CN')}&ceid={loc.get('ceid','CN:zh')}"
                f"&q={quote_plus(query)}")
        for attempt in range(max(1, int(retries)) + 1):
            try:
                tmo = float(timeout_sec) * (float(backoff) ** attempt)
                print(f"[RSS] GET {base} (tmo={tmo:.1f}s, try={attempt+1})", file=sys.stderr, flush=True)
                r = requests.get(base, headers=headers, timeout=tmo, proxies=proxies)
                if r.status_code == 200 and r.text:
                    return r.text
                print(f"[RSS][ERR] status {r.status_code}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[RSS][ERR] {e}", file=sys.stderr, flush=True)
            time.sleep(0.2 + random.random() * 0.3)

    return None

def _sentiment_from_text(txt: str, neg_kw: List[str], pos_kw: List[str]) -> int:
    t = (txt or "")
    # 中文关键词区分大小写意义不大，英文统一小写
    tl = t.lower()
    neg = sum(1 for k in (neg_kw or []) if isinstance(k, str) and k and (k.lower() in tl or k in t))
    pos = sum(1 for k in (pos_kw or []) if isinstance(k, str) and k and (k.lower() in tl or k in t))
    return pos - neg

# 放在 score_news 之前
def _score_news_html_fallback(q: str, sources: list, neg_kw: list, pos_kw: list, lookback_days: int = 7):
    if not sources:
        return None, "no_html_sources"
    if requests is None or BeautifulSoup is None:
        return None, "requests_or_bs4_missing"

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    total, used = 0, 0
    headers = {"User-Agent": "Mozilla/5.0"}
    proxies = {
        "http": os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("HTTPS_PROXY"),
    }

    for src in sources:
        try:
            url_tpl = src.get("url")
            if not url_tpl:
                continue
            url = url_tpl.format(q=q)
            print(f"[HTML] GET {url}", file=sys.stderr, flush=True)

            r = requests.get(url, headers=headers, timeout=8, proxies=proxies)
            if r.status_code != 200 or not r.text:
                print(f"[HTML][ERR] status {r.status_code}", file=sys.stderr, flush=True)
                continue

            soup = BeautifulSoup(r.text, "lxml")

            # 先尝试 Bing News 常见结构：h2 > a
            titles = [a.get_text(strip=True) for a in soup.select("h2 a")]

            # 兜底：页面结构变动时取所有可见 <a> 的文本
            if not titles:
                titles = [a.get_text(strip=True) for a in soup.find_all("a")]

            # 简单中文标题归一化/过滤（可根据需要扩展）
            cn_titles = [t for t in titles if isinstance(t, str) and t]

            for tt in cn_titles[:50]:
                s = _sentiment_from_text(tt, neg_kw, pos_kw)
                total += s
                used += 1

        except Exception as e:
            print(f"[HTML][ERR] {e}", file=sys.stderr, flush=True)
            continue

    if used == 0:
        return None, "html_no_items"

    avg = total / used
    sc = float(_clip(50.0 + 10.0 * avg, 0.0, 100.0))
    return sc, None


def score_news(sym: str,
               meta: Dict[str, Any],
               global_neg: List[str],
               global_pos: List[str],
               lookback_days: int = 7):
    # 注意：即使 feedparser 不在，也允许 HTML 兜底生效
    try:
        # per-symbol 附加关键词
        per_neg = meta.get("negative_keywords", []) if isinstance(meta, dict) else []
        per_pos = meta.get("positive_keywords", []) if isinstance(meta, dict) else []

        # 读取 RSS 配置（由 main() 注入到 meta）
        rss_cfg = (meta.get("rss_cfg") or {})
        rss_enabled = bool(rss_cfg.get("enabled", True))
        rss_timeout = int(rss_cfg.get("timeout_sec", 10))
        rss_retries = int(rss_cfg.get("retries", 2))
        rss_backoff = float(rss_cfg.get("backoff", 1.8))
        rss_locales = rss_cfg.get("locales") or None

        terms = _pick_query_terms(sym, meta)
        since = datetime.utcnow() - timedelta(days=lookback_days)
        all_scores = []

        # —— RSS（可禁用；且 feedparser 缺失时自动跳过）——
        if rss_enabled and (feedparser is not None):
            for term in terms:
                text = _fetch_google_news_rss(
                    term,
                    timeout_sec=rss_timeout,
                    retries=rss_retries,
                    backoff=rss_backoff,
                    locales=rss_locales
                )
                if not text:
                    continue
                feed = feedparser.parse(text)
                entries = getattr(feed, "entries", []) or []
                if not entries:
                    continue

                total = 0
                used = 0
                for e in entries[:50]:
                    dt = None
                    try:
                        if hasattr(e, "published_parsed") and e.published_parsed:
                            dt = datetime(*e.published_parsed[:6])
                        elif hasattr(e, "updated_parsed") and e.updated_parsed:
                            dt = datetime(*e.updated_parsed[:6])
                    except Exception:
                        dt = None
                    if dt and dt < since:
                        continue
                    title = getattr(e, "title", "") or ""
                    summ  = getattr(e, "summary", "") or ""
                    s = _sentiment_from_text(
                        f"{title} {summ}",
                        (global_neg or []) + (per_neg or []),
                        (global_pos or []) + (per_pos or [])
                    )
                    total += s
                    used += 1

                if used > 0:
                    avg = total / used
                    sc  = float(_clip(50.0 + 10.0 * avg, 0.0, 100.0))
                    all_scores.append(sc)

        # —— RSS 无结果：HTML 兜底 —— 
        if not all_scores:
            html_cfg = meta.get("html_scrape") if isinstance(meta, dict) else {}
            if isinstance(html_cfg, dict) and html_cfg.get("enabled"):
                last_err = None
                for t in terms:  # 多关键词依次尝试
                    sc2, err2 = _score_news_html_fallback(
                        t,
                        html_cfg.get("sources", []),
                        (global_neg or []) + (per_neg or []),
                        (global_pos or []) + (per_pos or []),
                        lookback_days=lookback_days
                    )
                    if sc2 is not None:
                        return sc2, None
                    last_err = err2
                return 50.0, (last_err or "no_rss_html")
            return 50.0, "无新闻"

        # —— RSS 有结果：返回均值 —— 
        return float(sum(all_scores) / len(all_scores)), None

    except Exception as e:
        return 50.0, f"新闻打分出错: {e}"

# ---------- 模型分 ----------
def _load_model_detail_map(model_json_path: str) -> dict:
    try:
        if not os.path.exists(model_json_path):
            return {}
        with open(model_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "details" in obj and isinstance(obj["details"], list):
            return {d["symbol"]: d for d in obj["details"] if isinstance(d, dict) and "symbol" in d}
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}

def load_model_score(symbol: str, model_json_path: str):
    m = _load_model_detail_map(model_json_path)
    d = m.get(symbol, {})
    try:
        if "score" in d and d["score"] is not None:
            return float(d["score"]), None
        if "prob" in d and d["prob"] is not None:
            val = float(d["prob"])
            if val <= 1.0:
                val *= 100.0
            return val, None
    except Exception:
        pass
    return 50.0, "无模型分数"

def load_model_raw_mode(symbol: str, model_json_path: str):
    m = _load_model_detail_map(model_json_path)
    d = m.get(symbol, {})
    raw = d.get("raw", None)
    mode = d.get("mode", None)
    return raw, mode

# ---------- 主函数 ----------
def main():
    import argparse, json, os, sys, threading
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "signals_config.yaml"))
    ap.add_argument("--model_json", default=os.path.join("logs","model_scores.json"))
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6, help="并发线程数（默认6）")
    args = ap.parse_args()

    # 读取配置
    conf = load_yaml(args.config)

    # —— ① 代理自动注入 + 可达性探测 —— 
    px = conf.get("proxy") or {}
    http_p  = px.get("http")  or os.environ.get("HTTP_PROXY")  or ""
    https_p = px.get("https") or os.environ.get("HTTPS_PROXY") or ""

    def _probe_proxy(url="https://www.bing.com", timeout=3):
        try:
            if requests is None:
                return False
            _proxies = {}
            if http_p:  _proxies["http"]  = http_p
            if https_p: _proxies["https"] = https_p
            r = requests.get(url, timeout=timeout, proxies=_proxies)
            return (r.status_code == 200)
        except Exception:
            return False

    if http_p or https_p:
        ok = _probe_proxy()
        if ok:
            if http_p:  os.environ["HTTP_PROXY"]  = http_p
            if https_p: os.environ["HTTPS_PROXY"] = https_p
            print(f"[INFO] proxy enabled http={http_p} https={https_p}", file=sys.stderr)
        else:
            os.environ.pop("HTTP_PROXY",  None)
            os.environ.pop("HTTPS_PROXY", None)
            print("[WARN] proxy configured but unreachable; disabled for this run.", file=sys.stderr)
    else:
        os.environ.pop("HTTP_PROXY",  None)
        os.environ.pop("HTTPS_PROXY", None)
        print("[INFO] proxy not set", file=sys.stderr)

    # —— ② 符号、权重与参数 —— 
    SYMS = symbols_dict(conf)
    print(f"[INFO] loaded symbols: {len(SYMS)}", file=sys.stderr)

    W = conf.get("weights", {}) or {}
    neg_kw = conf.get("negative_keywords", []) or []
    pos_kw = conf.get("positive_keywords", []) or []

    news_cfg = conf.get("news", {}) or {}
    lookback_days = int(news_cfg.get("lookback_days", 7))

    earnings_cfg = conf.get("earnings_metrics", {}) or {}
    metric_weights = (earnings_cfg.get("weights") or {})

    # —— ③ 并发执行（一定要在 main() 内）——
    total_syms = len(SYMS)
    items = list(SYMS.items())

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "block_threshold": float(conf.get("thresholds", {}).get("block_threshold", 65)),
            "warn_threshold":  float(conf.get("thresholds", {}).get("warn_threshold", 55)),
            "news_lookback_days": lookback_days,
        }
    }

    out_lock = threading.Lock()
    progress = {"n": 0}

    def _worker(pair):
        sym, meta = pair
        try:
            if not isinstance(meta, dict):
                meta = {}
            meta_local = dict(meta)

            # 合并全局 HTML/RSS 配置到 per-symbol
            global_html = (conf.get("news") or {}).get("html_scrape") or {}
            meta_local.setdefault("html_scrape", global_html)
            rss_cfg_global = ((conf.get("news") or {}).get("rss") or {})
            meta_local.setdefault("rss_cfg", {
                "enabled": bool(rss_cfg_global.get("enabled", True)),
                "timeout_sec": int(rss_cfg_global.get("timeout_sec", 10)),
                "retries": int(rss_cfg_global.get("retries", 2)),
                "backoff": float(rss_cfg_global.get("backoff", 1.8)),
                "locales": rss_cfg_global.get("locales") or None,
            })

            # 价格 / 财报 / 新闻 / 模型
            ps, _   = score_price(sym, 20)
            es, ee  = run_with_timeout(score_earnings_multi, args.timeout, sym, metric_weights)
            ns, ne  = run_with_timeout(score_news, args.timeout, sym, meta_local, neg_kw, pos_kw, lookback_days)
            ms, _   = load_model_score(sym, args.model_json)
            mraw, mmode = load_model_raw_mode(sym, args.model_json)

            total = (
                float(ps) * float(W.get("price_trend", 0)) +
                float(es) * float(W.get("earnings_trend", 0)) +
                float(ns) * float(W.get("news_sentiment", 0)) +
                float(ms) * float(W.get("model_score", 0))
            )
            res = {
                "price_score": round(float(ps), 2),
                "earnings_score": round(float(es), 2), "earnings_err": ee,
                "news_score": round(float(ns), 2),     "news_err": ne,
                "model_score": round(float(ms), 2),
                "model_raw": mraw, "model_mode": mmode,
                "total": round(float(total), 2),
            }
        except Exception as ex:
            res = {
                "price_score": 50.0, "earnings_score": 50.0, "earnings_err": str(ex),
                "news_score": 50.0,  "news_err": str(ex),
                "model_score": 50.0, "model_raw": None, "model_mode": None,
                "total": 50.0,
            }

        # 线程安全写结果 & 打印进度
        with out_lock:
            progress["n"] += 1
            print(f"[RUN] {progress['n']}/{total_syms} {sym}", file=sys.stderr, flush=True)
            out[sym] = res

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for _ in as_completed([ex.submit(_worker, it) for it in items]):
            pass

    # —— ④ 输出 JSON（给 orchestrator 读取）——
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
