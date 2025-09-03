# -*- coding: utf-8 -*-
"""
orchestrator.py — 交易编排（复用 tools/check_signals.py 打分）
步骤：
1) 调用 tools/check_signals.py 获取每只股票的综合分（price/earnings/news/model/total）
2) 选出 TopN（默认5）
3) 按总预算做按比分配（A股一手=100股），计算买入手数/成本/预计7日收益
4) 写入 Excel：
   - logs/DCA_Report.xlsx（累计）
   - logs/weekly_budget/Report_YYYYMMDD.xlsx（当周快照）

依赖：
- Python 3.9+
- pandas, openpyxl（写Excel用）
- data/{SYMBOL}.csv 用于取最新收盘价（列名自动识别 "收盘"/"Close"）

预算文件（示例 budget.yaml，字段都可选有默认）：
budget:
  total: 20000         # 总预算（元）
  lot_size: 100        # A股一手=100
  min_cash_reserve: 0  # 预留现金，不参与分配
fees:
  broker_bps: 2.5      # 券商买入费（bps=万分之，2.5 bps=0.025%）
  min_fee: 5           # 单笔最低买入手续费
  sell_stamp_bps: 10   # 卖出印花税（仅卖出，默认10 bps=0.1%）
  sell_broker_bps: 2.5 # 卖出券商费（bps），不填用 broker_bps
forecast:
  horizon_days: 7
  score_to_ret_max: 0.10   # total=100 映射 +10%，total=0 映射 -10%，50→0%
  floor_lots_if_affordable: true  # 有预算至少买1手（若买得起）
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# ------------ 路径 ------------
ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
DATA_DIR = ROOT / "data"
LOG_DIR  = ROOT / "logs"
WEEKLY_DIR = LOG_DIR / "weekly_budget"
LOG_DIR.mkdir(exist_ok=True, parents=True)
WEEKLY_DIR.mkdir(exist_ok=True, parents=True)

# ------------ 小工具 ------------
def load_yaml(path: Path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _last_close_from_csv(symbol: str) -> float:
    """从 data/{symbol}.csv 里拿最新收盘价（兼容“收盘”/“Close”/最后一列）。"""
    p = DATA_DIR / f"{symbol}.csv"
    if not p.exists():
        raise FileNotFoundError(f"no csv for {symbol}")
    df = pd.read_csv(p)
    cols = df.columns.tolist()
    if "收盘" in cols:
        s = pd.to_numeric(df["收盘"], errors="coerce").dropna()
    elif "Close" in cols:
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    else:
        s = pd.to_numeric(df[cols[-1]], errors="coerce").dropna()
    if len(s) == 0:
        raise ValueError(f"no close price in csv for {symbol}")
    return float(s.iloc[-1])

def _run_check_signals(signals_yaml: Path, model_json: Path,
                       timeout_sec: int = 12, workers: int = 6) -> dict:
    """
    调用 tools/check_signals.py，拿回 stdout 的 JSON。
    这将复用你在 check_signals.py 里实现的所有增强逻辑（东财优先/代理/新闻兜底等）。
    """
    cmd = [
        sys.executable, str(TOOLS / "check_signals.py"),
        "--config", str(signals_yaml),
        "--model_json", str(model_json),
        "--timeout", str(timeout_sec),
        "--workers", str(workers),

    ]
    proc = None
    try:
        proc = __import__("subprocess").run(
            cmd, cwd=str(ROOT), capture_output=True, text=True
        )
    except Exception as e:
        raise RuntimeError(f"failed to spawn check_signals: {e}")

    if proc.returncode != 0:
        raise RuntimeError(
            "check_signals failed.\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout[:2000]}"
        )

    # 保留 stderr 作为进度日志显示在控制台
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    try:
        data = json.loads(proc.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"parse check_signals JSON failed: {e}\nSTDOUT head:\n{proc.stdout[:2000]}")
    return data

def _score_to_expected_return(total_score: float, cfg: dict) -> float:
    """
    把综合分 total(0~100) 映射成 7日收益率估计。
    默认线性映射：50->0%，100->+max，0->-max；max 默认 10%（可在 budget.yaml 的 forecast.score_to_ret_max 调整）
    """
    fc = (cfg or {})
    max_abs = float(fc.get("score_to_ret_max", 0.10))
    # 线性映射
    return float(max(-max_abs, min(max_abs, (total_score - 50.0) / 50.0 * max_abs)))

def _buy_fee(amount: float, fees_cfg: dict) -> float:
    """买入手续费（券商费，考虑最低 5 元）。"""
    if amount <= 0:
        return 0.0
    bps = float(fees_cfg.get("broker_bps", 2.5))  # 默认 2.5 bps = 0.025%
    min_fee = float(fees_cfg.get("min_fee", 5.0))
    fee = amount * (bps / 10000.0)
    return float(max(min_fee, fee))

def _sell_fee(amount: float, fees_cfg: dict) -> float:
    """卖出总费（印花税 + 券商费）。"""
    if amount <= 0:
        return 0.0
    stamp_bps = float(fees_cfg.get("sell_stamp_bps", 10.0))   # 交易印花税（默认10 bps=0.1%）
    broker_bps = float(fees_cfg.get("sell_broker_bps", fees_cfg.get("broker_bps", 2.5)))
    tax = amount * (stamp_bps / 10000.0)
    fee = amount * (broker_bps / 10000.0)
    return float(tax + fee)

def _alloc_lots_linear(budget: float, candidates: pd.DataFrame, lot_size: int, floor_one: bool):
    """
    线性按权重分配预算到“手数”（每手=lot_size股）。
    candidates: DataFrame[ symbol, price, weight ]
    返回 DataFrame 增加：lots, shares, cost, buy_fee, cash_used
    """
    df = candidates.copy()
    # 归一化权重（>0）
    df["weight"] = df["weight"].clip(lower=1e-9)
    df["w_norm"] = df["weight"] / df["weight"].sum()

    # 先按现金目标算“理想手数”
    df["target_cash"] = budget * df["w_norm"]
    df["lot_cash"] = df["price"] * lot_size
    df["lots"] = (df["target_cash"] / df["lot_cash"]).apply(math.floor).astype(int)

    # 可选：若买得起至少1手且 lots=0 则抬 1 手
    if floor_one:
        afford_mask = df["lot_cash"] <= df["target_cash"]
        df.loc[afford_mask & (df["lots"] == 0), "lots"] = 1

    # 根据剩余现金，按“剩余误差”再增配
    spent = (df["lots"] * df["lot_cash"]).sum()
    remain = budget - spent
    # 允许在可负担范围内，按“(target_cash - allocated_cash)”从大到小再加手
    df["alloc_cash"] = df["lots"] * df["lot_cash"]
    df["gap"] = df["target_cash"] - df["alloc_cash"]

    # 贪心补齐
    # 为避免死循环，最多尝试 N * 2 轮
    max_round = int(len(df) * 2)
    for _ in range(max_round):
        if remain < df["lot_cash"].min():
            break
        # 选择 gap 最大且还能买得起的一只
        cand = df[df["lot_cash"] <= remain].sort_values(["gap", "w_norm"], ascending=False).head(1)
        if cand.empty:
            break
        idx = cand.index[0]
        df.at[idx, "lots"] += 1
        df.at[idx, "alloc_cash"] += df.at[idx, "lot_cash"]
        df.at[idx, "gap"] = df.at[idx, "target_cash"] - df.at[idx, "alloc_cash"]
        remain -= df.at[idx, "lot_cash"]

    df["shares"] = df["lots"] * lot_size
    return df

# ------------ 主流程 ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget",  default=str(ROOT / "budget.yaml"))
    ap.add_argument("--signals", default=str(ROOT / "signals_config.yaml"))
    ap.add_argument("--model_json", default=str(LOG_DIR / "model_scores.json"))
    ap.add_argument("--topn", type=int, default=5)      # 你要的 Top5
    ap.add_argument("--timeout", type=int, default=12)  # 传给 check_signals 的每票超时
    ap.add_argument("--workers", type=int, default=6,   # 👈 一定要在 parse_args 之前
                    help="并发线程数（传给 check_signals.py）")
    args = ap.parse_args()


    budget_cfg = load_yaml(Path(args.budget)) if Path(args.budget).exists() else {}
    total_budget = float(((budget_cfg.get("budget") or {}).get("total", 20000)))
    lot_size = int(((budget_cfg.get("budget") or {}).get("lot_size", 100)))
    min_cash_reserve = float(((budget_cfg.get("budget") or {}).get("min_cash_reserve", 0.0)))
    fees_cfg = (budget_cfg.get("fees") or {})
    fc_cfg = (budget_cfg.get("forecast") or {})
    floor_one = bool(fc_cfg.get("floor_lots_if_affordable", True))

    usable_budget = max(0.0, total_budget - min_cash_reserve)

    # 1) 运行 check_signals，拿到所有分数
    scores = _run_check_signals(Path(args.signals), Path(args.model_json),
                            timeout_sec=args.timeout,
                            workers=args.workers)
    meta = scores.get("meta", {})
    rows = []
    for sym, d in scores.items():
        if sym == "meta":
            continue
        if not isinstance(d, dict):
            continue
        rows.append({
            "symbol": sym,
            "price_score": float(d.get("price_score", 50.0)),
            "earnings_score": float(d.get("earnings_score", 50.0)),
            "news_score": float(d.get("news_score", 50.0)),
            "model_score": float(d.get("model_score", 50.0)),
            "total": float(d.get("total", 50.0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no symbols from check_signals")

    # 2) TopN
    df = df.sort_values("total", ascending=False).reset_index(drop=True)
    topn = max(1, int(args.topn))
    df_top = df.head(topn).copy()

    # 3) 读取最新价格 & 预计7日收益率
    prices = []
    exp_rets = []
    for sym in df_top["symbol"]:
        price = _last_close_from_csv(sym)
        prices.append(price)
    df_top["price"] = prices
    for sc in df_top["total"]:
        exp_rets.append(_score_to_expected_return(sc, fc_cfg))
    df_top["exp_7d_ret"] = exp_rets  # 比例，如 0.06=+6%

    # 4) 分配手数（按 total 权重线性分配；避免 0 权重）
    df_alloc = _alloc_lots_linear(
        usable_budget,
        df_top[["symbol", "price"]].assign(weight=df_top["total"].clip(lower=1.0)),
        lot_size=lot_size,
        floor_one=floor_one
    )

    # 5) 费用、成本、预计收益
    df_alloc["cost"] = df_alloc["lots"] * df_alloc["price"] * lot_size
    df_alloc["buy_fee"] = df_alloc["cost"].apply(lambda x: _buy_fee(x, fees_cfg))
    df_alloc["cash_used"] = df_alloc["cost"] + df_alloc["buy_fee"]

    # 预计 7 日卖出价 / 卖出收入 / 预计净收益（扣除卖出费与印花税）
    df_alloc = df_alloc.merge(df_top[["symbol", "exp_7d_ret"]], on="symbol", how="left")
    df_alloc["exp_sell_price"] = df_alloc["price"] * (1.0 + df_alloc["exp_7d_ret"])
    df_alloc["sell_amount"] = df_alloc["shares"] * df_alloc["exp_sell_price"]
    df_alloc["sell_fee"] = df_alloc["sell_amount"].apply(lambda x: _sell_fee(x, fees_cfg))
    df_alloc["exp_net_pnl"] = df_alloc["sell_amount"] - df_alloc["cash_used"] - df_alloc["sell_fee"]
    df_alloc["exp_7d_ret_pct_on_cash"] = df_alloc["exp_net_pnl"] / df_alloc["cash_used"]

    # 6) 汇总表
    today = datetime.now().strftime("%Y-%m-%d")
    report = df_alloc[[
        "symbol","price","lots","shares","cost","buy_fee","cash_used",
        "exp_sell_price","sell_amount","sell_fee","exp_net_pnl","exp_7d_ret","exp_7d_ret_pct_on_cash"
    ]].copy()
    report.insert(0, "date", today)
    report = report.round({
        "price": 4, "exp_sell_price": 4,
        "cost": 2, "buy_fee": 2, "cash_used": 2, "sell_amount": 2, "sell_fee": 2, "exp_net_pnl": 2,
        "exp_7d_ret": 4, "exp_7d_ret_pct_on_cash": 4
    })

    # 7) 写 Excel
    # 7.1 累计报表：logs/DCA_Report.xlsx（追加）
    rpt_path = LOG_DIR / "DCA_Report.xlsx"
    if rpt_path.exists():
        with pd.ExcelWriter(rpt_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as wx:
            # 统一把累计写入一个 sheet 'Report'，以追加行的方式维护（不覆盖）
            # openpyxl 在 overlay 模式不会自动对齐行，这里简单策略：读回旧表，concat 再重写
            old = pd.read_excel(rpt_path, sheet_name="Report") if "Report" in pd.ExcelFile(rpt_path).sheet_names else pd.DataFrame()
        # 重写（避免 overlay 的奇怪行为）
        all_df = pd.concat([old, report], ignore_index=True) if not old.empty else report
        with pd.ExcelWriter(rpt_path, engine="openpyxl", mode="w") as wx:
            all_df.to_excel(wx, sheet_name="Report", index=False)
    else:
        with pd.ExcelWriter(rpt_path, engine="openpyxl", mode="w") as wx:
            report.to_excel(wx, sheet_name="Report", index=False)

    # 7.2 当周快照：logs/weekly_budget/Report_YYYYMMDD.xlsx
    snap_path = WEEKLY_DIR / f"Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(snap_path, engine="openpyxl", mode="w") as wx:
        # 第一张表：买入计划
        report.to_excel(wx, sheet_name="Plan", index=False)
        # 第二张表：评分明细（便于核对）
        df_top.sort_values("total", ascending=False).to_excel(wx, sheet_name="Scores", index=False)
        # 第三张表：元信息
        meta_df = pd.DataFrame([{
            "generated_at": meta.get("generated_at"),
            "block_threshold": meta.get("block_threshold"),
            "warn_threshold": meta.get("warn_threshold"),
            "news_lookback_days": meta.get("news_lookback_days"),
            "total_budget": total_budget,
            "usable_budget": usable_budget,
            "lot_size": lot_size,
            "min_cash_reserve": min_cash_reserve
        }])
        meta_df.to_excel(wx, sheet_name="Meta", index=False)

    print("[OK] Orchestrator finished.")
    print(f"  - Weekly snapshot: {snap_path}")
    print(f"  - Accumulated    : {rpt_path}")

if __name__ == "__main__":
    main()
