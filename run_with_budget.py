import os, glob, argparse, subprocess, sys
import pandas as pd
from datetime import datetime
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def sum_invested(log_dir: str) -> float:
    """统计历史已投入金额。
    优先用 CSV 里若存在的 fill_qty*fill_price；否则用 notional 累加。
    只统计该计划目录下的 manual_orders_*.csv。
    """
    total = 0.0
    files = sorted(glob.glob(os.path.join(log_dir, "manual_orders_*.csv")))
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        # 兼容字段
        if "fill_qty" in df.columns and "fill_price" in df.columns:
            amt = (df["fill_qty"].fillna(0) * df["fill_price"].fillna(0)).sum()
        else:
            amt = df.get("notional", pd.Series([0]*len(df))).fillna(0).sum()
        total += float(amt)
    return float(total)

def periods_used(log_dir: str) -> int:
    """估算已执行期数：按该计划目录下，存在 CSV 的不同日期计数。"""
    files = sorted(glob.glob(os.path.join(log_dir, "manual_orders_*.csv")))
    dates = set()
    for fp in files:
        base = os.path.basename(fp)
        # manual_orders_YYYYMMDD.csv
        try:
            ymd = base.split("_")[2].split(".")[0]
            datetime.strptime(ymd, "%Y%m%d")
            dates.add(ymd)
        except Exception:
            pass
    return len(dates)

def make_temp_config(manual_log_dir: str, legs: list) -> str:
    """生成临时 config.yaml，喂给 main.py 使用。"""
    cfg = {
        "accounts": {
            "manual_a_share": {
                "broker": "manual",
                "params": {"output_dir": manual_log_dir}
            }
        },
        "plans": {
            "generated_run": {
                "account": "manual_a_share",
                "currency": "CNY",
                "legs": legs
            }
        }
    }
    path = "config.__temp__.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="budget.yaml")
    ap.add_argument("--plan", default="generated_run")
    ap.add_argument("--lot-size", type=int, default=None)
    args = ap.parse_args()

    budget = load_yaml(args.budget)
    total_budget = float(budget["total_budget"])
    periods = int(budget["periods"])
    log_dir = budget.get("log_dir", "./logs/weekly_budget")
    lot_size = args.lot_size or int(budget.get("lot_size", 100))
    weights = budget["weights"]  # dict: symbol -> weight

    ensure_dir(log_dir)

    # 已投入金额 & 已执行期数
    invested = sum_invested(log_dir)
    used = periods_used(log_dir)
    remaining_budget = max(0.0, total_budget - invested)
    remaining_periods = max(0, periods - used)

    if remaining_periods == 0 or remaining_budget <= 0:
        print(f"[INFO] 预算已用完或期数已跑满：invested={invested:.2f}, total={total_budget:.2f}, used_periods={used}/{periods}")
        return

    # 本期总可用 = 剩余预算 / 剩余期数（保证用完不超支）
    per_period_budget = remaining_budget / remaining_periods

    # 按权重拆分到每只股票，生成 legs
    legs = []
    for symbol, w in weights.items():
        notional = per_period_budget * float(w)
        legs.append({
            "symbol": symbol,
            "notional": round(notional),  # 金额取整
            "lot_size": lot_size,
            "order_type": "market"
        })

    # 生成临时 config 并调用 main.py 执行
    temp_cfg = make_temp_config(log_dir, legs)
    print(f"[INFO] 本期预算：{per_period_budget:.2f}，剩余预算：{remaining_budget:.2f}，剩余期数：{remaining_periods}")
    print(f"[INFO] 生成临时计划到 {temp_cfg}，legs = {legs}")

    # 调 main.py
    cmd = [sys.executable, "main.py", "--config", temp_cfg, "--run-plan", "generated_run"]
    r = subprocess.run(cmd, capture_output=False, text=True)
    if r.returncode == 0:
        print("[OK] 本期清单已生成。")
    else:
        print("[ERR] main.py 执行失败，returncode=", r.returncode)

if __name__ == "__main__":
    main()
