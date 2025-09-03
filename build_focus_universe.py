# -*- coding: utf-8 -*-
"""
从全 A（或全上证）里挑选 TopN 上证标的，生成：
- tools/model_config.focus.yaml   (只含重点池 symbols)
- budget.focus.yaml               (只含重点池 weights，等权或按市值占比)
依赖：akshare (pip install akshare)

用法示例：
python tools/build_focus_universe.py --topk 100 --weight cap --market sh
"""
import os, yaml, argparse, datetime
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
OUT_MODEL = os.path.join(TOOLS, "model_config.focus.yaml")
OUT_BUDGET = os.path.join(ROOT, "budget.focus.yaml")

def get_shanghai_symbols_via_ak() -> pd.DataFrame:
    import akshare as ak
    # 获取 A 股代码与名称
    base = ak.stock_info_a_code_name()  # columns: ['code','name']
    # 获取 A 股当日快照（含市值、成交额等）
    spot = ak.stock_zh_a_spot_em()      # columns 含 '代码','名称','总市值','流通市值','成交额'
    # 规范列
    base["symbol"] = base["code"].astype(str).apply(lambda x: (x + ".SH") if x.startswith("6") else (x + ".SZ"))
    base = base[["symbol","name"]]
    # 只留上证
    base = base[base["symbol"].str.endswith(".SH")].copy()
    # 合并市值/成交额
    spot = spot.rename(columns={"代码":"raw","名称":"name","总市值":"mkt_cap","成交额":"amount"})
    # 统一 symbol 形式
    spot["symbol"] = spot["raw"].astype(str).apply(lambda x: (x + ".SH") if x.startswith("6") else (x + ".SZ"))
    spot = spot[["symbol","mkt_cap","amount"]]
    df = pd.merge(base, spot, on="symbol", how="left")
    # mkt_cap/amount 可能是中文带单位的字符串，这里尽量转数字
    for c in ["mkt_cap","amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df  # columns: symbol, name, mkt_cap, amount

def select_top_shanghai(df: pd.DataFrame, topk: int, weight_metric: str) -> pd.DataFrame:
    # weight_metric: 'cap'（按总市值排序） | 'amount'（按成交额排序） | 'equal'
    df = df.copy()
    if weight_metric == "amount":
        df = df.sort_values(by="amount", ascending=False)
    else:
        df = df.sort_values(by="mkt_cap", ascending=False)
    if topk > 0:
        df = df.head(topk)
    df = df.reset_index(drop=True)
    # weights:
    if weight_metric == "equal":
        df["weight_raw"] = 1.0
    else:
        key = "mkt_cap" if weight_metric == "cap" else "amount"
        v = df[key].fillna(0)
        if v.sum() > 0:
            df["weight_raw"] = v / v.sum()
        else:
            df["weight_raw"] = 1.0
    # 归一到近似整数比例，便于阅读
    df["weight"] = (df["weight_raw"] / df["weight_raw"].sum()).round(6)
    return df

def write_model_config_focus(df: pd.DataFrame, base_model_cfg_path: str, out_path: str):
    # 读取原 model_config.yaml，复用除 symbols 外的其它参数
    with open(base_model_cfg_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base = base or {}
    base["symbols"] = { sym: {} for sym in df["symbol"].tolist() }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, allow_unicode=True, sort_keys=False)
    print(f"[OK] model_config.focus.yaml -> {out_path} ({len(df)} symbols)")

def write_budget_focus(df: pd.DataFrame, base_budget_path: str, out_path: str):
    with open(base_budget_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base = base or {}
    # 只保留 weights，按选出的重点池覆盖
    base["weights"] = { sym: float(w) for sym, w in zip(df["symbol"], df["weight"]) }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, allow_unicode=True, sort_keys=False)
    print(f"[OK] budget.focus.yaml -> {out_path} ({len(base['weights'])} weights)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100, help="Top N Shanghai A-shares")
    ap.add_argument("--weight", choices=["cap","amount","equal"], default="cap",
                    help="Focus weights: cap(市值) / amount(成交额) / equal(等权)")
    ap.add_argument("--market", choices=["sh"], default="sh")
    ap.add_argument("--model_cfg", default=os.path.join(TOOLS, "model_config.yaml"))
    ap.add_argument("--budget_cfg", default=os.path.join(ROOT, "budget.yaml"))
    args = ap.parse_args()

    df_all = get_shanghai_symbols_via_ak()
    if df_all.empty:
        raise SystemExit("Failed to load Shanghai universe via akshare.")
    df_sel = select_top_shanghai(df_all, topk=args.topk, weight_metric=args.weight)

    write_model_config_focus(df_sel, args.model_cfg, OUT_MODEL)
    write_budget_focus(df_sel, args.budget_cfg, OUT_BUDGET)

    # 顺便导出一个 txt 方便肉眼核对
    txt = os.path.join(ROOT, "logs", f"focus_sh_{args.topk}_{args.weight}.txt")
    os.makedirs(os.path.dirname(txt), exist_ok=True)
    with open(txt, "w", encoding="utf-8") as f:
        for s in df_sel["symbol"]:
            f.write(s + "\n")
    print(f"[OK] focus list dump -> {txt}")

if __name__ == "__main__":
    main()
