# -*- coding: utf-8 -*-
from sklearn.dummy import DummyClassifier
import numpy as np

"""
训练脚本（读取 model_config.yaml）：
...
"""

"""
训练脚本（读取 model_config.yaml）：
- 按配置逐票训练（或你以后扩展成池化）
- 自动对齐特征与标签
- 滚动验证并打印指标
- 保存模型到 models/<name>_<symbol>.pkl
- 训练完每票后将摘要写入 logs/train_summary.csv
"""

import os, yaml, argparse
import numpy as np
import pandas as pd

from features import make_features
from labels import label_binary_future_return, target_reg_future_return
from model import fit_classifier, fit_regressor, save_model
from eval import walk_splits, topk_precision, simple_kday_backtest
from data_loader import get_price_df


def load_config(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def build_dataset(symbols, timeout=12, use_cache=True, min_rows=800, skip_log=None):
    """返回 {symbol: DataFrame}，不足/失败的票自动跳过。并可记录跳过原因到 skip_log(list)."""
    data = {}
    for s in symbols:
        reason = None
        try:
            df = get_price_df(s, timeout_sec=timeout, use_cache=use_cache)
        except Exception as e:
            reason = f"load_error: {e}"
            print(f"[WARN] skip {s}: {reason}")
            if skip_log is not None:
                skip_log.append({"symbol": s, "reason": reason, "rows": None})
            continue
        n = 0 if df is None else len(df)
        if df is None:
            reason = "none_df"
        elif n < min_rows:
            reason = f"insufficient_rows ({n} < {min_rows})"
        if reason:
            print(f"[WARN] skip {s}: {reason}")
            if skip_log is not None:
                skip_log.append({"symbol": s, "reason": reason, "rows": n})
            continue
        data[s] = df
    if not data:
        raise RuntimeError("No usable symbols after filtering.")
    return data
def __safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def _predict_batch(obj, X, mode):
    """批量打分（与训练时相同的列与标准化），返回 raw 与 score 数组。"""
    feats = obj.get("features", list(X.columns))
    cont_cols = obj.get("cont_cols", [])
    scaler = None
    # Extract scaler if present in pipeline
    mdl = obj["model"]
    if hasattr(mdl, "named_steps") and "scaler" in mdl.named_steps:
        scaler = mdl.named_steps["scaler"]
        if scaler == "passthrough":
            scaler = None

    Xb = X.copy()
    # 缺失特征补0，并按训练列顺序对齐
    for c in feats:
        if c not in Xb.columns:
            Xb[c] = 0.0
    Xb = Xb[feats]

    # 只对连续列做标准化（与训练一致）
    if scaler is not None and cont_cols:
        Xb.loc[:, cont_cols] = scaler.transform(Xb[cont_cols])

    if mode == "clf":
        if hasattr(mdl, "predict_proba"):
            raw = mdl.predict_proba(Xb)[:, 1]
        elif hasattr(mdl, "decision_function"):
            s = mdl.decision_function(Xb)
            raw = 1.0 / (1.0 + np.exp(-s))
        else:
            raw = mdl.predict(Xb)
            raw = 1.0 / (1.0 + np.exp(-raw))
        score = (raw * 100).astype(int)
        return raw, score
    else:
        raw = mdl.predict(Xb)  # 预测未来收益（比例）
        score = np.clip(np.round(50 + raw * 200), 0, 100).astype(int)
        return raw, score


def train_for_symbol(symbol, df, cfg, cname=None):
    """训练单票：对齐→拟合→CV评估→落盘"""
    # 1) 特征
    X = make_features(df)

    # 2) 标签
    task = cfg["task"]["type"]  # "binary" | "reg"
    k = cfg["task"].get("horizon", 10)
    thr = cfg["task"].get("threshold", 0.05)

    if task == "binary":
        y = label_binary_future_return(df, k=k, thr=thr)
        y_clean = y.dropna().astype(int)
        pos = int((y_clean == 1).sum())
        neg = int((y_clean == 0).sum())
        total = int(len(y_clean))
        mode = "clf"

        # 记录类别占比（仅信息提示，不做强制跳过；单一类别由 fit_classifier 兜底）
        print(f"[INFO] {symbol} label dist: pos={pos}, neg={neg}, total={total}")
    else:
        y = target_reg_future_return(df, k=k)
        mode = "reg"
    # —— 在对齐和CV之前做一次“全局单一类别”兜底 —— 
    if task == "binary":
        y_uniq = y_clean.unique()
        if len(y_uniq) < 2:
            # 构造一个 DummyClassifier，并把它打包成与 fit_classifier 返回一致的 obj
            majority = 1 if (y_clean == 1).sum() >= (y_clean == 0).sum() else 0
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            # 先对齐一次（用原有流程变量名）
            X_glob, y_glob = X.align(y, join="inner", axis=0)
            X_glob = X_glob.replace([np.inf, -np.inf], np.nan)

            cont_cols = [c for c in X_glob.columns if np.issubdtype(X_glob[c].dtype, np.number)]
            pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler() if cont_cols else "passthrough"),
                ("clf", DummyClassifier(strategy="constant", constant=majority)),
            ])
            # 只用非空标签样本拟合
            mask = ~y_glob.isna()
            pipe.fit(X_glob.loc[mask], y_glob.loc[mask].astype(int))

            obj = {
                "model": pipe,
                "report": {
                    "cv_auc_mean": np.nan,
                    "cv_auc_std": np.nan,
                    "cv_acc_mean": float((y_clean == majority).mean()) if len(y_clean) else np.nan,
                    "cv_acc_std": 0.0,
                    "n_samples": int(X_glob.shape[0]),
                    "n_features": int(X_glob.shape[1]),
                    "n_pos": int((y_clean == 1).sum()),
                    "n_neg": int((y_clean == 0).sum()),
                    "note": "single-class-global; fallback=DummyClassifier",
                },
                "features": list(X_glob.columns),
                "cont_cols": cont_cols,
            }

            # 直接进入保存与汇总，跳过后面的常规CV逻辑
            out_dir = cfg["output"]["dir"]
            name = cfg["output"]["name"]
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{name}_{symbol}.pkl")
            save_model(obj, out_path)
            print(f"  → Saved (dummy): {out_path}")

            # 写入 train_summary.csv
            try:
                rep = obj.get("report", {})
                row = {
                    "symbol": symbol,
                    "mode": mode,
                    "n_samples": rep.get("n_samples"),
                    "n_features": rep.get("n_features"),
                    "n_pos": rep.get("n_pos"),
                    "n_neg": rep.get("n_neg"),
                    "cv_auc_mean": rep.get("cv_auc_mean"),
                    "cv_acc_mean": rep.get("cv_acc_mean"),
                    "cv_r2_mean": rep.get("cv_r2_mean"),
                    "note": rep.get("note"),
                }
                os.makedirs("logs", exist_ok=True)
                sum_path = os.path.join("logs", "train_summary.csv")
                pd.DataFrame([row]).to_csv(sum_path, mode="a", header=not os.path.exists(sum_path), index=False, encoding="utf-8")
            except Exception:
                pass

            return obj, mode

    # 3) 显式对齐（行交集）
    X, y = X.align(y, join="inner", axis=0)
    # 对齐后，先把 X 中的 inf/-inf 置为 NaN，防止后续评估中提前炸掉
    X = X.replace([np.inf, -np.inf], np.nan)
    # 不在这里 dropna（让 fit_* 里统一清洗并与 y 一起处理）

    # 4) 基础检查
    min_need = cfg["cv"]["train_len"] + cfg["cv"]["test_len"]
    if len(X) < max(100, min_need):
        print(f"[WARN] {symbol} 数据较少（对齐后 {len(X)} 行），效果可能不稳。")

    # 5) 拟合整段样本（产出最终可用模型）
    if mode == "clf":
        obj = fit_classifier(X, y)
    else:
        obj = fit_regressor(X, y)



    # 6) 滚动验证（不污染最终模型；每折单独拟合）
    train_len = cfg["cv"]["train_len"]; test_len = cfg["cv"]["test_len"]; step = cfg["cv"]["step"]
    close = df["close"].loc[X.index] if "close" in df.columns else None

    scores_all = []

    for tr, te in walk_splits(X, train_len, test_len, step):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr]
        yte = y.iloc[te]

    if mode == "clf":
        ytr_clean = ytr.dropna().astype(int)
        if ytr_clean.nunique() < 2:
            # 单折兜底：用 DummyClassifier 训练本折
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            majority = 1 if (ytr_clean == 1).sum() >= (ytr_clean == 0).sum() else 0
            cont_cols_fold = [c for c in Xtr.columns if np.issubdtype(Xtr[c].dtype, np.number)]
            pipe_fold = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler() if cont_cols_fold else "passthrough"),
                ("clf", DummyClassifier(strategy="constant", constant=majority)),
            ])
            pipe_fold.fit(Xtr, ytr_clean)
            proba = pipe_fold.predict_proba(Xte)[:, 1]
        else:
            fold_obj = fit_classifier(Xtr, ytr)
            proba, _ = _predict_batch(fold_obj, Xte, mode="clf")

        y_true = yte.values.astype(int)
        try:
            auc = float(np.nan_to_num(__safe_auc(y_true, proba)))
        except Exception:
            auc = np.nan

        ptop = float(topk_precision(y_true, proba, k_ratio=0.2))
        ann, mdd = (np.nan, np.nan)
        if close is not None:
            ann, mdd = simple_kday_backtest(close.loc[Xte.index], proba, k=k, pick_ratio=0.2)
        scores_all.append({"AUC": auc, "P@20%": ptop, "Ann": ann, "MDD": mdd})


        show_name = f"{symbol}（{cname}）" if cname else symbol
        print(f"[{show_name}] CV metrics:")
        if scores_all:
            for i, m in enumerate(scores_all, 1):
                print(f"  Fold{i}: {m}")
            avg = {k: float(np.nanmean([d.get(k, np.nan) for d in scores_all])) for k in scores_all[0]}
            print(f"  AVG : {avg}")
        else:
            print("  [WARN] CV 无有效折（样本可能不足）。")

    # 7) 保存模型
    out_dir = cfg["output"]["dir"]
    name = cfg["output"]["name"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_{symbol}.pkl")
    save_model(obj, out_path)
    print(f"  → Saved: {out_path}")

    # 8) 记录简要训练摘要（便于排查单一类别/数据量少）
    try:
        rep = obj.get("report", {}) if isinstance(obj, dict) else {}
        row = {
            "symbol": symbol,
            "mode": mode,                # clf / reg
            "n_samples": rep.get("n_samples"),
            "n_features": rep.get("n_features"),
            "n_pos": rep.get("n_pos"),
            "n_neg": rep.get("n_neg"),
            "cv_auc_mean": rep.get("cv_auc_mean"),
            "cv_acc_mean": rep.get("cv_acc_mean"),
            "cv_r2_mean": rep.get("cv_r2_mean"),
            "note": rep.get("note"),
        }
        os.makedirs("logs", exist_ok=True)
        sum_path = os.path.join("logs", "train_summary.csv")
        pd.DataFrame([row]).to_csv(sum_path, mode="a", header=not os.path.exists(sum_path), index=False, encoding="utf-8")
    except Exception:
        pass

    return obj, mode



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "model_config.yaml"))
    ap.add_argument("--timeout", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # symbol -> 中文名（如果有）
    name_map = {s: (meta.get("name") if isinstance(meta, dict) else None)
                for s, meta in cfg.get("symbols", {}).items()}
    symbols = list(name_map.keys())

    # 从配置读取最少行数（默认 800）
    data_min_rows = int(cfg.get("data", {}).get("min_rows", 800))

    # 收集被跳过的原因
    skip_log = []
    data = build_dataset(symbols, timeout=args.timeout, use_cache=True, min_rows=data_min_rows, skip_log=skip_log)

    trained, modes = 0, {"clf": 0, "reg": 0}
    for sym, df in data.items():
        cname = name_map.get(sym)
        try:
            obj, mode = train_for_symbol(sym, df, cfg, cname=cname)
            trained += 1
            modes[mode] = modes.get(mode, 0) + 1
        except Exception as e:
            # 训练阶段的报错也记入 skip_log（以便统计）
            skip_log.append({"symbol": sym, "reason": f"train_error: {e}", "rows": len(df)})
            print(f"[ERROR] {sym} 训练失败：{e}")

    total = len(symbols)
    print(f"=== SUMMARY ===")
    print(f"Total symbols in config : {total}")
    print(f"Trained successfully     : {trained}")
    print(f"Skipped                  : {total - trained} (see logs/skip_summary.csv)")

    # 写出 skip_summary.csv
    try:
        os.makedirs("logs", exist_ok=True)
        pd.DataFrame(skip_log).to_csv(os.path.join("logs", "skip_summary.csv"), index=False, encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
