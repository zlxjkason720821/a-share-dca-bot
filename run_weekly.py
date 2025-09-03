# -*- coding: utf-8 -*-
"""
run_weekly.py — 周度管线：可选抓数 → 预测 → 选股落表
- 使用当前 Python 解释器（推荐先激活 .venv）
- 日志分步落盘；失败时抛出并提示日志路径
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # 建议先在命令行里激活 .venv
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

def run(cmd, log_path: Path, must_exist: bool = True):
    if must_exist and not Path(cmd[1]).exists():
        raise FileNotFoundError(f"Not found: {cmd[1]}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(map(str, cmd))}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\nSee log: {log_path}")

def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 0) 可选：抓取/增量更新 CSV（你已有计划任务；这里作为兜底，可注释掉）
   # fetch_py = ROOT / "fetch_csv.py"
   # if fetch_py.exists():
      #  run([PYTHON, str(fetch_py)], LOGS / f"fetch_{now}.log", must_exist=False)

    ## 1) 预测：用 data/*.csv 自动扫描（生成 logs/model_scores.json）
   # predict_py = ROOT / "tools" / "predict_model.py"
    #predict_cmd = [
      #  PYTHON, str(predict_py),
     #   "--symbols_from_data",
    #    "--model_dir", "models",
      #  "--model_name", "a_stock",
      #  "--timeout", "10",
        # 如使用池化（单模型）再打开： "--pooled",
   # ]
   # run(predict_cmd, LOGS / f"predict_{now}.log")

    # 2) 选股与报表：调用根目录 orchestrator.py，明确传入 budget/signals
    orch_py = ROOT / "orchestrator.py"  # 注意：在根目录
    orch_cmd = [
        PYTHON, str(orch_py),
        "--budget", "budget.yaml",
        "--signals", "signals_config.yaml",
    ]
    run(orch_cmd, LOGS / f"orchestrator_{now}.log")

    print("[OK] Weekly pipeline finished.")
    print(f"  - Predict log   : {LOGS / f'predict_{now}.log'}")
    print(f"  - Orchestrator  : {LOGS / f'orchestrator_{now}.log'}")
    print("  - Reports written to logs/DCA_Report.xlsx and logs/weekly_budget/*.xlsx")

if __name__ == "__main__":
    main()
