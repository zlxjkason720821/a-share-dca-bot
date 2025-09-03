@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 固定到脚本目录
cd /d %~dp0
if not exist logs mkdir logs
set LOG=logs\predict_run_%DATE:~0,4%-%DATE:~5,2%-%DATE:~8,2%.log

echo [START] %DATE% %TIME% > "%LOG%"
echo 工作目录: %CD% >> "%LOG%"

REM 激活 venv（无则用系统 python/py）
set VENV_ACT=.venv\Scripts\activate
set PYTHON_EXE=python
if exist "%VENV_ACT%" (
  echo 激活虚拟环境 .venv ... >> "%LOG%"
  call "%VENV_ACT%" >> "%LOG%" 2>&1
) else (
  where python >> "%LOG%" 2>&1
  if errorlevel 1 set PYTHON_EXE=py
)

REM === 配置模型目录与名称（按你的训练产物命名） ===
set MODEL_DIR=models
set MODEL_NAME=a_stock
set OUT_JSON=logs\model_scores.json
set OUT_CSV=logs\model_scores.csv

REM === 自动扫描 data 下的全部标的（由 predict_model.py 自己完成）
echo 运行 predict_model.py（自动扫描 data/*.csv） ... >> "%LOG%"
%PYTHON_EXE% tools\predict_model.py ^
  --symbols_from_data ^
  --model_dir %MODEL_DIR% ^
  --model_name %MODEL_NAME% ^
  --out_json %OUT_JSON% ^
  --out_csv %OUT_CSV% >> "%LOG%" 2>&1

set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo [ERROR] predict_model.py 退出码 %ERR% >> "%LOG%"
  exit /b %ERR%
)

echo [DONE] %DATE% %TIME% >> "%LOG%"
echo 成功完成，日志见：%LOG%
exit /b 0
