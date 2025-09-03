@echo off
REM ===========================
REM dca_bot - 每日抓取CSV（首建全量+增量追加）
REM 1) 固定工作目录到脚本所在处
REM 2) 激活本地虚拟环境 .venv
REM 3) 执行 fetch_csv.py，输出到日志
REM ===========================

chcp 65001 >nul
setlocal enabledelayedexpansion

REM ---- 定位到脚本所在目录（不论从哪启动都OK）
cd /d %~dp0

REM ---- 目录准备
if not exist logs mkdir logs
set LOG=logs\fetch_run_%DATE:~0,4%-%DATE:~5,2%-%DATE:~8,2%.log

echo [START] %DATE% %TIME% > "%LOG%"
echo 工作目录: %CD% >> "%LOG%"

REM ---- 查找 Python & venv
set VENV_ACT=.venv\Scripts\activate
set PYTHON_EXE=python

if exist "%VENV_ACT%" (
  echo 激活虚拟环境 .venv ... >> "%LOG%"
  call "%VENV_ACT%" >> "%LOG%" 2>&1
) else (
  echo [警告] 未发现 .venv\Scripts\activate，尝试系统 Python/py 启动 >> "%LOG%"
  where python >> "%LOG%" 2>&1
  if errorlevel 1 (
    set PYTHON_EXE=py
  )
)

REM ---- 执行抓取脚本（fetch_csv.py：首建全量，已存在则增量只补新交易日）
echo 运行 fetch_csv.py ... >> "%LOG%"
%PYTHON_EXE% fetch_csv.py >> "%LOG%" 2>&1
set ERR=%ERRORLEVEL%

if %ERR% NEQ 0 (
  echo [ERROR] fetch_csv.py 退出码 %ERR% >> "%LOG%"
  echo [ERROR] 抓取失败，详见 %LOG%
  exit /b %ERR%
)

echo [DONE] %DATE% %TIME% >> "%LOG%"
echo 成功完成，日志见：%LOG%
exit /b 0
