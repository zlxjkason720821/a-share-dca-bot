@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ===== Python 解释器（优先虚拟环境）=====
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

rem ===== 训练配置：就用 model_config.yaml（含100支）=====
set "TRAIN_CFG=tools\model_config.yaml"

rem ===== 时间戳 & 日志目录 =====
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyyMMdd_HHmmss\")"') do set "NOW=%%i"
if not exist "logs" mkdir "logs"

rem ===== 读取 YAML 中的 output.dir 与 output.name =====
for /f "usebackq tokens=* delims=" %%A in (`
  %PY% -c "import yaml; d=yaml.safe_load(open(r'%TRAIN_CFG%',encoding='utf-8')); print(d.get('output',{}).get('dir','models'))"
`) do set "OUT_DIR=%%A"

for /f "usebackq tokens=* delims=" %%A in (`
  %PY% -c "import yaml; d=yaml.safe_load(open(r'%TRAIN_CFG%',encoding='utf-8')); print(d.get('output',{}).get('name','a_stock'))"
`) do set "MODEL_NAME=%%A"

if not defined OUT_DIR  (echo [ERROR] cannot read output.dir & exit /b 1)
if not defined MODEL_NAME (echo [ERROR] cannot read output.name & exit /b 1)

rem ===== 清理旧模型文件（只清理当前 OUT_DIR 下本前缀）=====
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
echo [INFO] Clean old models: %OUT_DIR%\%MODEL_NAME%_*.pkl
del /q "%OUT_DIR%\%MODEL_NAME%_*.pkl" 2>nul

rem ===== 开始训练（去掉 --n-jobs）=====
echo [INFO] Train start, cfg=%TRAIN_CFG%, out=%OUT_DIR%, model=%MODEL_NAME%
%PY% "tools\train_model.py" --config "%TRAIN_CFG%" --timeout 12 > "logs\train_%NOW%.log" 2>&1

if errorlevel 1 (
  echo [ERROR] Train FAILED. See logs\train_%NOW%.log
  exit /b 1
) else (
  echo [OK] Train FINISHED. Models -> %OUT_DIR% (one .pkl per symbol). See logs\train_%NOW%.log
)

endlocal
