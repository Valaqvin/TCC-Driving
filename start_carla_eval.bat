@echo off
REM ============================================================
REM start_carla_eval.bat — Start CARLA and run bargaining evaluation
REM ============================================================
echo ============================================================
echo   CoLMDriver + Rubinstein Bargaining — CARLA Evaluation
echo ============================================================
echo.

set CARLA_DIR=%~dp0simulation\carla\WindowsNoEditor
set RESULT_DIR=%~dp0Result\carla_eval_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RESULT_DIR=%RESULT_DIR: =0%

REM Step 1: Check CARLA installation
if not exist "%CARLA_DIR%\CarlaUE4.exe" (
    echo [ERROR] CARLA not found at %CARLA_DIR%
    echo Please download CARLA 0.9.15 for Windows from:
    echo   https://github.com/carla-simulator/carla/releases/tag/0.9.15
    echo And extract to: %CARLA_DIR%
    pause
    exit /b 1
)

REM Step 2: Check checkpoints
if not exist "ckpt\codriving\perception" (
    echo [WARNING] Perception checkpoint not found in ckpt\codriving\perception
    echo Place the CoLMDriver pre-trained perception model there before running.
)
if not exist "ckpt\codriving\planning\epoch_26.ckpt" (
    echo [WARNING] Planning checkpoint not found
)

REM Step 3: Merge routes if needed
if not exist "simulation\leaderboard\data\routes_codriving.xml" (
    echo [INFO] Merging route files...
    python merge_routes.py
)

REM Step 4: Start CARLA server in background
echo [INFO] Starting CARLA server on port 2000...
start "CARLA Server" "%CARLA_DIR%\CarlaUE4.exe" -quality-level=Low -fps=10 -windowed -carla-rpc-port=2000
echo [INFO] Waiting for CARLA server to initialize...
timeout /t 15 /nobreak > nul

REM Step 5: Run evaluation
echo [INFO] Starting bargaining evaluation...
echo [INFO] Results will be saved to: %RESULT_DIR%
mkdir %RESULT_DIR% 2>nul

"%USERPROFILE%\.conda\envs\carla37\python.exe" launch_bargaining_eval.py ^
    --carla-port 2000 ^
    --ego-num 2 ^
    --repetitions 1 ^
    --record %RESULT_DIR% ^
    --show-viz

echo.
echo [DONE] Evaluation complete. Results in: %RESULT_DIR%
pause
