@echo off
cd /d "%~dp0"

echo ========================================
echo   Starting AI Illustration Generator...
echo ========================================

rem Reuse the running server instead of starting a second process on port 7860.
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>nul
if %errorlevel%==0 (
    echo Server is already running at http://127.0.0.1:7860
    start "" "http://127.0.0.1:7860"
    exit /b 0
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 main.py
) else (
    python main.py
)
pause
