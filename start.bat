@echo off
cd /d "%~dp0"

echo ========================================
echo   Starting AI Illustration Generator...
echo ========================================
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 main.py
) else (
    python main.py
)
pause
