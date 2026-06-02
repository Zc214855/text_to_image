@echo off
cd /d "%~dp0"

taskkill /F /IM python.exe >nul 2>&1

set PYTHON=
for %%p in (
    "python"
    "python3"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) do (
    where %%~p >nul 2>&1 && set "PYTHON=%%~p" && goto :found_python
)
echo ERROR: Python not found! Please install Python 3.11+
pause
exit /b 1
:found_python
echo Found Python: %PYTHON%

echo ========================================
echo   Installing dependencies...
echo ========================================
"%PYTHON%" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo ========================================
echo   Starting AI Illustration Generator...
echo ========================================
"%PYTHON%" main.py
if errorlevel 1 (
    echo.
    echo ERROR: Program exited with errors!
)
pause
