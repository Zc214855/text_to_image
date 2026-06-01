@echo off
cd /d "%~dp0"

REM ===== 自动查找 Python =====
set PYTHON=
for %%p in (
    "python"
    "python3"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Python313\python.exe"
) do (
    where %%~p >nul 2>&1 && set "PYTHON=%%~p" && goto :found_python
)
echo ERROR: Python not found! Please install Python 3.11+
pause
exit /b 1
:found_python
echo Found Python: %PYTHON%

REM ===== 自动查找 FreeLLMAPI =====
set FREEAPI_DIR=
if exist "%~dp0..\freellmapi\package.json" (
    set "FREEAPI_DIR=%~dp0..\freellmapi"
) else if exist "%~dp0freellmapi\package.json" (
    set "FREEAPI_DIR=%~dp0freellmapi"
)

if defined FREEAPI_DIR (
    echo ========================================
    echo   Starting FreeLLMAPI...
    echo ========================================
    start "FreeLLMAPI" cmd /k "cd /d "%FREEAPI_DIR%" && npm run dev --workspace=server"
    echo Waiting for FreeLLMAPI (5s^)...
    timeout /t 5 /nobreak >nul
) else (
    echo ========================================
    echo   FreeLLMAPI not found, skipping.
    echo   LLM will use SiliconFlow directly.
    echo ========================================
)

echo ========================================
echo   Starting AI Illustration Generator...
echo ========================================
"%PYTHON%" main.py
pause
