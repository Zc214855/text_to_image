@echo off
cd /d "%~dp0"

echo ========================================
echo   Starting FreeLLMAPI...
echo ========================================
start "FreeLLMAPI" cmd /k "cd /d F:\MyTool\freellmapi && npm run dev --workspace=server"

echo Waiting for FreeLLMAPI (5s)...
timeout /t 5 /nobreak >nul

echo ========================================
echo   Starting AI Illustration Generator...
echo ========================================
"C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe" main.py
pause
