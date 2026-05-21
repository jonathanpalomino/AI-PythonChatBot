@echo off
setlocal

echo.
echo ══════════════════════════════════════════════════
echo   Iniciando Edge con remote debugging (Copilot365)
echo ══════════════════════════════════════════════════
echo.

set "EDGE_EXE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE_EXE%" set "EDGE_EXE=C:\Program Files\Microsoft\Edge\Application\msedge.exe"

if not exist "%EDGE_EXE%" (
    echo [ERROR] No se encontro msedge.exe. Edita EDGE_EXE en este script.
    pause & exit /b 1
)

curl -s http://localhost:9222/json/version >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Edge ya esta corriendo con remote debugging en puerto 9222.
    pause & exit /b 0
)

echo [INFO] Iniciando Edge con remote debugging...
start "" "%EDGE_EXE%" ^
    --remote-debugging-port=9222 ^
    --user-data-dir="%LOCALAPPDATA%\Microsoft\Edge\User Data Chatbot" ^
    --start-minimized ^
    --new-window ^
    "https://m365.cloud.microsoft/chat"

echo [OK] Edge iniciado. La API puede conectarse via CDP en puerto 9222.
echo.
pause
