@echo off
setlocal ENABLEDELAYEDEXPANSION

echo =============================================================================
echo  celery/stop.bat
echo  Stop Celery Worker and Flower on Windows
echo =============================================================================
echo.

REM --- Directorios base
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"

REM --- Puerto de Flower
set "FLOWER_PORT=5555"

echo [INFO] Verificando estado de servicios Celery...
echo.

REM --- Verificar Worker
set "WORKER_RUNNING=0"
powershell -Command "$processes = Get-Process python*,pythonw* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*celery*worker*' }; if ($processes) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    set "WORKER_RUNNING=1"
    echo [INFO] Celery Worker: CORRIENDO
) else (
    echo [INFO] Celery Worker: DETENIDO
)

REM --- Verificar Flower
set "FLOWER_RUNNING=0"
netstat -ano | findstr ":%FLOWER_PORT%" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    set "FLOWER_RUNNING=1"
    echo [INFO] Flower Monitor: CORRIENDO (puerto %FLOWER_PORT%)
) else (
    echo [INFO] Flower Monitor: DETENIDO
)

echo.

REM --- Si no hay servicios corriendo, salir
if %WORKER_RUNNING%==0 if %FLOWER_RUNNING%==0 (
    echo [INFO] No hay servicios Celery corriendo.
    goto :END
)

REM --- Detener Worker
if %WORKER_RUNNING%==1 (
    echo [INFO] Deteniendo Celery Worker...

    REM Intentar detener por WINDOWTITLE
    taskkill /F /T /FI "WINDOWTITLE eq Celery Worker*" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Ventana "Celery Worker" cerrada.
    )

    REM Intentar detener procesos celery worker usando PowerShell
    for /f "delims=" %%P in ('powershell -Command "Get-Process python*,pythonw* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*celery*worker*' } | Select-Object -ExpandProperty Id" 2^>nul') do (
        taskkill /F /PID %%P >nul 2>&1
    )

    echo [INFO] Celery Worker detenido.
)

echo.

REM --- Detener Flower
if %FLOWER_RUNNING%==1 (
    echo [INFO] Deteniendo Flower Monitor...

    REM Intentar detener por WINDOWTITLE
    taskkill /F /T /FI "WINDOWTITLE eq Flower Monitor*" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Ventana "Flower Monitor" cerrada.
    )

    REM Intentar detener procesos flower usando PowerShell
    for /f "delims=" %%P in ('powershell -Command "Get-Process python*,pythonw* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*flower*' } | Select-Object -ExpandProperty Id" 2^>nul') do (
        taskkill /F /PID %%P >nul 2>&1
    )

    REM Intentar detener por puerto (obtener PID del proceso que usa el puerto)
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%FLOWER_PORT%" ^| findstr "LISTENING"') do (
        taskkill /F /PID %%P >nul 2>&1
    )

    echo [INFO] Flower Monitor detenido.
)

echo.
echo =============================================================================
echo [INFO] Servicios Celery detenidos.
echo =============================================================================
echo.

REM --- Esperar un momento antes de cerrar
timeout /t 2 /nobreak >nul

:END
endlocal
