@echo off
title Celery Worker - Proyecto PythonChatbot
setlocal ENABLEDELAYEDEXPANSION

echo =============================================================================
echo  celery/start_worker.bat
echo  Start Celery Worker on Windows (con deteccion de procesos)
echo =============================================================================
echo.

REM --- Directorios base
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"

REM --- Venv
set "VENV_ACTIVATE=%PROJECT_ROOT%\.venv\Scripts\activate"

REM --- Módulo de app Celery
set "CELERY_APP=src.config.celery:app"

echo [INFO] PROJECT_ROOT = %PROJECT_ROOT%
echo [INFO] VENV_ACTIVATE = %VENV_ACTIVATE%
echo [INFO] CELERY_APP    = %CELERY_APP%
echo.

REM --- Verificaciones básicas
if not exist "%VENV_ACTIVATE%" (
    echo [ERROR] No se encontro el activador del venv: "%VENV_ACTIVATE%"
    goto :END
)
if not exist "%PROJECT_ROOT%\src" (
    echo [ERROR] No existe la carpeta "src" en %PROJECT_ROOT%. Ajusta PROJECT_ROOT o la estructura.
    goto :END
)

REM --- Activar venv
call "%VENV_ACTIVATE%" || (echo [ERROR] Fallo la activacion del entorno virtual & goto :END)

REM --- Confirmar celery instalado
where celery >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se encontro el comando "celery" en el venv. Instala: pip install celery[redis] flower
    goto :END
)

REM --- Exportar PYTHONPATH
set "PYTHONPATH=%PROJECT_ROOT%"

REM --- DETECCION DE PROCESOS CELERY WORKER
echo [INFO] Verificando si Celery Worker ya esta corriendo...

REM Buscar procesos python con "celery worker" en la línea de comandos
REM Usar PowerShell en lugar de wmic (deprecated)
powershell -Command "$processes = Get-Process python*,pythonw* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*celery*worker*' }; if ($processes) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Se detectaron procesos Celery Worker ya corriendo.
    echo [INFO] Para ver los procesos activos, ejecuta: powershell -Command \"Get-Process python*,pythonw* | Where-Object { $_.CommandLine -like '*celery*worker*' } | Select-Object Id,CommandLine\"
    echo [INFO] Si deseas detenerlos, ejecuta: celery\stop.bat
    echo.
    echo [INFO] No se iniciara un nuevo worker.
    goto :END
)

echo [INFO] No se detectaron procesos Celery Worker activos.
echo [INFO] Iniciando nuevo Celery Worker...
echo.

REM --- Iniciar Worker
REM start "Celery Worker" /D "%PROJECT_ROOT%" cmd /k ^
REM "set PYTHONPATH=%PROJECT_ROOT% && call "%VENV_ACTIVATE%" && celery --workdir "%PROJECT_ROOT%" -A %CELERY_APP% worker -Q default -l info --pool=solo"

cd /d "%PROJECT_ROOT%"
celery --workdir "%PROJECT_ROOT%" -A %CELERY_APP% worker -Q default -l info --pool=solo

echo.
echo [INFO] Celery Worker iniciado en ventana separada.
echo [INFO] Presiona cualquier tecla para cerrar esta ventana (el worker seguira corriendo)...
pause >nul

:END
endlocal
