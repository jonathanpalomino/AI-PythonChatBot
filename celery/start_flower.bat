@echo off
title Flower Monitor - Proyecto PythonChatbot
setlocal ENABLEDELAYEDEXPANSION

echo =============================================================================
echo  celery/start_flower.bat
echo  Start Flower Monitor on Windows (con deteccion de procesos)
echo =============================================================================
echo.

REM --- Directorios base
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"

REM --- Venv
set "VENV_ACTIVATE=%PROJECT_ROOT%\.venv\Scripts\activate"

REM --- Módulo de app Celery
set "CELERY_APP=src.config.celery:app"

REM --- Puerto de Flower
set "FLOWER_PORT=5555"

echo [INFO] PROJECT_ROOT = %PROJECT_ROOT%
echo [INFO] VENV_ACTIVATE = %VENV_ACTIVATE%
echo [INFO] CELERY_APP    = %CELERY_APP%
echo [INFO] FLOWER_PORT   = %FLOWER_PORT%
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

REM --- DETECCION DE PROCESOS FLOWER
echo [INFO] Verificando si Flower ya esta corriendo...

REM Verificar si el puerto 5555 está en uso (usando netstat)
netstat -ano | findstr ":%FLOWER_PORT%" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Se detecto que el puerto %FLOWER_PORT% ya esta en uso.
    echo [INFO] Flower probablemente ya esta corriendo.
    echo [INFO] Para ver el proceso que usa el puerto, ejecuta: netstat -ano ^| findstr ":%FLOWER_PORT%"
    echo [INFO] Si deseas detenerlo, ejecuta: celery\stop.bat
    echo.
    echo [INFO] No se iniciara un nuevo Flower.
    goto :END
)

REM Verificación adicional: buscar procesos python con "flower" en la línea de comandos
REM Usar PowerShell en lugar de wmic (deprecated)
powershell -Command "$processes = Get-Process python*,pythonw* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*flower*' }; if ($processes) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Se detecto un proceso de Flower corriendo.
    echo [INFO] Para ver los procesos, ejecuta: powershell -Command \"Get-Process python*,pythonw* | Where-Object { $_.CommandLine -like '*flower*' } | Select-Object Id,CommandLine\"
    echo [INFO] Si deseas detenerlo, ejecuta: celery\stop.bat
    echo.
    echo [INFO] No se iniciara un nuevo Flower.
    goto :END
)

echo [INFO] No se detectaron procesos Flower activos.
echo [INFO] Iniciando nuevo Flower Monitor...
echo.

REM --- Iniciar Flower
REM start "Flower Monitor" /D "%PROJECT_ROOT%" cmd /k ^
REM "set PYTHONPATH=%PROJECT_ROOT% && call "%VENV_ACTIVATE%" && celery --workdir "%PROJECT_ROOT%" -A %CELERY_APP% flower --address=127.0.0.1 --port=%FLOWER_PORT%"

cd /d "%PROJECT_ROOT%"
celery --workdir "%PROJECT_ROOT%" -A %CELERY_APP% flower --address=127.0.0.1 --port=%FLOWER_PORT%


echo.
echo [INFO] Flower Monitor iniciado en ventana separada.
echo [INFO] Accede a Flower en: http://127.0.0.1:%FLOWER_PORT%
echo [INFO] Presiona cualquier tecla para cerrar esta ventana (Flower seguira corriendo)...
pause >nul

:END
endlocal
