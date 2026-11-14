@echo off
title zar manah diatom tdi
cd /d "%~dp0"

echo ====================================================
echo zar manah diatom tdi
echo ====================================================

:: 1. Buscar Python en el sistema
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=py
    ) else (
        echo [!] No se encontro Python en el sistema.
        echo Por favor instala Python desde python.org
        pause
        exit /b 1
    )
)

:: 2. Crear entorno virtual autonomo si no existe
if not exist ".venv\Scripts\python.exe" (
    echo [*] Preparando entorno local autonomo (primera vez)...
    %PY_CMD% -m venv .venv
)

set VENV_PY=.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
    set VENV_PY=%PY_CMD%
)

:: 3. Instalar dependencias automaticamente si faltan
"%VENV_PY%" -c "import torch, PIL" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [*] Descargando e instalando componentes (PyTorch, Pillow)...
    "%VENV_PY%" -m pip install --quiet --upgrade pip
    "%VENV_PY%" -m pip install --quiet torch torchvision pillow
)

:: 4. Lanzar servidor local
"%VENV_PY%" app\server.py --lan
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Servidor finalizado con codigo %ERRORLEVEL%.
    pause
)
