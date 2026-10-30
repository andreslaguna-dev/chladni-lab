@echo off
chcp 65001 >nul
title Laboratorio de Figuras de Chladni - Windows
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ============================================================
    echo Configurando entorno virtual de Python para Windows...
    echo ============================================================
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual. Verifica que Python este instalado.
        pause
        exit /b 1
    )
    echo Instalando dependencias ^(numpy, scipy, sounddevice, PyQt5, pyqtgraph^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Fallo la instalacion de dependencias.
        pause
        exit /b 1
    )
)

echo Iniciando Laboratorio de Figuras de Chladni...
".venv\Scripts\python.exe" chladni_lab.py
if errorlevel 1 (
    echo.
    echo [AVISO] El programa se cerro con un codigo de error.
    pause
)
