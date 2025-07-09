@echo off
echo ========================================
echo Plant Disease Detection - Quick Start
echo ========================================
echo.

REM Wechsle zum Projektverzeichnis
cd /d "%~dp0"

REM Aktiviere UV-Umgebung falls verfügbar
if exist ".venv\Scripts\activate.bat" (
    echo ✓ Aktiviere UV-Umgebung...
    call .venv\Scripts\activate.bat
) else (
    echo ! Keine UV-Umgebung gefunden - verwende System-Python
)

echo.
echo Mögliche Optionen:
echo 1. Vollständiger Start (empfohlen)
echo 2. Nur System-Check
echo 3. Qdrant-Setup
echo 4. Nur Gradio-App
echo.

set /p choice="Wähle Option (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starte vollständiges System...
    python launch.py --start-qdrant --setup-data
) else if "%choice%"=="2" (
    echo.
    echo 🔍 Führe System-Check durch...
    python launch.py --check-only
) else if "%choice%"=="3" (
    echo.
    echo 📊 Setup Qdrant-Datenbank...
    python -m scripts.setup_qdrant
) else if "%choice%"=="4" (
    echo.
    echo 🌐 Starte nur Gradio-App...
    python -m src.gradio_app
) else (
    echo.
    echo ❌ Ungültige Auswahl
    pause
    exit /b 1
)

echo.
pause
