# Plant Disease Detection - PowerShell Launch Script

param(
    [switch]$StartQdrant,
    [switch]$SetupData,
    [switch]$ForceReload,
    [switch]$CheckOnly,
    [switch]$Interactive = $true
)

# Farbige Ausgabe
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }
function Write-Info { Write-ColorOutput Cyan $args }

Write-Info "=========================================="
Write-Info "🌱 Plant Disease Detection - PowerShell"
Write-Info "=========================================="
Write-Output ""

# Zum Projektverzeichnis wechseln
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# UV-Umgebung aktivieren falls vorhanden
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Success "✓ Aktiviere UV-Umgebung..."
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Warning "! Keine UV-Umgebung gefunden - verwende System-Python"
}

# Prüfe Python-Verfügbarkeit
try {
    $pythonVersion = python --version 2>&1
    Write-Success "✓ Python verfügbar: $pythonVersion"
} catch {
    Write-Error "❌ Python nicht gefunden!"
    exit 1
}

# Interaktiver Modus
if ($Interactive -and -not ($StartQdrant -or $SetupData -or $CheckOnly)) {
    Write-Output ""
    Write-Info "Verfügbare Optionen:"
    Write-Output "1. 🚀 Vollständiger Start (empfohlen)"
    Write-Output "2. 🔍 System-Check"
    Write-Output "3. 📊 Qdrant-Setup"
    Write-Output "4. 🌐 Nur Gradio-App"
    Write-Output "5. 🛠️ Dependencies installieren"
    Write-Output ""
    
    $choice = Read-Host "Wähle Option (1-5)"
    
    switch ($choice) {
        "1" { 
            Write-Info "🚀 Starte vollständiges System..."
            python launch.py --start-qdrant --setup-data
        }
        "2" { 
            Write-Info "🔍 Führe System-Check durch..."
            python launch.py --check-only
        }
        "3" { 
            Write-Info "📊 Setup Qdrant-Datenbank..."
            python -m scripts.setup_qdrant
        }
        "4" { 
            Write-Info "🌐 Starte nur Gradio-App..."
            python -m src.gradio_app
        }
        "5" {
            Write-Info "🛠️ Installiere Dependencies..."
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                uv sync --extra jupyter --extra dev
            } else {
                pip install -e ".[jupyter,dev]"
            }
        }
        default { 
            Write-Error "❌ Ungültige Auswahl"
            exit 1
        }
    }
} else {
    # Direkte Parameter-Verarbeitung
    $args = @()
    
    if ($StartQdrant) { $args += "--start-qdrant" }
    if ($SetupData) { $args += "--setup-data" }
    if ($ForceReload) { $args += "--force-data-reload" }
    if ($CheckOnly) { $args += "--check-only" }
    
    if ($args.Length -gt 0) {
        Write-Info "🚀 Starte mit Parametern: $($args -join ' ')"
        python launch.py @args
    } else {
        Write-Info "🌐 Starte Standard-Gradio-App..."
        python -m src.gradio_app
    }
}

Write-Output ""
Write-Info "Fertig! Drücke Enter zum Beenden..."
Read-Host
