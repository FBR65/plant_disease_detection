#!/bin/bash

# Trainingscript für das Pflanzenkrankheits-Klassifikationsmodell
echo "🚀 Starte Training des Pflanzenkrankheits-Klassifikationsmodells..."

# Konfiguration
PYTHON_CMD="python"
MODEL_CONFIG="config/model_config.yaml"
DATA_CONFIG="config/dataset_config.yaml"
LOGS_DIR="models/classification_model/logs"
RESULTS_DIR="models/classification_model"

# Überprüfe ob Python verfügbar ist
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "❌ Python nicht gefunden!"
        exit 1
    fi
fi

echo "🐍 Verwende Python: $PYTHON_CMD"

# Erstelle notwendige Verzeichnisse
echo "📁 Erstelle Verzeichnisse..."
mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR/checkpoint"

# Überprüfe Konfigurationsdateien
echo "🔍 Überprüfe Konfiguration..."
if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "❌ Modellkonfiguration nicht gefunden: $MODEL_CONFIG"
    exit 1
fi

if [[ ! -f "$DATA_CONFIG" ]]; then
    echo "❌ Datenkonfiguration nicht gefunden: $DATA_CONFIG"
    exit 1
fi

echo "✅ Konfigurationsdateien gefunden"

# Überprüfe Daten
echo "🔍 Überprüfe Trainingsdaten..."
$PYTHON_CMD -c "
import sys
sys.path.append('src')
from data_loader import PlantDiseaseDataLoader

loader = PlantDiseaseDataLoader('$DATA_CONFIG')
stats = loader.get_dataset_statistics()

total_train = sum(stats.get('train', {}).values())
total_val = sum(stats.get('val', {}).values())

if total_train == 0:
    print('❌ Keine Trainingsdaten gefunden!')
    exit(1)

if total_val == 0:
    print('⚠️  Keine Validierungsdaten gefunden!')

print(f'✅ Trainingsdaten: {total_train} Bilder')
print(f'✅ Validierungsdaten: {total_val} Bilder')
"

if [[ $? -ne 0 ]]; then
    echo "❌ Datenüberprüfung fehlgeschlagen!"
    exit 1
fi

# Parameter verarbeiten
ARCHITECTURE="ResNet50"
FINE_TUNE=false
EPOCHS=""

# Kommandozeilenargumente verarbeiten
while [[ $# -gt 0 ]]; do
    case $1 in
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --fine-tune)
            FINE_TUNE=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Verwendung: $0 [OPTIONEN]"
            echo ""
            echo "Optionen:"
            echo "  --architecture ARCH   Modellarchitektur (ResNet50, EfficientNetB0, VGG16)"
            echo "  --fine-tune          Fine-tuning nach initialem Training durchführen"
            echo "  --epochs N           Anzahl Epochen (überschreibt Config)"
            echo "  --help               Diese Hilfe anzeigen"
            exit 0
            ;;
        *)
            echo "❌ Unbekannte Option: $1"
            echo "Verwende --help für Hilfe"
            exit 1
            ;;
    esac
done

echo "📋 Training-Parameter:"
echo "  Architektur: $ARCHITECTURE"
echo "  Fine-tuning: $FINE_TUNE"
if [[ -n "$EPOCHS" ]]; then
    echo "  Epochen: $EPOCHS"
fi

# GPU-Verfügbarkeit prüfen
echo "🔍 Überprüfe GPU-Verfügbarkeit..."
$PYTHON_CMD -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ {len(gpus)} GPU(s) verfügbar: {[gpu.name for gpu in gpus]}')
    # GPU-Memory-Growth aktivieren
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('⚠️  Keine GPU verfügbar - verwende CPU')
"

# Training starten
echo "🚀 Starte Training..."
TRAIN_CMD="$PYTHON_CMD src/train.py --model-config $MODEL_CONFIG --data-config $DATA_CONFIG"

if [[ "$ARCHITECTURE" != "ResNet50" ]]; then
    TRAIN_CMD="$TRAIN_CMD --architecture $ARCHITECTURE"
fi

if [[ "$FINE_TUNE" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD --fine-tune"
fi

echo "🔧 Ausgeführter Befehl: $TRAIN_CMD"
echo ""

# Training mit Zeitstempel
START_TIME=$(date +%s)
echo "⏰ Training gestartet: $(date)"

# Training ausführen und Output loggen
$TRAIN_CMD 2>&1 | tee "$LOGS_DIR/training_$(date +%Y%m%d_%H%M%S).log"

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "⏰ Training beendet: $(date)"
echo "⏱️  Dauer: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "🎉 Training erfolgreich abgeschlossen!"
    
    # Ergebnisse anzeigen
    if [[ -f "$RESULTS_DIR/training_results.yaml" ]]; then
        echo ""
        echo "📊 Training-Ergebnisse:"
        $PYTHON_CMD -c "
import yaml
with open('$RESULTS_DIR/training_results.yaml', 'r') as f:
    results = yaml.safe_load(f)

print(f\"  Finale Validation Accuracy: {results.get('final_val_accuracy', 'N/A'):.4f}\")
print(f\"  Finale Validation Loss: {results.get('final_val_loss', 'N/A'):.4f}\")
print(f\"  Beste Validation Accuracy: {results.get('best_val_accuracy', 'N/A'):.4f}\")
print(f\"  Trainierte Epochen: {results.get('total_epochs', 'N/A')}\")
"
    fi
    
    echo ""
    echo "📋 Nächste Schritte:"
    echo "  1. Überprüfe die Ergebnisse in $RESULTS_DIR/"
    echo "  2. Teste das Modell mit der Streamlit-App: streamlit run src/app.py"
    echo "  3. Optional: Erstelle VLM-Embeddings mit Notebook 03_vlm_embedding.ipynb"
    echo "  4. Optional: Integriere mit Qdrant mit Notebook 04_qdrant_integration.ipynb"
    
else
    echo "❌ Training fehlgeschlagen (Exit Code: $TRAIN_EXIT_CODE)"
    echo "📋 Debugging-Tipps:"
    echo "  1. Überprüfe das Log: $LOGS_DIR/training_$(date +%Y%m%d)*.log"
    echo "  2. Überprüfe die Konfigurationsdateien"
    echo "  3. Überprüfe die Datenqualität"
    exit 1
fi
