#!/bin/bash

# Datenvorverarbeitungsscript für Pflanzenkrankheitserkennung
echo "🌱 Starte Datenvorverarbeitung..."

# Konfiguration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"
PYTHON_CMD="python"

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

# Erstelle Verzeichnisse
echo "📁 Erstelle Verzeichnisse..."
mkdir -p "$PROCESSED_DIR/images"
mkdir -p "$RAW_DIR/train/healthy"
mkdir -p "$RAW_DIR/train/disease_A"
mkdir -p "$RAW_DIR/train/disease_B"
mkdir -p "$RAW_DIR/val/healthy"
mkdir -p "$RAW_DIR/val/disease_A"
mkdir -p "$RAW_DIR/val/disease_B"

# Überprüfe Rohdaten
echo "🔍 Überprüfe Rohdaten..."
for split in train val; do
    for category in healthy disease_A disease_B; do
        image_count=$(find "$RAW_DIR/$split/$category" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
        echo "  $split/$category: $image_count Bilder"
    done
done

# Erstelle Metadaten-CSV
echo "📊 Erstelle Metadaten..."
$PYTHON_CMD -c "
import sys
sys.path.append('src')
from data_loader import PlantDiseaseDataLoader

loader = PlantDiseaseDataLoader()
stats = loader.get_dataset_statistics()
print('Datensatz-Statistiken:')
for split, split_stats in stats.items():
    print(f'  {split.upper()}:')
    total = 0
    for class_name, count in split_stats.items():
        print(f'    {class_name}: {count} Bilder')
        total += count
    print(f'    Gesamt: {total} Bilder')

# Metadaten-CSV erstellen
try:
    df = loader.create_metadata_csv('$PROCESSED_DIR/metadata.csv')
    print(f'✅ Metadaten-CSV erstellt mit {len(df)} Einträgen')
except Exception as e:
    print(f'❌ Fehler beim Erstellen der Metadaten: {e}')
"

# Bildvorverarbeitung (optional)
echo "🖼️ Bildvorverarbeitung..."
echo "  (Implementierung je nach Bedarf - z.B. Resize, Normalisierung, etc.)"

# Datenvalidierung
echo "✅ Validiere Daten..."
$PYTHON_CMD -c "
import sys
sys.path.append('src')
from data_loader import PlantDiseaseDataLoader
from pathlib import Path

loader = PlantDiseaseDataLoader()
stats = loader.get_dataset_statistics()

# Überprüfe ob Daten vorhanden sind
total_images = 0
for split_stats in stats.values():
    for count in split_stats.values():
        total_images += count

if total_images == 0:
    print('❌ Keine Bilder gefunden! Bitte Rohdaten hinzufügen.')
    exit(1)
else:
    print(f'✅ Validierung erfolgreich: {total_images} Bilder gefunden')

# Überprüfe Klassenverteilung
for split, split_stats in stats.items():
    class_counts = list(split_stats.values())
    if len(set(class_counts)) > 1:
        print(f'⚠️  Unausgewogene Klassenverteilung in {split}: {split_stats}')
    else:
        print(f'✅ Ausgewogene Klassenverteilung in {split}')
"

echo "🎉 Datenvorverarbeitung abgeschlossen!"
echo ""
echo "📋 Nächste Schritte:"
echo "  1. Überprüfe die generierten Metadaten in $PROCESSED_DIR/metadata.csv"
echo "  2. Führe das Training aus: ./scripts/run_training.sh"
echo "  3. Optional: Erstelle VLM-Embeddings mit Notebook 03_vlm_embedding.ipynb"
