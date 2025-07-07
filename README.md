# 🌱 Plant Disease Detection

Ein umfassendes System zur Erkennung von Pflanzenkrankheiten mittels Deep Learning und Vision-Language Models (VLM).

## 📋 Projektübersicht

Dieses Projekt implementiert ein modernes System zur automatischen Erkennung von Pflanzenkrankheiten mit folgenden Hauptfunktionen:

- **Klassifikation**: Deep Learning-Modelle zur Krankheitserkennung
- **Ähnlichkeitssuche**: VLM-basierte Embeddings für ähnliche Bilder
- **Vector Database**: Qdrant-Integration für skalierbare Suche
- **Web-Interface**: Streamlit-App für einfache Nutzung

## 🏗️ Projektstruktur

```
plant_disease_detection/
├── data/
│   ├── raw/                    # Rohdaten (Trainings- und Validierungsbilder)
│   │   ├── train/
│   │   │   ├── healthy/
│   │   │   ├── disease_A/
│   │   │   └── disease_B/
│   │   └── val/
│   │       ├── healthy/
│   │       ├── disease_A/
│   │       └── disease_B/
│   └── processed/              # Verarbeitete Daten
│       ├── images/
│       └── metadata.csv
├── notebooks/                  # Jupyter Notebooks für Exploration und Training
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_vlm_embedding.ipynb
│   └── 04_qdrant_integration.ipynb
├── models/                     # Trainierte Modelle
│   ├── classification_model/
│   │   ├── best_model.h5
│   │   └── checkpoint/
│   └── vlm_embedder/
│       └── model_config.json
├── src/                        # Quellcode
│   ├── data_loader.py         # Datenlade- und Vorverarbeitungsklassen
│   ├── model.py               # Modellarchitekturen
│   ├── train.py               # Trainingsskript
│   ├── vlm_utils.py           # VLM-Embedding-Funktionen
│   ├── qdrant_handler.py      # Qdrant-Integration
│   └── app.py                 # Streamlit-Webanwendung
├── config/                     # Konfigurationsdateien
│   ├── dataset_config.yaml
│   ├── model_config.yaml
│   └── qdrant_config.yaml
├── tests/                      # Unit Tests
│   ├── test_data_loader.py
│   └── test_model.py
├── scripts/                    # Shell-Skripte
│   ├── preprocess_data.sh
│   └── run_training.sh
├── requirements.txt            # Python-Abhängigkeiten
├── README.md                   # Projektdokumentation
└── .gitignore                  # Git-Ignore-Datei
```

## 🚀 Schnellstart

### 1. Installation

```bash
# Repository klonen
git clone <repository-url>
cd plant_disease_detection

# Virtual Environment erstellen und aktivieren
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. Daten vorbereiten

```bash
# Füge deine Bilder in die entsprechenden Ordner ein:
# data/raw/train/{healthy,disease_A,disease_B}/
# data/raw/val/{healthy,disease_A,disease_B}/

# Datenvorverarbeitung ausführen
chmod +x scripts/preprocess_data.sh
./scripts/preprocess_data.sh
```

### 3. Modell trainieren

```bash
# Standard-Training mit ResNet50
chmod +x scripts/run_training.sh
./scripts/run_training.sh

# Mit benutzerdefinierten Parametern
./scripts/run_training.sh --architecture EfficientNetB0 --fine-tune --epochs 100
```

### 4. Web-App starten

```bash
# Streamlit-App starten
streamlit run src/app.py
```

## 📊 Verwendung der Notebooks

### 1. Datenexploration
```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

### 2. Modelltraining
```bash
jupyter lab notebooks/02_model_training.ipynb
```

### 3. VLM-Embeddings erstellen
```bash
jupyter lab notebooks/03_vlm_embedding.ipynb
```

### 4. Qdrant-Integration
```bash
# Qdrant-Server starten (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Notebook ausführen
jupyter lab notebooks/04_qdrant_integration.ipynb
```

## ⚙️ Konfiguration

### Dataset-Konfiguration (`config/dataset_config.yaml`)
```yaml
data_path: "data/raw"
image_processing:
  target_size: [224, 224]
  normalization: "imagenet"
dataloader:
  batch_size: 32
  validation_split: 0.2
classes:
  - "healthy"
  - "disease_A" 
  - "disease_B"
```

### Modell-Konfiguration (`config/model_config.yaml`)
```yaml
model_architecture: "ResNet50"
input_shape: [224, 224, 3]
num_classes: 3
training:
  learning_rate: 0.001
  epochs: 50
  batch_size: 32
```

### Qdrant-Konfiguration (`config/qdrant_config.yaml`)
```yaml
connection:
  host: "localhost"
  port: 6333
collection:
  name: "plant_disease_embeddings"
  vector_size: 512
```

## 🧪 Tests ausführen

```bash
# Alle Tests
python -m pytest tests/ -v

# Spezifische Tests
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_model.py -v
```

## 📈 Modellarchitekturen

Das System unterstützt verschiedene vortrainierte Modelle:

- **ResNet50**: Bewährte Architektur, gute Balance zwischen Genauigkeit und Geschwindigkeit
- **EfficientNetB0**: Effiziente Architektur mit geringem Speicherbedarf
- **VGG16**: Klassische Architektur, robust aber langsamer

### Transfer Learning
Alle Modelle verwenden Transfer Learning mit ImageNet-vortrainierten Gewichten:
1. Frozen Base Model Training (Initial)
2. Fine-tuning (Optional)

## 🔍 VLM-Integration

Das System nutzt CLIP (Contrastive Language-Image Pre-training) für:
- Multimodale Embeddings (Bild + Text)
- Ähnlichkeitssuche
- Zero-shot Klassifikation

### Unterstützte VLM-Modelle
- `openai/clip-vit-base-patch32` (Standard)
- Weitere CLIP-Varianten über Konfiguration

## 🗄️ Vector Database (Qdrant)

Qdrant wird für skalierbare Ähnlichkeitssuche verwendet:

### Setup
```bash
# Qdrant mit Docker starten
docker run -p 6333:6333 qdrant/qdrant

# Oder lokale Installation
pip install qdrant-client
```

### Funktionen
- Embedding-Speicherung
- Cosinus-Ähnlichkeitssuche
- Metadaten-Filterung
- Batch-Operations

## 🌐 Web-Interface

Die Streamlit-App bietet:
- Drag & Drop Bild-Upload
- Echtzeit-Klassifikation
- Ähnlichkeitssuche-Visualisierung
- Konfidenz-Scores
- Model-Performance-Metriken

### Features
- 📤 **Bild-Upload**: Unterstützt JPG, PNG
- 🔍 **Klassifikation**: Echtzeit-Krankheitserkennung
- 🎯 **Ähnlichkeitssuche**: Finde ähnliche Fälle
- 📊 **Visualisierung**: Interaktive Ergebnisdarstellung

## 🛠️ Entwicklung

### Code-Struktur
```python
# Datenloader
from src.data_loader import PlantDiseaseDataLoader

# Modell
from src.model import PlantDiseaseClassifier

# VLM-Embeddings
from src.vlm_utils import PlantDiseaseEmbedder

# Qdrant-Integration
from src.qdrant_handler import QdrantHandler
```

### Neue Krankheitsklassen hinzufügen
1. Datenordner erstellen: `data/raw/train/neue_krankheit/`
2. Konfiguration anpassen: `config/dataset_config.yaml`
3. Modell neu trainieren

### Custom Modellarchitekturen
Erweitere `src/model.py` für neue Architekturen:
```python
def create_custom_model(input_shape, num_classes):
    # Deine Modellarchitektur
    pass
```

## 📊 Performance & Monitoring

### Metriken
- **Accuracy**: Klassifikationsgenauigkeit
- **Precision/Recall**: Pro Klasse
- **F1-Score**: Harmonisches Mittel
- **Confusion Matrix**: Detaillierte Fehleranalyse

### Logging
- Training-Logs: `models/classification_model/logs/`
- Modell-Checkpoints: `models/classification_model/checkpoint/`
- Ergebnisse: `models/classification_model/training_results.yaml`

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle einen Feature-Branch: `git checkout -b feature/neue-funktion`
3. Commit deine Änderungen: `git commit -am 'Neue Funktion hinzugefügt'`
4. Push zum Branch: `git push origin feature/neue-funktion`
5. Erstelle einen Pull Request

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe `LICENSE` Datei für Details.

## 🆘 Support

Bei Fragen oder Problemen:
1. Überprüfe die [Issues](../../issues)
2. Erstelle ein neues Issue mit detaillierter Beschreibung
3. Verwende die bereitgestellten Log-Dateien für Debugging

## 🔗 Nützliche Links

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [CLIP Model Documentation](https://huggingface.co/openai/clip-vit-base-patch32)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Erstellt für die moderne Pflanzenkrankheitserkennung mit KI** 🌱🤖