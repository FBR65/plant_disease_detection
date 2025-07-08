# 🌱 Plant Disease Detection

Ein umfassendes System zur Erkennung von Pflanzenkrankheiten mittels PyTorch und Vision-Language Models (VLM).

## 🎯 Universelle Anwendbarkeit

**Dieses System ist als generische Vorlage für KI-basierte Bildanalyse und intelligente Suchsysteme konzipiert.** Die Architektur kombiniert trainierte Machine Learning-Modelle mit Vision-Language Models (VLM) und Vector-Datenbanken zu einem flexiblen Framework, das auf verschiedenste Anwendungsfälle übertragbar ist:

- **🔄 Modular aufgebaut**: Austauschbare Komponenten für Datenverarbeitung, Modellerstellung und Inferenz
- **🎨 Domain-agnostisch**: Einfache Anpassung an beliebige Bildklassifikations-Aufgaben (Medizin, Industrie, Einzelhandel, etc.)
- **🚀 Produktionsreif**: Robuste Datenverarbeitung, Fehlerbehandlung und skalierbare Deployment-Optionen
- **🔍 Intelligente Suche**: VLM-Embeddings ermöglichen semantische Ähnlichkeitssuche über Domänen hinweg
- **💾 Skalierbare Persistierung**: Vector-Database-Integration für große Datenmengen und Echtzeit-Abfragen

Die **Pflanzenkrankheitserkennung** dient als praktisches Beispiel und Referenzimplementierung. Die gleiche Architektur kann für Qualitätskontrolle in der Fertigung, medizinische Bilddiagnostik, Produktklassifikation im E-Commerce oder beliebige andere Computer Vision-Aufgaben eingesetzt werden.

## 📋 Projektübersicht

Dieses Projekt implementiert ein **produktionsreifes** System zur automatischen Erkennung von z. B. Pflanzenkrankheiten mit folgenden Hauptfunktionen:

- **🎯 Intelligente Klassifikation**: PyTorch-basierte Deep Learning-Modelle mit automatischem Klassenbalancing
- **🔄 Adaptive Datenaugmentation**: Synthetische Bilderzeugung für Minderheitsklassen (löst 89:1 Ungleichgewicht)
- **📊 Datenqualitäts-Analyse**: Automatische Erkennung von Problemen (Größenvariation, Klassenungleichgewicht)
- **⚖️ Smart Sampling**: WeightedRandomSampler für ausbalancierte Training-Batches
- **🌐 Moderne Web-UI**: Gradio-basierte Benutzeroberfläche für einfache Nutzung
- **🔍 Ähnlichkeitssuche**: VLM-basierte Embeddings für ähnliche Bilder (geplant)
- **💾 Vector Database**: Qdrant-Integration für skalierbare Suche (geplant)

## 🏗️ Projektstruktur

```
plant_disease_detection/
├── data/
│   └── PlantDoc/               # PlantDoc-Dataset
│       ├── train/              # Trainingsbilder
│       │   ├── healthy/
│       │   ├── disease_A/
│       │   └── disease_B/
│       └── test/               # Testbilder
│           ├── healthy/
│           ├── disease_A/
│           └── disease_B/
├── src/                        # Quellcode (Python-Module)
│   ├── data_exploration.py     # Erweiterte Datenanalyse mit Balancing-Empfehlungen
│   ├── advanced_augmentation.py # Intelligente Datenaugmentation & Klassenbalancing
│   ├── train.py               # PyTorch-Trainingsskript
│   ├── gradio_app.py          # Gradio-Webanwendung
│   ├── model.py               # Modellarchitekturen
│   ├── data_loader.py         # Datenlade- und Vorverarbeitungsklassen (Legacy)
│   ├── vlm_utils.py           # VLM-Embedding-Funktionen
│   └── qdrant_handler.py      # Qdrant-Integration
├── models/                     # Trainierte Modelle
│   ├── classification_model/
│   │   ├── model.pth
│   │   └── classes.json
│   └── vlm_embedder/
│       └── model_config.json
├── config/                     # Konfigurationsdateien
│   ├── dataset_config.yaml
│   ├── model_config.yaml
│   └── qdrant_config.yaml
├── tests/                      # Unit Tests
│   ├── test_data_loader.py
│   └── test_model.py
├── reports/                    # Berichte und Visualisierungen
│   ├── class_distribution.png
│   └── training_history.png
├── pyproject.toml              # Python-Abhängigkeiten (uv-kompatibel)
├── README.md                   # Projektdokumentation
└── .gitignore                  # Git-Ignore-Datei
```
PlantDoc Dataset: https://paperswithcode.com/dataset/plantdoc

## 🚀 Schnellstart

### 1. Installation

```bash
# Repository klonen
git clone <repository-url>
cd plant_disease_detection

# Virtual Environment mit uv erstellen
uv venv
source .venv/bin/activate  # Linux/Mac
# oder
.venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
uv sync
```

### 2. Daten vorbereiten

```bash
# PlantDoc-Dataset in folgende Struktur bringen:
# data/PlantDoc/train/{klasse1,klasse2,...}/
# data/PlantDoc/test/{klasse1,klasse2,...}/

# Datenexploration ausführen
uv run explore-data
```

### 3. Modell trainieren

```bash
# PyTorch-basiertes Training starten
uv run train-model
```

### 4. Web-App starten

```bash
# Gradio-App starten
uv run run-app
```

## 📊 Verfügbare Kommandos

### 🔍 Erweiterte Datenanalyse
```bash
# Vollständige Datenexploration mit Balancing-Empfehlungen
uv run explore-data

# Ergebnis: Automatische Erkennung von Klassenungleichgewicht (89:1 Ratio)
# + Empfehlungen für Augmentation und Sampling-Strategien
```

### 🔄 Intelligentes Training
```bash
# Training mit automatischem Klassenbalancing
uv run train-model

# Features:
# - Automatische Minoritätsklassen-Erkennung
# - Synthetische Bilderzeugung (3x Faktor)
# - WeightedRandomSampler für ausbalancierte Batches
# - Adaptive Augmentation (stärker für kleine Klassen)
```

### 🌐 Web-Anwendung
```bash
# Gradio-App mit verbesserter UI
uv run run-app

# Features:
# - Drag & Drop Bildupload
# - Real-time Klassifikation  
# - Konfidenz-Anzeige
# - Beispielbilder-Galerie
```

## 🛠️ Technologie-Stack

### Core ML Framework
- **PyTorch 2.0+** - Deep Learning Framework (Windows-kompatibel)
- **torchvision** - Computer Vision Utilities
- **transformers** - Hugging Face Transformers für VLM

### Computer Vision
- **Pillow** - Bildverarbeitung
- **OpenCV** - Computer Vision Operations
- **albumentations** - Datenaugmentation

### Web Interface
- **Gradio** - Moderne Web-UI für ML-Modelle
- **FastAPI** - REST API Backend (optional)

### Database & Storage
- **Qdrant** - Vector Database für Ähnlichkeitssuche
- **HDF5** - Effiziente Datenspeicherung

### Development Tools
- **uv** - Schneller Python Package Manager
- **ruff** - Code Formatting

## 🔧 Konfiguration

### Dataset-Konfiguration (`config/dataset_config.yaml`)
```yaml
data_path: "data/PlantDoc"
image_processing:
  target_size: [224, 224]
  normalization: "imagenet"
dataloader:
  batch_size: 32
  num_workers: 4
classes:
  # Automatisch aus Ordnerstruktur erkannt
```

### Model-Konfiguration (`config/model_config.yaml`)
```yaml
architecture: "resnet50"
pretrained: true
num_classes: auto  # Automatisch erkannt
training:
  epochs: 25
  learning_rate: 0.001
  batch_size: 32
  optimizer: "adam"
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

### Klassenbalancing-Konfiguration

```yaml
# Erweiterte Augmentation-Konfiguration
augmentation:
  # Automatisches Balancing
  enable_synthetic_generation: true
  synthetic_factor: 3              # 3x mehr Samples für Minorities
  minority_threshold: 0.5          # < 50% der max. Klassengröße
  
  # Weighted Sampling
  use_weighted_sampling: true
  
  # Augmentation-Stärke
  strong_augmentation_for_minorities: true
  
  # Transformationen
  image_size: 224
  resize_intermediate: 256
  geometric_transforms:
    rotation_limit: 30
    horizontal_flip: 0.5
    vertical_flip: 0.2
  
  photometric_transforms:
    brightness_limit: 0.2
    contrast_limit: 0.2
    hue_shift_limit: 10
    
  plant_specific:
    random_shadow: 0.3
    sun_flare: 0.1
    color_jitter: 0.3
```

## 📊 Datenqualitäts-Features

### Automatische Problemerkennung

Das System analysiert automatisch Ihren Datensatz und identifiziert:

```bash
uv run explore-data
```

**Erkannte Probleme (PlantDoc-Beispiel):**
- 🚨 **Kritisches Ungleichgewicht**: 89.5:1 Ratio (179 vs 2 Samples)
- 📐 **Extreme Größenvariation**: 143-5472px Breite (38x Unterschied)
- ⚠️ **Problematische Klasse**: "Tomato two spotted spider mites leaf" (nur 2 Bilder)

**Automatische Lösungsvorschläge:**
```
💡 EMPFEHLUNGEN:
🔄 DATENAUGMENTATION:
  - Synthetische Bilder für Minderheitsklassen generieren
  - Faktor 2-5x Augmentation für kleine Klassen
  
⚖️ SAMPLING STRATEGIEN:
  - WeightedRandomSampler für Klassenbalancing
  - Focal Loss für schwierige Klassen
  
🚨 KRITISCHES UNGLEICHGEWICHT:
  - Evaluation-Metriken: Precision/Recall statt Accuracy
```

### Intelligente Datenerweiterung

**Vor dem Balancing:**
```
Tomato two spotted spider mites leaf: 2 Samples    ⚠️
Corn leaf blight: 179 Samples                      📈
Ratio: 89.5:1 (kritisch)
```

**Nach automatischem Balancing:**
```
Tomato two spotted spider mites leaf: 82+ Samples  ✅
Corn leaf blight: 179 Samples                      📈  
Ratio: ~2:1 (ausbalanciert)
```

### Produktionsreife Features

- **📊 Datenqualitäts-Dashboard**: Automatische Analyse und Visualisierung
- **🔄 Adaptive Augmentation**: Stärke basierend auf Klassengröße
- **⚖️ Smart Sampling**: WeightedRandomSampler + Stratified Sampling  
- **🎯 Evaluation-Metriken**: Precision/Recall/F1 für unbalancierte Daten
- **💾 Caching**: Intelligentes Speichern verarbeiteter Bilder

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

## 🔬 Erweiterte Features

### Automatisches Klassenbalancing

Das System erkennt automatisch Klassenungleichgewichte und behebt diese:

```bash
# Erweiterte Datenexploration mit Balancing-Empfehlungen
uv run explore-data
```

**Automatische Problemerkennung:**
- ⚖️ **Klassenungleichgewicht-Analyse** (Ratio-Berechnung)
- 📐 **Bildgrößen-Variation** (119-6000px Breite erkannt)
- 🎯 **Minderheitsklassen-Identifikation** (< 50% der Median-Klassengröße)

**Intelligente Lösungen:**
- 🔄 **Synthetische Bilderzeugung** für Minderheitsklassen
- ⚖️ **WeightedRandomSampler** für ausbalancierte Batches  
- 🎨 **Adaptive Augmentation** (stärker für kleine Klassen)
- 📊 **Automatische Größen-Vereinheitlichung** (256→224px)

### Advanced Data Augmentation

```python
# Verwendung der erweiterten Augmentation
from src.advanced_augmentation import create_balanced_dataloader

# Automatisch ausbalancierter DataLoader
train_loader, dataset = create_balanced_dataloader(
    data_dir="data/PlantDoc",
    split="train",
    augment_minority_classes=True,  # Synthetische Bilder für kleine Klassen
    synthetic_factor=3,             # 3x mehr Augmentation für Minorities
    use_weighted_sampling=True      # Balanced Sampling
)
```

**Augmentation-Pipeline:**
- 🌱 **Pflanzen-spezifisch**: RandomShadow, SunFlare, ColorJitter
- 🔄 **Geometrisch**: Rotation, Flip, ElasticTransform, GridDistortion
- 🎨 **Photometrisch**: Brightness, Contrast, Hue, Saturation
- 🔍 **Qualität**: GaussNoise, GaussianBlur, CLAHE
- 📏 **Normalisierung**: ImageNet-Standards (mean/std)

**Beispiel-Ergebnis für PlantDoc:**
```
Klasse 'Tomato two spotted spider mites leaf': 2 → 80+ Samples
Automatisch: 89.5:1 Ratio → 4:1 Ratio (ausbalanciert)
```

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

Dieses Projekt steht unter der AGPLv3-Lizenz. Siehe `LICENSE` Datei für Details.

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