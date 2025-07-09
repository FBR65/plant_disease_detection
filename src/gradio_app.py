"""
Erweiterte Gradio-App für Plant Disease Detection
Kombiniert CNN-Klassifikation, Qdrant-Ähnlichkeitssuche und LLM-Bewertung
"""

import gradio as gr
import numpy as np
import torch
import json
import asyncio
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os

# Eigene Module
try:
    from .train import PlantDiseaseClassifier
    from .qdrant_handler import (
        PlantDiseaseQdrantHandler,
        initialize_qdrant_handler,
        SimilarCase,
    )
    from .evaluation_agent import PlantDiseaseEvaluationAgent
    from .advanced_augmentation import AdvancedPlantDataset
except ImportError:
    # Fallback für direkten Aufruf
    from train import PlantDiseaseClassifier
    from qdrant_handler import (
        PlantDiseaseQdrantHandler,
        initialize_qdrant_handler,
        SimilarCase,
    )
    from evaluation_agent import PlantDiseaseEvaluationAgent
    from advanced_augmentation import AdvancedPlantDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

logger = logging.getLogger(__name__)


class IntegratedPlantDiseaseApp:
    """
    Integrierte Anwendung für Pflanzenkrankheitserkennung

    Features:
    - CNN-Klassifikation mit PyTorch
    - VLM-basierte Ähnlichkeitssuche via Qdrant
    - LLM-gestützte finale Bewertung mit PydanticAI
    - Moderne Gradio-Benutzeroberfläche
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = None
        self.class_names = []
        self.qdrant_handler = None
        self.evaluation_agent = None
        self.transform = None

        logger.info(f"🚀 Initialisiere Plant Disease App auf {self.device}")

        # System initialisieren
        self._load_cnn_model()
        self._initialize_qdrant()
        self._initialize_evaluation_agent()
        self._setup_transforms()

        # Startup-Status loggen
        self._log_startup_status()

    def _load_cnn_model(self):
        """Lädt das trainierte CNN-Modell"""
        try:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "models" / "classification_model" / "model.pth"
            classes_path = (
                project_root / "models" / "classification_model" / "classes.json"
            )

            if model_path.exists() and classes_path.exists():
                # Klassen laden
                with open(classes_path, "r") as f:
                    self.class_names = json.load(f)

                # Modell initialisieren und laden
                self.cnn_model = PlantDiseaseClassifier(
                    num_classes=len(self.class_names), architecture="resnet50"
                )

                # State dict laden
                state_dict = torch.load(model_path, map_location=self.device)
                self.cnn_model.load_state_dict(state_dict)
                self.cnn_model.to(self.device)
                self.cnn_model.eval()

                logger.info(f"CNN-Modell geladen: {len(self.class_names)} Klassen")
            else:
                logger.warning("Kein trainiertes Modell gefunden. Verwende Demo-Modus.")

        except Exception as e:
            logger.error(f"Fehler beim Laden des CNN-Modells: {e}")
            self.cnn_model = None

    def _initialize_qdrant(self):
        """Initialisiert Qdrant-Handler"""
        try:
            self.qdrant_handler = initialize_qdrant_handler()
            logger.info("Qdrant-Handler initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei Qdrant-Initialisierung: {e}")
            self.qdrant_handler = None

    def _initialize_evaluation_agent(self):
        """Initialisiert PydanticAI-Bewertungsagent"""
        try:
            self.evaluation_agent = PlantDiseaseEvaluationAgent()
            logger.info("Bewertungsagent initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei Agent-Initialisierung: {e}")
            self.evaluation_agent = None

    def _setup_transforms(self):
        """Setzt up Bild-Transformationen für CNN"""
        self.transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.CenterCrop(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def _log_startup_status(self):
        """Loggt den Status aller Systemkomponenten beim Start"""
        logger.info("📊 System-Komponenten Status:")
        logger.info(
            f"  🤖 CNN-Modell: {'✅ Geladen' if self.cnn_model else '❌ Nicht verfügbar'}"
        )
        logger.info(
            f"  🗄️ Qdrant: {'✅ Verbunden' if self.qdrant_handler else '❌ Nicht verbunden'}"
        )
        logger.info(
            f"  🧠 LLM-Agent: {'✅ Initialisiert' if self.evaluation_agent else '❌ Nicht konfiguriert'}"
        )
        logger.info(f"  ⚙️ Device: {self.device}")

        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )

    def predict_with_cnn(self, image_path: str) -> Dict[str, Any]:
        """CNN-Klassifikation durchführen"""
        try:
            if self.cnn_model is None:
                return self._demo_cnn_prediction()

            # Bild laden und vorverarbeiten
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

            # Transformationen anwenden
            transformed = self.transform(image=image)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)

            # Vorhersage
            with torch.no_grad():
                outputs = self.cnn_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Ergebnisse verarbeiten
            probs = probabilities.cpu().numpy()[0]
            predicted_idx = np.argmax(probs)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probs[predicted_idx])

            # Top-3 Vorhersagen
            top3_indices = np.argsort(probs)[-3:][::-1]
            top_predictions = [
                (self.class_names[idx], float(probs[idx])) for idx in top3_indices
            ]

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_predictions": top_predictions,
            }

        except Exception as e:
            logger.error(f"Fehler bei CNN-Vorhersage: {e}")
            return self._demo_cnn_prediction()

    def _demo_cnn_prediction(self) -> Dict[str, Any]:
        """Demo-Vorhersage wenn kein Modell verfügbar"""
        import random

        demo_classes = [
            "Healthy leaf",
            "Tomato Early blight leaf",
            "Apple rust leaf",
            "Corn Gray leaf spot",
            "Potato leaf early blight",
        ]

        predicted_class = random.choice(demo_classes)
        confidence = random.uniform(0.7, 0.95)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": [
                (predicted_class, confidence),
                (random.choice(demo_classes), random.uniform(0.1, 0.3)),
                (random.choice(demo_classes), random.uniform(0.05, 0.15)),
            ],
        }

    def search_similar_cases(self, image_path: str) -> List[SimilarCase]:
        """Ähnlichkeitssuche mit Qdrant"""
        try:
            if self.qdrant_handler is None:
                return self._demo_similar_cases()

            similar_cases = self.qdrant_handler.search_similar_images(
                query_image_path=image_path, limit=5, min_similarity=0.5
            )

            return similar_cases

        except Exception as e:
            logger.error(f"Fehler bei Ähnlichkeitssuche: {e}")
            return self._demo_similar_cases()

    def _demo_similar_cases(self) -> List[SimilarCase]:
        """Demo ähnliche Fälle"""
        import random

        demo_diseases = ["Tomato Early blight leaf", "Apple rust leaf", "Healthy leaf"]

        cases = []
        for i in range(3):
            case = SimilarCase(
                image_id=f"demo_{i}",
                similarity_score=random.uniform(0.75, 0.95),
                disease_label=random.choice(demo_diseases),
                metadata={
                    "plant": "demo_plant",
                    "upload_date": "2024-01-01",
                    "source": "demo",
                },
            )
            cases.append(case)

        return cases

    async def comprehensive_evaluation(self, image_path: str) -> Dict[str, Any]:
        """Vollständige Bewertung mit allen Systemen"""

        # 1. CNN-Klassifikation
        cnn_prediction = self.predict_with_cnn(image_path)

        # 2. Ähnlichkeitssuche
        similar_cases = self.search_similar_cases(image_path)

        # 3. LLM-Bewertung
        if self.evaluation_agent:
            try:
                comprehensive_result = (
                    await self.evaluation_agent.run_comprehensive_evaluation(
                        image_path=image_path,
                        cnn_prediction=cnn_prediction,
                        similar_cases=similar_cases,
                        additional_context="Benutzer-Upload über Gradio-Interface",
                    )
                )

                return comprehensive_result

            except Exception as e:
                logger.error(f"Fehler bei LLM-Bewertung: {e}")
                return self._create_fallback_evaluation(cnn_prediction, similar_cases)
        else:
            return self._create_fallback_evaluation(cnn_prediction, similar_cases)

    def _create_fallback_evaluation(
        self, cnn_prediction: Dict[str, Any], similar_cases: List[SimilarCase]
    ) -> Dict[str, Any]:
        """Fallback-Bewertung ohne LLM"""

        has_disease = "healthy" not in cnn_prediction["predicted_class"].lower()

        return {
            "llm_assessment": {
                "has_disease": has_disease,
                "disease_name": cnn_prediction["predicted_class"],
                "confidence_score": cnn_prediction["confidence"],
                "reasoning": "Bewertung basiert nur auf CNN-Klassifikation (LLM nicht verfügbar)",
                "symptoms_detected": ["Automatisch erkannt"] if has_disease else [],
                "recommendation": "Für detaillierte Analyse LLM-System konfigurieren",
                "severity": "Unbekannt",
            },
            "cnn_prediction": cnn_prediction,
            "similar_cases": [case.model_dump() for case in similar_cases],
            "consistency_score": 0.5,
            "final_recommendation": "Basis-Bewertung ohne LLM-Integration",
            "confidence_level": cnn_prediction["confidence"]
            * 0.7,  # Reduziert ohne LLM
        }


# Globale App-Instanz
app_instance = None


def initialize_app():
    """Initialisiert die App-Instanz"""
    global app_instance
    if app_instance is None:
        app_instance = IntegratedPlantDiseaseApp()
    return app_instance


async def process_image_comprehensive(image) -> Tuple[str, str, str, str]:
    """
    Hauptfunktion für umfassende Bildverarbeitung

    Returns:
        (cnn_result, similar_cases_result, llm_assessment, final_summary)
    """
    if image is None:
        return ("Kein Bild hochgeladen", "", "", "Bitte laden Sie ein Bild hoch.")

    try:
        # App initialisieren
        app = initialize_app()

        # Temporäre Datei erstellen
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Umfassende Bewertung durchführen
            result = await app.comprehensive_evaluation(temp_path)

            # Ergebnisse formatieren
            cnn_result = f"""🤖 **CNN-KLASSIFIKATION**
Vorhersage: {result["cnn_prediction"]["predicted_class"]}
Konfidenz: {result["cnn_prediction"]["confidence"]:.3f}

Top-3 Vorhersagen:
{chr(10).join([f"  {i + 1}. {cls}: {conf:.3f}" for i, (cls, conf) in enumerate(result["cnn_prediction"]["top_predictions"])])}"""

            similar_cases_result = f"""🔍 **ÄHNLICHE FÄLLE** (Qdrant)
Gefunden: {len(result["similar_cases"])} ähnliche Bilder

{
                chr(10).join(
                    [
                        f"  📸 Fall {i + 1}: {case['disease_label']} (Ähnlichkeit: {case['similarity_score']:.3f})"
                        for i, case in enumerate(result["similar_cases"][:3])
                    ]
                )
            }"""

            llm_assessment = f"""🧠 **LLM-BEWERTUNG**
Krankheit erkannt: {"✅ Ja" if result["llm_assessment"]["has_disease"] else "❌ Nein"}
Diagnose: {result["llm_assessment"]["disease_name"]}
Konfidenz: {result["llm_assessment"]["confidence_score"]:.3f}
Schweregrad: {result["llm_assessment"]["severity"]}

Begründung: {result["llm_assessment"]["reasoning"]}

Empfehlung: {result["llm_assessment"]["recommendation"]}"""

            final_summary = f"""🎯 **FINALE BEWERTUNG**

Konsistenz zwischen Systemen: {result["consistency_score"]:.3f}
Gesamt-Konfidenz: {result["confidence_level"]:.3f}

🔍 **Endgültige Empfehlung:**
{result["final_recommendation"]}

💡 **Erkannte Symptome:**
{", ".join(result["llm_assessment"]["symptoms_detected"]) if result["llm_assessment"]["symptoms_detected"] else "Keine spezifischen Symptome erkannt"}"""

            return (cnn_result, similar_cases_result, llm_assessment, final_summary)

        finally:
            # Temporäre Datei löschen
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        error_msg = f"Fehler bei der Verarbeitung: {str(e)}"
        logger.error(error_msg)
        return (error_msg, "", "", "Fehler bei der Bildverarbeitung")


def process_image_wrapper(image):
    """Synchroner Wrapper für async-Funktion"""
    return asyncio.run(process_image_comprehensive(image))


def get_system_status():
    """Gibt den Status aller Systemkomponenten zurück"""
    try:
        app = initialize_app()

        status = f"""🔍 **SYSTEMSTATUS**

🤖 CNN-Modell: {"✅ Geladen" if app.cnn_model else "❌ Nicht verfügbar (Demo-Modus)"}
🗄️ Qdrant-DB: {"✅ Verbunden" if app.qdrant_handler else "❌ Nicht verbunden"}
🧠 LLM-Agent: {"✅ Initialisiert" if app.evaluation_agent else "❌ Nicht konfiguriert"}
⚙️ GPU: {"✅ Verfügbar" if torch.cuda.is_available() else "❌ CPU-Modus"}

📊 **GPU-INFO:**
{f"Device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Keine GPU"}
{f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else ""}

🛠️ **KONFIGURATION:**
Modell-Architektur: ResNet50
VLM-Embeddings: CLIP ViT-B/32
LLM-Endpunkt: {os.getenv("BASE_URL", "Nicht konfiguriert")}
"""

        return status

    except Exception as e:
        return f"Fehler beim Abrufen des Status: {e}"


def main():
    """Hauptfunktion für die erweiterte Gradio-App"""

    # Interface erstellen
    with gr.Blocks(
        theme=gr.themes.Soft(), title="🌱 Plant Disease Detection - AI-Powered Analysis"
    ) as interface:
        gr.Markdown("# 🌱 Plant Disease Detection")
        gr.Markdown(
            "**KI-gestützte Pflanzenkrankheitserkennung mit CNN, VLM-Ähnlichkeitssuche und LLM-Bewertung**"
        )

        with gr.Tab("🔍 Krankheitserkennung"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil", label="Pflanzenbild hochladen", height=400
                    )
                    analyze_btn = gr.Button(
                        "🔬 Umfassende Analyse starten", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    cnn_output = gr.Textbox(
                        label="🤖 CNN-Klassifikation", lines=8, max_lines=12
                    )

                    similar_cases_output = gr.Textbox(
                        label="🔍 Ähnliche Fälle (Qdrant)", lines=6, max_lines=10
                    )

            with gr.Row():
                llm_output = gr.Textbox(
                    label="🧠 LLM-Bewertung", lines=10, max_lines=15
                )

                final_output = gr.Textbox(
                    label="🎯 Finale Bewertung", lines=10, max_lines=15
                )

        with gr.Tab("⚙️ System-Status"):
            status_btn = gr.Button("🔄 Status aktualisieren")
            status_output = gr.Textbox(
                label="Systemstatus", lines=15, max_lines=20, value=get_system_status()
            )

        with gr.Tab("📚 Informationen"):
            gr.Markdown("""
            ## 🎯 Über dieses System
            
            Diese Anwendung kombiniert drei verschiedene KI-Ansätze für die Pflanzenkrankheitserkennung:
            
            ### 🤖 **CNN-Klassifikation (PyTorch)**
            - ResNet50-Architektur mit Transfer Learning
            - Trainiert auf PlantDoc-Dataset (28 Klassen)
            - GPU-optimiert für schnelle Inferenz
            
            ### 🔍 **VLM-Ähnlichkeitssuche (Qdrant + CLIP)**
            - CLIP-basierte Bild-Embeddings 
            - Vector-Database für semantische Suche
            - Findet visuell ähnliche Krankheitsfälle
            
            ### 🧠 **LLM-Bewertung (PydanticAI)**
            - Intelligente Kombination aller Datenquellen
            - Strukturierte Bewertung mit Begründung
            - Handlungsempfehlungen und Konfidenz-Bewertung
            
            ### 🚀 **Vorteile der Multi-System-Architektur**
            - **Robustheit**: Fehler in einem System werden durch andere kompensiert
            - **Transparenz**: Jedes System kann einzeln evaluiert werden  
            - **Flexibilität**: Einfache Anpassung an neue Anwendungsfälle
            - **Skalierbarkeit**: Komponenten können unabhängig optimiert werden
            """)

        # Event-Handler
        analyze_btn.click(
            fn=process_image_wrapper,
            inputs=[image_input],
            outputs=[cnn_output, similar_cases_output, llm_output, final_output],
        )

        status_btn.click(fn=get_system_status, outputs=[status_output])

    # App starten
    interface.launch(
        server_name="0.0.0.0", server_port=7860, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
