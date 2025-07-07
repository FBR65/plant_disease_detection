"""
Hauptanwendung für die Pflanzenkrankheitserkennung
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
import yaml
import json

# Eigene Module
from model import PlantDiseaseClassifier
from vlm_utils import PlantDiseaseEmbedder
from qdrant_handler import QdrantHandler
from data_loader import preprocess_image_for_model

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDiseaseApp:
    """Hauptanwendungsklasse für die Pflanzenkrankheitserkennung"""

    def __init__(self):
        """Initialisiert die Anwendung"""
        self.classifier = None
        self.embedder = None
        self.qdrant_handler = None
        self.embeddings_data = None

        # Modelle lazy loading
        self._models_loaded = False

    def load_models(self):
        """Lädt alle benötigten Modelle"""
        if self._models_loaded:
            return

        try:
            # Klassifikationsmodell laden
            self.classifier = PlantDiseaseClassifier()
            model_path = "models/classification_model/best_model.h5"

            if Path(model_path).exists():
                self.classifier.load_model(model_path)
                logger.info("Klassifikationsmodell geladen")
            else:
                logger.warning(f"Modell nicht gefunden: {model_path}")

            # VLM Embedder laden
            self.embedder = PlantDiseaseEmbedder()
            logger.info("VLM Embedder geladen")

            # Qdrant Handler laden
            try:
                self.qdrant_handler = QdrantHandler()
                logger.info("Qdrant Handler geladen")
            except Exception as e:
                logger.warning(f"Qdrant nicht verfügbar: {e}")

            # Embeddings laden
            embeddings_path = "data/processed/embeddings.json"
            if Path(embeddings_path).exists():
                self.embeddings_data = self.embedder.load_embeddings(embeddings_path)
                logger.info(f"Embeddings geladen: {len(self.embeddings_data)} Einträge")

            self._models_loaded = True

        except Exception as e:
            logger.error(f"Fehler beim Laden der Modelle: {e}")
            st.error(f"Fehler beim Laden der Modelle: {e}")

    def classify_image(self, image: Image.Image) -> dict:
        """
        Klassifiziert ein Bild

        Args:
            image: PIL Image

        Returns:
            Klassifikationsergebnisse
        """
        try:
            # Bild für Modell vorbereiten
            # Temporäre Datei erstellen
            temp_path = "temp_image.jpg"
            image.save(temp_path)

            processed_image = preprocess_image_for_model(temp_path)

            if processed_image is None:
                return {"error": "Fehler bei der Bildvorverarbeitung"}

            # Klassifikation durchführen
            if self.classifier and self.classifier.model:
                result = self.classifier.predict(processed_image)

                # Temporäre Datei löschen
                Path(temp_path).unlink(missing_ok=True)

                return result
            else:
                return {"error": "Klassifikationsmodell nicht verfügbar"}

        except Exception as e:
            logger.error(f"Fehler bei der Klassifikation: {e}")
            return {"error": str(e)}

    def find_similar_images(self, image: Image.Image, top_k: int = 5) -> list:
        """
        Findet ähnliche Bilder basierend auf Embeddings

        Args:
            image: PIL Image
            top_k: Anzahl der ähnlichsten Bilder

        Returns:
            Liste ähnlicher Bilder
        """
        try:
            if not self.embedder:
                return []

            # Temporäre Datei erstellen
            temp_path = "temp_image.jpg"
            image.save(temp_path)

            # Ähnliche Bilder finden
            similar_images = self.embedder.find_similar_images(
                temp_path, self.embeddings_data or [], top_k=top_k
            )

            # Temporäre Datei löschen
            Path(temp_path).unlink(missing_ok=True)

            return similar_images

        except Exception as e:
            logger.error(f"Fehler bei der Ähnlichkeitssuche: {e}")
            return []


def create_streamlit_app():
    """Erstellt die Streamlit-Anwendung"""
    st.set_page_config(
        page_title="Pflanzenkrankheitserkennung", page_icon="🌱", layout="wide"
    )

    st.title("🌱 Pflanzenkrankheitserkennung")
    st.write(
        "Lade ein Bild einer Pflanze hoch, um Krankheiten zu erkennen und ähnliche Bilder zu finden."
    )

    # App-Instanz erstellen
    if "app" not in st.session_state:
        st.session_state.app = PlantDiseaseApp()

    app = st.session_state.app

    # Sidebar für Konfiguration
    with st.sidebar:
        st.header("⚙️ Einstellungen")

        load_models = st.button("🔄 Modelle laden")
        if load_models:
            with st.spinner("Lade Modelle..."):
                app.load_models()
            st.success("Modelle geladen!")

        st.write("---")

        show_similar = st.checkbox("🔍 Ähnliche Bilder anzeigen", value=True)
        similar_count = st.slider("Anzahl ähnlicher Bilder", 1, 10, 5)

        st.write("---")
        st.write("### 📊 Modellstatus")

        if app._models_loaded:
            st.success("✅ Modelle geladen")
        else:
            st.warning("⚠️ Modelle nicht geladen")

    # Hauptinhalt
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Bild hochladen")

        uploaded_file = st.file_uploader(
            "Wähle ein Bild aus...",
            type=["jpg", "jpeg", "png"],
            help="Unterstützte Formate: JPG, JPEG, PNG",
        )

        if uploaded_file is not None:
            # Bild anzeigen
            image = Image.open(uploaded_file)
            st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

            # Analyse-Button
            if st.button("🔍 Bild analysieren", type="primary"):
                if not app._models_loaded:
                    st.error("Bitte lade zuerst die Modelle in der Sidebar.")
                else:
                    with st.spinner("Analysiere Bild..."):
                        # Klassifikation
                        classification = app.classify_image(image)

                        # Ähnliche Bilder
                        similar_images = []
                        if show_similar:
                            similar_images = app.find_similar_images(
                                image, similar_count
                            )

                    # Ergebnisse in Session State speichern
                    st.session_state.classification = classification
                    st.session_state.similar_images = similar_images

    with col2:
        st.header("📊 Analyseergebnisse")

        # Klassifikationsergebnisse anzeigen
        if hasattr(st.session_state, "classification"):
            classification = st.session_state.classification

            if "error" in classification:
                st.error(f"❌ Fehler: {classification['error']}")
            else:
                st.success("✅ Analyse abgeschlossen!")

                # Vorhersage anzeigen
                predicted_class = classification.get("predicted_class", "Unbekannt")
                confidence = classification.get("confidence", 0.0)

                st.metric(
                    "🎯 Vorhersage", predicted_class, f"Konfidenz: {confidence:.2%}"
                )

                # Wahrscheinlichkeiten für alle Klassen
                if "class_probabilities" in classification:
                    st.write("### 📈 Klassenwahrscheinlichkeiten")

                    probs = classification["class_probabilities"]
                    for class_name, prob in probs.items():
                        st.write(f"**{class_name}**: {prob:.2%}")
                        st.progress(prob)

        # Ähnliche Bilder anzeigen
        if (
            hasattr(st.session_state, "similar_images")
            and st.session_state.similar_images
        ):
            st.write("---")
            st.header("🔍 Ähnliche Bilder")

            similar_images = st.session_state.similar_images

            for i, similar in enumerate(similar_images[:3]):  # Top 3 anzeigen
                with st.expander(
                    f"Ähnliches Bild {i + 1} (Ähnlichkeit: {similar['similarity']:.3f})"
                ):
                    st.write(f"**Datei**: {similar['filename']}")
                    st.write(f"**Kategorie**: {similar['category']}")
                    st.write(f"**Split**: {similar['split']}")

                    # Bild anzeigen falls vorhanden
                    image_path = Path(similar["image_path"])
                    if image_path.exists():
                        try:
                            similar_image = Image.open(image_path)
                            st.image(similar_image, use_column_width=True)
                        except Exception as e:
                            st.write(f"Bild konnte nicht geladen werden: {e}")

    # Footer
    st.write("---")
    st.write("### ℹ️ Informationen")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.write("**Unterstützte Krankheiten:**")
        st.write("- Gesunde Pflanzen")
        st.write("- Krankheit A")
        st.write("- Krankheit B")

    with info_col2:
        st.write("**Verwendete Technologien:**")
        st.write("- Deep Learning (ResNet50)")
        st.write("- Vision-Language Models (CLIP)")
        st.write("- Vector Database (Qdrant)")

    with info_col3:
        st.write("**Funktionen:**")
        st.write("- Krankheitserkennung")
        st.write("- Ähnlichkeitssuche")
        st.write("- Konfidenzanalyse")


def main():
    """Hauptfunktion"""
    try:
        create_streamlit_app()
    except Exception as e:
        st.error(f"Fehler beim Starten der Anwendung: {e}")
        logger.error(f"Fehler beim Starten der Anwendung: {e}")


if __name__ == "__main__":
    main()
