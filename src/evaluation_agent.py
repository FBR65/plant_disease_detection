"""
PydanticAI-basierter Bewertungsagent für Pflanzenkrankheitserkennung
Kombiniert CNN-Klassifikation mit VLM-Ähnlichkeitssuche und LLM-Bewertung
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import base64
from io import BytesIO

# .env-Datei laden
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fallback ohne python-dotenv
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class DiseaseAssessment(BaseModel):
    """Strukturierte Krankheitsbewertung"""

    has_disease: bool = Field(description="Ob eine Krankheit erkannt wurde")
    disease_name: str = Field(description="Name der erkannten Krankheit oder 'Gesund'")
    confidence_score: float = Field(description="Konfidenz-Score zwischen 0.0 und 1.0")
    reasoning: str = Field(description="Begründung der Bewertung")
    symptoms_detected: List[str] = Field(description="Liste der erkannten Symptome")
    recommendation: str = Field(description="Handlungsempfehlung")
    severity: str = Field(
        description="Schweregrad: 'Leicht', 'Mittel', 'Schwer', 'Keine'"
    )


class SimilarCase(BaseModel):
    """Ähnlicher Fall aus der Qdrant-Suche"""

    image_id: str
    similarity_score: float
    disease_label: str
    metadata: Dict[str, Any]


class PlantDiseaseEvaluationAgent:
    """
    PydanticAI-basierter Agent zur intelligenten Pflanzenkrankheitsbewertung

    Kombiniert:
    - CNN-Klassifikation
    - VLM-basierte Ähnlichkeitssuche
    - LLM-gestützte finale Bewertung
    """

    def __init__(self):
        self.provider = None
        self.model = None
        self.agent = None
        self.setup_llm()
        self.setup_agent()

    def setup_llm(self):
        """LLM-Konfiguration für OpenAI-kompatible Endpunkte"""
        self.llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        self.llm_api_key = os.getenv("API_KEY", "ollama")
        self.llm_model_name = os.getenv("MODEL_NAME", "granite-code:8b")

        logger.info(f"Initialisiere LLM: {self.llm_model_name} @ {self.llm_endpoint}")

        self.provider = OpenAIProvider(
            base_url=self.llm_endpoint, api_key=self.llm_api_key
        )
        self.model = OpenAIModel(provider=self.provider, model_name=self.llm_model_name)

    def setup_agent(self):
        """PydanticAI Agent konfigurieren mit verbessertem System-Prompt"""
        system_prompt = """Du bist ein erfahrener Pflanzenpathologe und KI-Spezialist. 
Deine Aufgabe ist es, Pflanzenkrankheiten basierend auf verschiedenen Datenquellen zu bewerten:

1. CNN-KLASSIFIKATION: Ein Deep Learning-Modell hat eine Vorhersage gemacht
2. ÄHNLICHE FÄLLE: Eine Vektorsuche hat ähnliche Bilder aus der Datenbank gefunden
3. BILDBESCHREIBUNG: Eine detaillierte Beschreibung des aktuellen Bildes

BEWERTUNGSKRITERIEN:
- Analysiere ALLE verfügbaren Informationen kritisch
- Berücksichtige die Konfidenz-Scores der verschiedenen Quellen
- Erkenne typische Krankheitssymptome (Flecken, Verfärbungen, Deformation, etc.)
- Bewerte die Konsistenz zwischen CNN-Vorhersage und ähnlichen Fällen
- Gib eine fundierte finale Einschätzung ab

WICHTIGE HINWEISE:
- Wenn CNN und ähnliche Fälle widersprüchlich sind, analysiere genauer
- Niedrige Konfidenz-Scores erfordern vorsichtige Bewertung
- Bei unklaren Fällen empfehle zusätzliche Untersuchungen
- Berücksichtige auch umweltbedingte Faktoren (Licht, Winkel, Bildqualität)

Antworte IMMER im strukturierten Format mit allen erforderlichen Feldern."""

        self.agent = Agent(
            model=self.model,
            result_type=DiseaseAssessment,
            retries=5,
            system_prompt=system_prompt,
        )

    async def evaluate_plant_image(
        self,
        image_path: str,
        cnn_prediction: Dict[str, Any],
        similar_cases: List[SimilarCase],
        image_description: str = "",
    ) -> DiseaseAssessment:
        """
        Hauptbewertungsfunktion: Kombiniert alle Datenquellen

        Args:
            image_path: Pfad zum Bild
            cnn_prediction: CNN-Klassifikationsergebnis
            similar_cases: Ähnliche Fälle aus Qdrant
            image_description: Zusätzliche Bildbeschreibung
        """

        # Bild für LLM vorbereiten
        image_b64 = self._encode_image_to_base64(image_path)

        # Kontext für LLM aufbauen
        context = self._build_evaluation_context(
            cnn_prediction, similar_cases, image_description
        )

        # LLM-Bewertung durchführen
        prompt = f"""Bewerte dieses Pflanzenbild auf Krankheiten:

{context}

Analysiere das Bild sorgfältig und berücksichtige alle verfügbaren Informationen. 
Gib eine strukturierte Bewertung zurück."""

        try:
            result = await self.agent.run(
                prompt,
                message_history=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
            )
            return result.data
        except Exception as e:
            logger.error(f"Fehler bei LLM-Bewertung: {e}")
            return self._create_fallback_assessment(cnn_prediction, similar_cases)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Bild zu Base64 für LLM konvertieren"""
        with Image.open(image_path) as img:
            # Auf maximale Größe für LLM beschränken
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_data = buffer.getvalue()

            return base64.b64encode(img_data).decode("utf-8")

    def _build_evaluation_context(
        self,
        cnn_prediction: Dict[str, Any],
        similar_cases: List[SimilarCase],
        image_description: str,
    ) -> str:
        """Kontext für LLM-Bewertung aufbauen"""

        context = f"""
=== CNN-KLASSIFIKATION ===
Vorhergesagte Klasse: {cnn_prediction.get("predicted_class", "Unbekannt")}
Konfidenz: {cnn_prediction.get("confidence", 0.0):.3f}
Top-3 Vorhersagen:
"""

        # Top-3 Predictions hinzufügen
        if "top_predictions" in cnn_prediction:
            for i, (cls, conf) in enumerate(cnn_prediction["top_predictions"][:3]):
                context += f"  {i + 1}. {cls}: {conf:.3f}\n"

        context += f"\n=== ÄHNLICHE FÄLLE (Qdrant-Suche) ===\n"
        context += f"Anzahl gefundener ähnlicher Fälle: {len(similar_cases)}\n"

        for i, case in enumerate(similar_cases[:5]):  # Top-5 ähnliche Fälle
            context += f"""
Fall {i + 1}:
  - Ähnlichkeit: {case.similarity_score:.3f}
  - Krankheit: {case.disease_label}
  - Metadaten: {case.metadata}
"""

        if image_description:
            context += f"\n=== BILDBESCHREIBUNG ===\n{image_description}\n"

        return context

    def _create_fallback_assessment(
        self, cnn_prediction: Dict[str, Any], similar_cases: List[SimilarCase]
    ) -> DiseaseAssessment:
        """Fallback-Bewertung wenn LLM nicht verfügbar"""

        cnn_class = cnn_prediction.get("predicted_class", "Unbekannt")
        cnn_confidence = cnn_prediction.get("confidence", 0.0)

        # Einfache Regel-basierte Bewertung
        has_disease = "leaf" in cnn_class.lower() and "healthy" not in cnn_class.lower()

        return DiseaseAssessment(
            has_disease=has_disease,
            disease_name=cnn_class,
            confidence_score=cnn_confidence,
            reasoning=f"Fallback-Bewertung basierend auf CNN-Klassifikation. LLM-Bewertung nicht verfügbar.",
            symptoms_detected=["Automatisch erkannt"] if has_disease else [],
            recommendation="LLM-System prüfen für detaillierte Analyse",
            severity="Unbekannt",
        )

    async def run_comprehensive_evaluation(
        self,
        image_path: str,
        cnn_prediction: Dict[str, Any],
        similar_cases: List[SimilarCase],
        additional_context: str = "",
    ) -> Dict[str, Any]:
        """
        Vollständige Bewertung mit allen verfügbaren Datenquellen

        Returns:
            Umfassender Bewertungsreport
        """

        # LLM-Bewertung durchführen
        assessment = await self.evaluate_plant_image(
            image_path, cnn_prediction, similar_cases, additional_context
        )

        # Konsistenz-Check zwischen verschiedenen Quellen
        consistency_score = self._calculate_consistency_score(
            cnn_prediction, similar_cases, assessment
        )

        # Finale Empfehlung basierend auf allen Faktoren
        final_recommendation = self._generate_final_recommendation(
            assessment, consistency_score, cnn_prediction, similar_cases
        )

        return {
            "llm_assessment": assessment.model_dump(),
            "cnn_prediction": cnn_prediction,
            "similar_cases": [case.model_dump() for case in similar_cases],
            "consistency_score": consistency_score,
            "final_recommendation": final_recommendation,
            "confidence_level": self._calculate_overall_confidence(
                assessment, consistency_score, cnn_prediction
            ),
        }

    def _calculate_consistency_score(
        self,
        cnn_prediction: Dict[str, Any],
        similar_cases: List[SimilarCase],
        llm_assessment: DiseaseAssessment,
    ) -> float:
        """Berechnet Konsistenz zwischen verschiedenen Bewertungsquellen"""

        if not similar_cases:
            return 0.5  # Mittlere Konsistenz ohne ähnliche Fälle

        # Prüfe Übereinstimmung zwischen CNN und ähnlichen Fällen
        cnn_class = cnn_prediction.get("predicted_class", "").lower()
        similar_diseases = [case.disease_label.lower() for case in similar_cases[:3]]

        cnn_similar_match = any(
            cnn_class in disease or disease in cnn_class for disease in similar_diseases
        )

        # Prüfe Übereinstimmung zwischen LLM und anderen Quellen
        llm_disease = llm_assessment.disease_name.lower()
        llm_cnn_match = llm_disease in cnn_class or cnn_class in llm_disease
        llm_similar_match = any(
            llm_disease in disease or disease in llm_disease
            for disease in similar_diseases
        )

        # Gewichtete Konsistenz-Berechnung
        consistency = 0.0
        if cnn_similar_match:
            consistency += 0.4
        if llm_cnn_match:
            consistency += 0.3
        if llm_similar_match:
            consistency += 0.3

        return min(consistency, 1.0)

    def _generate_final_recommendation(
        self,
        assessment: DiseaseAssessment,
        consistency_score: float,
        cnn_prediction: Dict[str, Any],
        similar_cases: List[SimilarCase],
    ) -> str:
        """Generiert finale Handlungsempfehlung"""

        if consistency_score > 0.8 and assessment.confidence_score > 0.8:
            if assessment.has_disease:
                return f"Hohe Sicherheit: {assessment.disease_name} erkannt. {assessment.recommendation}"
            else:
                return "Hohe Sicherheit: Pflanze erscheint gesund."

        elif consistency_score > 0.6 and assessment.confidence_score > 0.6:
            return f"Mittlere Sicherheit: {assessment.recommendation} Weitere Überwachung empfohlen."

        else:
            return (
                "Unsichere Bewertung aufgrund inkonsistenter Datenquellen. "
                "Zusätzliche Experten-Beurteilung oder weitere Bilder empfohlen."
            )

    def _calculate_overall_confidence(
        self,
        assessment: DiseaseAssessment,
        consistency_score: float,
        cnn_prediction: Dict[str, Any],
    ) -> float:
        """Berechnet Gesamt-Konfidenz basierend auf allen Faktoren"""

        llm_confidence = assessment.confidence_score
        cnn_confidence = cnn_prediction.get("confidence", 0.0)

        # Gewichtete Kombination verschiedener Konfidenz-Scores
        overall_confidence = (
            0.4 * llm_confidence + 0.3 * cnn_confidence + 0.3 * consistency_score
        )

        return round(overall_confidence, 3)


# Hilfsfunktionen für Gradio-Integration
async def create_evaluation_agent() -> PlantDiseaseEvaluationAgent:
    """Factory-Funktion für Gradio"""
    return PlantDiseaseEvaluationAgent()


if __name__ == "__main__":
    # Test des Bewertungsagents
    import asyncio

    async def test_agent():
        agent = PlantDiseaseEvaluationAgent()

        # Test-Daten
        cnn_prediction = {
            "predicted_class": "Tomato Early blight leaf",
            "confidence": 0.89,
            "top_predictions": [
                ("Tomato Early blight leaf", 0.89),
                ("Tomato leaf", 0.08),
                ("Tomato leaf late blight", 0.03),
            ],
        }

        similar_cases = [
            SimilarCase(
                image_id="img_001",
                similarity_score=0.92,
                disease_label="Tomato Early blight leaf",
                metadata={"plant": "tomato", "age": "mature"},
            ),
            SimilarCase(
                image_id="img_002",
                similarity_score=0.87,
                disease_label="Tomato Early blight leaf",
                metadata={"plant": "tomato", "age": "young"},
            ),
        ]

        # Mock-Bild (in echtem Szenario wäre das ein echtes Bild)
        test_image = "test_image.jpg"

        try:
            result = await agent.run_comprehensive_evaluation(
                test_image, cnn_prediction, similar_cases
            )
            print("Bewertungsergebnis:", json.dumps(result, indent=2))
        except Exception as e:
            print(f"Test-Fehler: {e}")

    # Test nur ausführen wenn Umgebungsvariablen gesetzt sind
    if os.getenv("BASE_URL"):
        asyncio.run(test_agent())
    else:
        print("Umgebungsvariablen für LLM nicht gesetzt. Test übersprungen.")
