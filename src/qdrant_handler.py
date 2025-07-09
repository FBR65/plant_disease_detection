"""
Erweiterte Qdrant-Integration für Pflanzenkrankheitserkennung
Kombiniert VLM-Embeddings mit strukturierter Metadaten-Speicherung
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)
from qdrant_client.http.exceptions import ResponseHandlingException
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pydantic import BaseModel
from typing import Dict, Any


# SimilarCase direkt hier definieren um zirkuläre Imports zu vermeiden
class SimilarCase(BaseModel):
    """Ähnlicher Fall aus der Qdrant-Suche"""

    image_id: str
    similarity_score: float
    disease_label: str
    metadata: Dict[str, Any]


logger = logging.getLogger(__name__)


class PlantDiseaseQdrantHandler:
    """
    Erweiterte Qdrant-Integration für Pflanzenkrankheitserkennung

    Features:
    - VLM-basierte Bild-Embeddings (CLIP)
    - Strukturierte Metadaten-Speicherung
    - Ähnlichkeitssuche mit Filterung
    - Batch-Upload und -Suche
    - Automatische Collection-Verwaltung
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "plant_disease_embeddings",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name

        # Qdrant Client initialisieren
        self.client = QdrantClient(host=host, port=port)

        # CLIP-Modell für Embeddings
        self.clip_model = None
        self.clip_processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._initialize_clip_model()
        self._ensure_collection_exists()

    def _initialize_clip_model(self):
        """CLIP-Modell für VLM-Embeddings initialisieren"""
        try:
            model_name = "openai/clip-vit-base-patch32"
            logger.info(f"Lade CLIP-Modell: {model_name}")

            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)

            self.clip_model.to(self.device)
            self.clip_model.eval()

            logger.info(f"CLIP-Modell geladen auf: {self.device}")

        except Exception as e:
            logger.error(f"Fehler beim Laden des CLIP-Modells: {e}")
            raise

    def _ensure_collection_exists(self):
        """Collection erstellen falls sie nicht existiert"""
        try:
            # Prüfe ob Collection existiert
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Erstelle neue Collection: {self.collection_name}")

                # CLIP ViT-B/32 hat 512 Dimensionen
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=512,  # CLIP ViT-B/32 Embedding-Größe
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' erstellt")
            else:
                logger.info(f"Collection '{self.collection_name}' bereits vorhanden")

        except ResponseHandlingException as e:
            logger.error(f"Qdrant-Verbindungsfehler: {e}")
            raise
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Collection: {e}")
            raise

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generiert VLM-Embedding für ein Bild

        Args:
            image_path: Pfad zum Bild

        Returns:
            512-dimensionales Embedding-Vektor
        """
        try:
            # Bild laden und vorverarbeiten
            image = Image.open(image_path).convert("RGB")

            # CLIP-Preprocessing
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Embedding generieren
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            # Normalisieren und zu numpy konvertieren
            embedding = image_features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)  # L2-Normalisierung

            return embedding

        except Exception as e:
            logger.error(f"Fehler beim Generieren des Embeddings für {image_path}: {e}")
            raise

    def add_image_to_database(
        self,
        image_path: str,
        disease_label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fügt ein Bild zur Qdrant-Datenbank hinzu

        Args:
            image_path: Pfad zum Bild
            disease_label: Krankheitslabel
            metadata: Zusätzliche Metadaten

        Returns:
            Eindeutige ID des gespeicherten Bildes
        """
        try:
            # Embedding generieren
            embedding = self.generate_image_embedding(image_path)

            # Eindeutige ID basierend auf Dateipfad und Inhalt
            image_id = self._generate_image_id(image_path)

            # Metadaten vorbereiten
            payload = {
                "image_path": str(image_path),
                "disease_label": disease_label,
                "upload_timestamp": datetime.now().isoformat(),
                "embedding_model": "openai/clip-vit-base-patch32",
                "file_size": Path(image_path).stat().st_size,
                "image_id": image_id,
            }

            # Zusätzliche Metadaten hinzufügen
            if metadata:
                payload.update(metadata)

            # In Qdrant speichern
            point = PointStruct(id=image_id, vector=embedding.tolist(), payload=payload)

            self.client.upsert(collection_name=self.collection_name, points=[point])

            logger.info(f"Bild gespeichert: {image_path} -> ID: {image_id}")
            return image_id

        except Exception as e:
            logger.error(f"Fehler beim Speichern von {image_path}: {e}")
            raise

    def batch_add_images(
        self, image_data: List[Tuple[str, str, Dict[str, Any]]], batch_size: int = 50
    ) -> List[str]:
        """
        Batch-Upload von Bildern für bessere Performance

        Args:
            image_data: Liste von (image_path, disease_label, metadata) Tupeln
            batch_size: Größe der Upload-Batches

        Returns:
            Liste der generierten Bild-IDs
        """
        all_ids = []

        for i in range(0, len(image_data), batch_size):
            batch = image_data[i : i + batch_size]
            batch_points = []
            batch_ids = []

            logger.info(f"Verarbeite Batch {i // batch_size + 1}: {len(batch)} Bilder")

            for image_path, disease_label, metadata in batch:
                try:
                    # Embedding generieren
                    embedding = self.generate_image_embedding(image_path)
                    image_id = self._generate_image_id(image_path)

                    # Payload erstellen
                    payload = {
                        "image_path": str(image_path),
                        "disease_label": disease_label,
                        "upload_timestamp": datetime.now().isoformat(),
                        "embedding_model": "openai/clip-vit-base-patch32",
                        "file_size": Path(image_path).stat().st_size,
                        "image_id": image_id,
                    }

                    if metadata:
                        payload.update(metadata)

                    # Point erstellen
                    point = PointStruct(
                        id=image_id, vector=embedding.tolist(), payload=payload
                    )

                    batch_points.append(point)
                    batch_ids.append(image_id)

                except Exception as e:
                    logger.error(f"Fehler bei Bild {image_path}: {e}")
                    continue

            # Batch upload
            if batch_points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name, points=batch_points
                    )
                    all_ids.extend(batch_ids)
                    logger.info(
                        f"Batch erfolgreich hochgeladen: {len(batch_points)} Bilder"
                    )
                except Exception as e:
                    logger.error(f"Fehler beim Batch-Upload: {e}")

        logger.info(f"Gesamt hochgeladen: {len(all_ids)} Bilder")
        return all_ids

    def search_similar_images(
        self,
        query_image_path: str,
        limit: int = 5,
        disease_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[SimilarCase]:
        """
        Sucht ähnliche Bilder in der Qdrant-Datenbank

        Args:
            query_image_path: Pfad zum Abfragebild
            limit: Maximale Anzahl Ergebnisse
            disease_filter: Filtere nach spezifischer Krankheit
            min_similarity: Minimale Ähnlichkeit (0.0-1.0)

        Returns:
            Liste ähnlicher Fälle
        """
        try:
            # Query-Embedding generieren
            query_embedding = self.generate_image_embedding(query_image_path)

            # Filter vorbereiten
            search_filter = None
            if disease_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="disease_label", match=MatchValue(value=disease_filter)
                        )
                    ]
                )

            # Ähnlichkeitssuche durchführen
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit,
                # params=SearchParams(hnsw_ef=128, exact=False),  # Temporär deaktiviert
            )

            # Ergebnisse zu SimilarCase-Objekten konvertieren
            similar_cases = []
            for result in search_results:
                if result.score >= min_similarity:
                    similar_case = SimilarCase(
                        image_id=str(result.id),
                        similarity_score=result.score,
                        disease_label=result.payload.get("disease_label", "Unknown"),
                        metadata=result.payload,
                    )
                    similar_cases.append(similar_case)

            logger.info(
                f"Gefunden: {len(similar_cases)} ähnliche Bilder für {query_image_path}"
            )
            return similar_cases

        except Exception as e:
            logger.error(f"Fehler bei Ähnlichkeitssuche: {e}")
            return []

    def get_database_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über die Datenbank zurück"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            # Krankheits-Verteilung abfragen
            # (Vereinfacht - in echter Implementation würde man Aggregations verwenden)
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Für Demo-Zwecke begrenzt
                with_payload=True,
                with_vectors=False,
            )[0]

            disease_counts = {}
            for point in all_points:
                disease = point.payload.get("disease_label", "Unknown")
                disease_counts[disease] = disease_counts.get(disease, 0) + 1

            return {
                "total_images": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "disease_distribution": disease_counts,
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {}

    def _generate_image_id(self, image_path: str) -> str:
        """Generiert eindeutige UUID für ein Bild"""
        import uuid

        # Kombination aus Pfad und Dateigröße für Eindeutigkeit
        path_str = str(Path(image_path).resolve())
        try:
            file_size = Path(image_path).stat().st_size
            unique_string = f"{path_str}_{file_size}"
        except Exception:
            unique_string = path_str

        # SHA256-Hash für konsistente Seeds, dann UUID generieren
        import hashlib

        seed = int(hashlib.sha256(unique_string.encode()).hexdigest()[:8], 16)

        # Deterministischen UUID basierend auf Seed generieren
        import random

        random.seed(seed)
        return str(uuid.uuid4())

    def delete_image(self, image_id: str) -> bool:
        """Löscht ein Bild aus der Datenbank"""
        try:
            self.client.delete(
                collection_name=self.collection_name, points_selector=[image_id]
            )
            logger.info(f"Bild gelöscht: {image_id}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Löschen von {image_id}: {e}")
            return False

    def update_image_metadata(
        self, image_id: str, new_metadata: Dict[str, Any]
    ) -> bool:
        """Aktualisiert Metadaten eines Bildes"""
        try:
            # Aktuelle Payload abrufen
            current_point = self.client.retrieve(
                collection_name=self.collection_name, ids=[image_id], with_payload=True
            )[0]

            # Metadaten aktualisieren
            updated_payload = current_point.payload.copy()
            updated_payload.update(new_metadata)
            updated_payload["last_updated"] = datetime.now().isoformat()

            # Point aktualisieren
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=updated_payload,
                points=[image_id],
            )

            logger.info(f"Metadaten aktualisiert für: {image_id}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Metadaten für {image_id}: {e}")
            return False


# Hilfsfunktionen für Integration mit anderen Modulen
def initialize_qdrant_handler(
    config_path: Optional[str] = None,
) -> PlantDiseaseQdrantHandler:
    """Factory-Funktion zur Initialisierung des Qdrant-Handlers"""

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = json.load(f)

        return PlantDiseaseQdrantHandler(
            host=config.get("host", "localhost"),
            port=config.get("port", 6333),
            collection_name=config.get("collection_name", "plant_disease_embeddings"),
        )
    else:
        # Standard-Konfiguration
        return PlantDiseaseQdrantHandler()


async def populate_database_from_dataset(
    qdrant_handler: PlantDiseaseQdrantHandler, dataset_path: str
) -> int:
    """
    Füllt die Qdrant-Datenbank mit Bildern aus einem Dataset

    Args:
        qdrant_handler: Initialisierter Qdrant-Handler
        dataset_path: Pfad zum Dataset (z.B. data/PlantDoc/train)

    Returns:
        Anzahl erfolgreich hochgeladener Bilder
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        logger.error(f"Dataset-Pfad nicht gefunden: {dataset_path}")
        return 0

    # Alle Bilder im Dataset sammeln
    image_data = []
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            disease_label = class_dir.name

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    metadata = {
                        "split": "train" if "train" in str(dataset_path) else "test",
                        "class_directory": disease_label,
                        "original_filename": img_path.name,
                    }

                    image_data.append((str(img_path), disease_label, metadata))

    logger.info(f"Gefunden: {len(image_data)} Bilder zum Upload")

    # Batch-Upload durchführen
    uploaded_ids = qdrant_handler.batch_add_images(image_data, batch_size=25)

    logger.info(
        f"Upload abgeschlossen: {len(uploaded_ids)} Bilder erfolgreich hochgeladen"
    )
    return len(uploaded_ids)


if __name__ == "__main__":
    # Test der Qdrant-Integration
    logging.basicConfig(level=logging.INFO)

    try:
        # Handler initialisieren
        handler = PlantDiseaseQdrantHandler()

        # Statistiken anzeigen
        stats = handler.get_database_statistics()
        print("Datenbank-Statistiken:", json.dumps(stats, indent=2))

        print("Qdrant-Integration erfolgreich getestet!")

    except Exception as e:
        print(f"Test-Fehler: {e}")
        print("Stelle sicher, dass Qdrant läuft: docker run -p 6333:6333 qdrant/qdrant")
