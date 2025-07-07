"""
Qdrant Handler für Vector Database Operationen
"""

import logging
import json
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    CollectionInfo,
    UpdateResult,
)
from qdrant_client.http.exceptions import UnexpectedResponse

# Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantHandler:
    """Handler für Qdrant Vector Database Operationen"""

    def __init__(self, config_path: str = "config/qdrant_config.yaml"):
        """
        Initialisiert den Qdrant Handler

        Args:
            config_path: Pfad zur Qdrant-Konfiguration
        """
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.collection_name = self.config["collection"]["name"]

        logger.info(
            f"Qdrant Handler initialisiert für Collection: {self.collection_name}"
        )

    def _load_config(self, config_path: str) -> dict:
        """Lädt die Qdrant-Konfiguration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(
                f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardwerte."
            )
            return {
                "connection": {"host": "localhost", "port": 6333},
                "collection": {
                    "name": "plant_disease_embeddings",
                    "vector_size": 512,
                    "distance": "Cosine",
                },
            }

    def _initialize_client(self) -> QdrantClient:
        """Initialisiert den Qdrant Client"""
        conn_config = self.config["connection"]

        try:
            if "cloud_config" in self.config:
                # Qdrant Cloud
                cloud_config = self.config["cloud_config"]
                client = QdrantClient(
                    url=cloud_config["url"], api_key=cloud_config["api_key"]
                )
            else:
                # Lokaler Qdrant Server
                client = QdrantClient(
                    host=conn_config["host"],
                    port=conn_config["port"],
                    timeout=conn_config.get("timeout", 30),
                )

            # Verbindung testen
            collections = client.get_collections()
            logger.info(
                f"Verbindung zu Qdrant erfolgreich. Verfügbare Collections: {len(collections.collections)}"
            )

            return client

        except Exception as e:
            logger.error(f"Fehler beim Verbinden zu Qdrant: {e}")
            raise

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Erstellt die Collection in Qdrant

        Args:
            recreate: Ob eine existierende Collection neu erstellt werden soll

        Returns:
            True wenn erfolgreich erstellt
        """
        collection_config = self.config["collection"]

        try:
            if recreate:
                logger.info(f"Erstelle Collection '{self.collection_name}' neu...")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=collection_config["vector_size"],
                        distance=Distance.COSINE
                        if collection_config["distance"] == "Cosine"
                        else Distance.DOT,
                    ),
                )
            else:
                # Prüfen ob Collection bereits existiert
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    logger.info(
                        f"Collection '{self.collection_name}' existiert bereits mit {collection_info.points_count} Punkten"
                    )
                    return True
                except UnexpectedResponse:
                    # Collection existiert nicht, erstellen
                    logger.info(f"Erstelle neue Collection '{self.collection_name}'...")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=collection_config["vector_size"],
                            distance=Distance.COSINE
                            if collection_config["distance"] == "Cosine"
                            else Distance.DOT,
                        ),
                    )

            logger.info(f"Collection '{self.collection_name}' erfolgreich erstellt")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Collection: {e}")
            return False

    def upload_embeddings(
        self, embeddings_data: List[Dict], batch_size: int = None
    ) -> bool:
        """
        Lädt Embeddings in Qdrant hoch

        Args:
            embeddings_data: Liste von Embedding-Dictionarys
            batch_size: Batch-Größe für Upload

        Returns:
            True wenn erfolgreich
        """
        if batch_size is None:
            batch_size = self.config.get("batch_upload", {}).get("batch_size", 100)

        try:
            points = []

            for i, data in enumerate(embeddings_data):
                # Embedding auswählen (bevorzuge multimodal_embedding)
                if "multimodal_embedding" in data:
                    vector = data["multimodal_embedding"]
                elif "combined_embedding" in data:
                    vector = data["combined_embedding"]
                elif "image_embedding" in data:
                    vector = data["image_embedding"]
                else:
                    logger.warning(f"Kein Embedding in Daten-Index {i} gefunden")
                    continue

                # Punkt erstellen
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "image_path": data.get("image_path", ""),
                        "relative_path": data.get("relative_path", ""),
                        "category": data.get("category", ""),
                        "split": data.get("split", ""),
                        "filename": data.get("filename", ""),
                        "embedding_dim": data.get("embedding_dim", len(vector)),
                        "upload_timestamp": datetime.now().isoformat(),
                        "index": i,
                    },
                )
                points.append(point)

                # Batch-Upload
                if len(points) >= batch_size:
                    self._upload_batch(points)
                    logger.info(
                        f"Hochgeladen: {i + 1}/{len(embeddings_data)} Embeddings"
                    )
                    points = []

            # Restliche Punkte hochladen
            if points:
                self._upload_batch(points)

            logger.info(
                f"Alle {len(embeddings_data)} Embeddings erfolgreich hochgeladen"
            )
            return True

        except Exception as e:
            logger.error(f"Fehler beim Hochladen der Embeddings: {e}")
            return False

    def _upload_batch(self, points: List[PointStruct]) -> bool:
        """Lädt einen Batch von Punkten hoch"""
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Batch-Upload: {e}")
            return False

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        category_filter: Optional[str] = None,
        split_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Sucht ähnliche Vektoren

        Args:
            query_vector: Such-Vektor
            limit: Anzahl der Ergebnisse
            category_filter: Filter nach Kategorie
            split_filter: Filter nach Split (train/val)

        Returns:
            Liste der ähnlichsten Ergebnisse
        """
        try:
            # Filter erstellen
            query_filter = None
            conditions = []

            if category_filter:
                conditions.append(
                    FieldCondition(
                        key="category", match=MatchValue(value=category_filter)
                    )
                )

            if split_filter:
                conditions.append(
                    FieldCondition(key="split", match=MatchValue(value=split_filter))
                )

            if conditions:
                query_filter = Filter(must=conditions)

            # Suche durchführen
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            # Ergebnisse formatieren
            results = []
            for result in search_results:
                results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "image_path": result.payload.get("image_path", ""),
                        "category": result.payload.get("category", ""),
                        "split": result.payload.get("split", ""),
                        "filename": result.payload.get("filename", ""),
                        "payload": result.payload,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Fehler bei der Suche: {e}")
            return []

    def get_collection_info(self) -> Optional[Dict]:
        """
        Gibt Informationen über die Collection zurück

        Returns:
            Dictionary mit Collection-Informationen
        """
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance,
                "status": info.status,
            }

        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Collection-Info: {e}")
            return None

    def delete_by_filter(
        self, category: Optional[str] = None, split: Optional[str] = None
    ) -> bool:
        """
        Löscht Punkte basierend auf Filtern

        Args:
            category: Kategorie zum Löschen
            split: Split zum Löschen

        Returns:
            True wenn erfolgreich
        """
        try:
            conditions = []

            if category:
                conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category))
                )

            if split:
                conditions.append(
                    FieldCondition(key="split", match=MatchValue(value=split))
                )

            if not conditions:
                logger.warning("Keine Filter angegeben - keine Löschung durchgeführt")
                return False

            delete_filter = Filter(must=conditions)

            result = self.client.delete(
                collection_name=self.collection_name, points_selector=delete_filter
            )

            logger.info(f"Löschung abgeschlossen. Status: {result}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Löschen: {e}")
            return False

    def backup_collection(self, backup_path: str) -> bool:
        """
        Erstellt ein Backup der Collection

        Args:
            backup_path: Pfad für das Backup

        Returns:
            True wenn erfolgreich
        """
        try:
            # Alle Punkte abrufen
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Anpassen je nach Datenmenge
            )

            backup_data = {
                "collection_name": self.collection_name,
                "backup_timestamp": datetime.now().isoformat(),
                "points": [],
            }

            for point in all_points[0]:
                backup_data["points"].append(
                    {"id": point.id, "vector": point.vector, "payload": point.payload}
                )

            # Backup speichern
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"Backup erfolgreich erstellt: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Backup: {e}")
            return False


if __name__ == "__main__":
    # Beispiel für die Verwendung
    handler = QdrantHandler()

    # Collection erstellen
    handler.create_collection()

    # Collection-Info anzeigen
    info = handler.get_collection_info()
    if info:
        print(f"Collection Info: {info}")

    print("Qdrant Handler bereit für Operationen")
