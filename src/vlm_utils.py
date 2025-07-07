"""
Utilities für Vision-Language Model (VLM) Embeddings
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import yaml
from typing import List, Dict, Union, Optional
from transformers import CLIPProcessor, CLIPModel
import logging

# Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMEmbedder:
    """Klasse für die Erstellung von Vision-Language Model Embeddings"""

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", device: str = None
    ):
        """
        Initialisiert den VLM Embedder

        Args:
            model_name: Name des CLIP-Modells
            device: Gerät für die Berechnung (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialisiere VLM Embedder mit {model_name} auf {self.device}")

        # Modell und Processor laden
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Modell auf das richtige Gerät verschieben
        self.model.to(self.device)
        self.model.eval()

        # Embedding-Dimension ermitteln
        self.embedding_dim = self.model.config.projection_dim

        logger.info(f"Embedding-Dimension: {self.embedding_dim}")

    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Erstellt ein Embedding für ein Bild

        Args:
            image: Pfad zum Bild oder PIL Image

        Returns:
            Normalisiertes Bild-Embedding
        """
        # Bild laden falls Pfad übergeben wurde
        if isinstance(image, str):
            image = Image.open(image)

        # Bild in RGB konvertieren
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Bild verarbeiten
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Embedding erstellen
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalisieren
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return image_features.cpu().numpy().squeeze()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Erstellt ein Embedding für Text

        Args:
            text: Textbeschreibung

        Returns:
            Normalisiertes Text-Embedding
        """
        # Text verarbeiten
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Embedding erstellen
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalisieren
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return text_features.cpu().numpy().squeeze()

    def encode_multimodal(
        self, image: Union[str, Image.Image], text: str
    ) -> Dict[str, np.ndarray]:
        """
        Erstellt sowohl Bild- als auch Text-Embeddings

        Args:
            image: Pfad zum Bild oder PIL Image
            text: Textbeschreibung

        Returns:
            Dictionary mit separaten und kombinierten Embeddings
        """
        # Bild laden falls Pfad übergeben wurde
        if isinstance(image, str):
            image = Image.open(image)

        # Bild in RGB konvertieren
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Beide Modalitäten verarbeiten
        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Embeddings erstellen
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Features extrahieren und normalisieren
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

            # Kombiniertes Embedding (Durchschnitt)
            combined_features = (image_features + text_features) / 2
            combined_features = combined_features / combined_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return {
            "image_embedding": image_features.cpu().numpy().squeeze(),
            "text_embedding": text_features.cpu().numpy().squeeze(),
            "combined_embedding": combined_features.cpu().numpy().squeeze(),
        }

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Embeddings

        Args:
            embedding1: Erstes Embedding
            embedding2: Zweites Embedding

        Returns:
            Kosinus-Ähnlichkeit zwischen 0 und 1
        """
        # Embeddings normalisieren
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Kosinus-Ähnlichkeit berechnen
        similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def batch_encode_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Erstellt Embeddings für mehrere Bilder

        Args:
            image_paths: Liste von Bildpfaden

        Returns:
            Liste von Embeddings
        """
        embeddings = []

        for i, image_path in enumerate(image_paths):
            try:
                embedding = self.encode_image(image_path)
                embeddings.append(embedding)

                if (i + 1) % 50 == 0:
                    logger.info(f"Verarbeitet: {i + 1}/{len(image_paths)} Bilder")

            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten von {image_path}: {e}")
                # Null-Embedding hinzufügen
                embeddings.append(np.zeros(self.embedding_dim))

        return embeddings


class PlantDiseaseEmbedder:
    """Spezialisierte Klasse für Pflanzenkrankheits-Embeddings"""

    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        """
        Initialisiert den Plant Disease Embedder

        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config = self._load_config(config_path)
        self.embedder = VLMEmbedder()

        # Vordefinierte Textbeschreibungen für Kategorien
        self.category_descriptions = {
            "healthy": [
                "healthy green plant leaf",
                "normal plant leaf without disease",
                "clean healthy plant foliage",
            ],
            "disease_A": [
                "plant leaf with disease A symptoms",
                "diseased plant leaf showing pathogen A",
                "plant leaf infected with disease A",
            ],
            "disease_B": [
                "plant leaf with disease B symptoms",
                "diseased plant leaf showing pathogen B",
                "plant leaf infected with disease B",
            ],
        }

    def _load_config(self, config_path: str) -> dict:
        """Lädt die Konfiguration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(
                f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardwerte."
            )
            return {"data_path": "data/raw"}

    def create_embeddings_for_dataset(
        self, output_path: str = "data/processed/embeddings.json"
    ):
        """
        Erstellt Embeddings für den gesamten Datensatz

        Args:
            output_path: Pfad für die Ausgabe-JSON-Datei
        """
        embeddings_data = []
        data_path = Path(self.config["data_path"])

        logger.info("Starte Embedding-Erstellung für Datensatz...")

        total_images = 0
        processed_images = 0

        # Zuerst alle Bilder zählen
        for split in ["train", "val"]:
            for category in self.category_descriptions.keys():
                category_path = data_path / split / category
                if category_path.exists():
                    total_images += len(list(category_path.glob("*.jpg")))

        logger.info(f"Gesamt zu verarbeitende Bilder: {total_images}")

        # Embeddings erstellen
        for split in ["train", "val"]:
            for category in self.category_descriptions.keys():
                category_path = data_path / split / category

                if not category_path.exists():
                    logger.warning(f"Pfad nicht gefunden: {category_path}")
                    continue

                logger.info(f"Verarbeite {split}/{category}...")

                # Verschiedene Textbeschreibungen für diese Kategorie
                descriptions = self.category_descriptions[category]

                for image_file in category_path.glob("*.jpg"):
                    try:
                        # Für jede Textbeschreibung ein Embedding erstellen
                        category_embeddings = []

                        for desc in descriptions:
                            multimodal_result = self.embedder.encode_multimodal(
                                str(image_file), desc
                            )
                            category_embeddings.append(
                                multimodal_result["combined_embedding"]
                            )

                        # Durchschnitt der Embeddings für robustere Repräsentation
                        avg_embedding = np.mean(category_embeddings, axis=0)

                        # Auch ein reines Bild-Embedding erstellen
                        image_embedding = self.embedder.encode_image(str(image_file))

                        # Metadaten sammeln
                        embeddings_data.append(
                            {
                                "image_path": str(image_file),
                                "relative_path": str(
                                    image_file.relative_to(data_path.parent)
                                ),
                                "category": category,
                                "split": split,
                                "filename": image_file.name,
                                "image_embedding": image_embedding.tolist(),
                                "multimodal_embedding": avg_embedding.tolist(),
                                "embedding_dim": len(avg_embedding),
                            }
                        )

                        processed_images += 1

                        if processed_images % 25 == 0:
                            logger.info(
                                f"Fortschritt: {processed_images}/{total_images} "
                                f"({processed_images / total_images * 100:.1f}%)"
                            )

                    except Exception as e:
                        logger.error(f"Fehler beim Verarbeiten von {image_file}: {e}")

        # Embeddings speichern
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(embeddings_data, f, indent=2)

        logger.info(
            f"Embeddings für {len(embeddings_data)} Bilder gespeichert in: {output_path}"
        )

        return embeddings_data

    def load_embeddings(
        self, embeddings_path: str = "data/processed/embeddings.json"
    ) -> List[Dict]:
        """
        Lädt gespeicherte Embeddings

        Args:
            embeddings_path: Pfad zur Embeddings-Datei

        Returns:
            Liste von Embedding-Dictionarys
        """
        try:
            with open(embeddings_path, "r") as f:
                embeddings_data = json.load(f)

            logger.info(f"Embeddings geladen: {len(embeddings_data)} Einträge")
            return embeddings_data

        except FileNotFoundError:
            logger.error(f"Embeddings-Datei nicht gefunden: {embeddings_path}")
            return []

    def find_similar_images(
        self, query_image_path: str, embeddings_data: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        """
        Findet ähnliche Bilder basierend auf Embeddings

        Args:
            query_image_path: Pfad zum Suchbild
            embeddings_data: Liste von Embeddings
            top_k: Anzahl der zurückzugebenden Ergebnisse

        Returns:
            Liste der ähnlichsten Bilder mit Similarity-Scores
        """
        # Query-Embedding erstellen
        query_embedding = self.embedder.encode_image(query_image_path)

        # Ähnlichkeiten berechnen
        similarities = []

        for data in embeddings_data:
            # Multimodales Embedding verwenden
            stored_embedding = np.array(data["multimodal_embedding"])

            similarity = self.embedder.compute_similarity(
                query_embedding, stored_embedding
            )

            similarities.append(
                {
                    "image_path": data["image_path"],
                    "category": data["category"],
                    "split": data["split"],
                    "similarity": similarity,
                    "filename": data["filename"],
                }
            )

        # Nach Ähnlichkeit sortieren
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]


if __name__ == "__main__":
    # Beispiel für die Verwendung
    embedder = PlantDiseaseEmbedder()

    # Embeddings für Datensatz erstellen (auskommentiert für Demo)
    # embeddings_data = embedder.create_embeddings_for_dataset()

    # Embeddings laden und ähnliche Bilder finden
    # embeddings_data = embedder.load_embeddings()
    # if embeddings_data:
    #     similar_images = embedder.find_similar_images(
    #         "data/raw/train/healthy/example.jpg",
    #         embeddings_data,
    #         top_k=5
    #     )
    #     print("Ähnliche Bilder:")
    #     for img in similar_images:
    #         print(f"  {img['filename']} (Ähnlichkeit: {img['similarity']:.3f})")

    print("VLM Utils initialisiert. Verwende die Klassen für Embedding-Operationen.")
