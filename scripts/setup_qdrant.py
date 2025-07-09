"""
Setup-Script für Qdrant-Database Population
Lädt alle Trainingsbilder in die Qdrant-Datenbank
"""

import asyncio
import logging
import argparse
from pathlib import Path
import sys
import os
import json

# Projektpfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from qdrant_handler import PlantDiseaseQdrantHandler, populate_database_from_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_qdrant_database(dataset_path: str, force_recreate: bool = False):
    """
    Qdrant-Datenbank mit Dataset-Bildern füllen

    Args:
        dataset_path: Pfad zum Dataset (z.B. data/PlantDoc)
        force_recreate: Collection neu erstellen wenn sie existiert
    """

    logger.info("🚀 Starte Qdrant-Database Setup...")

    try:
        # Qdrant-Handler initialisieren
        qdrant_handler = PlantDiseaseQdrantHandler()

        # Statistiken vor dem Upload
        initial_stats = qdrant_handler.get_database_statistics()
        logger.info(
            f"📊 Aktuelle DB-Größe: {initial_stats.get('total_images', 0)} Bilder"
        )

        # Wenn Collection bereits Daten hat und force_recreate=False
        if initial_stats.get("total_images", 0) > 0 and not force_recreate:
            logger.info(
                "📋 Collection enthält bereits Daten. Verwende --force-recreate zum Überschreiben."
            )
            return

        # Trainingsdaten hochladen
        train_path = Path(dataset_path) / "train"
        if train_path.exists():
            logger.info(f"📤 Lade Trainingsdaten aus: {train_path}")
            train_count = await populate_database_from_dataset(
                qdrant_handler, str(train_path)
            )
            logger.info(f"✅ Trainingsdaten hochgeladen: {train_count} Bilder")
        else:
            logger.warning(f"⚠️ Trainingspfad nicht gefunden: {train_path}")

        # Testdaten hochladen (optional)
        test_path = Path(dataset_path) / "test"
        if test_path.exists():
            logger.info(f"📤 Lade Testdaten aus: {test_path}")
            test_count = await populate_database_from_dataset(
                qdrant_handler, str(test_path)
            )
            logger.info(f"✅ Testdaten hochgeladen: {test_count} Bilder")
        else:
            logger.warning(f"⚠️ Testpfad nicht gefunden: {test_path}")

        # Finale Statistiken
        final_stats = qdrant_handler.get_database_statistics()
        logger.info("📊 FINALE STATISTIKEN:")
        logger.info(f"  Gesamt Bilder: {final_stats.get('total_images', 0)}")
        logger.info(
            f"  Krankheits-Verteilung: {len(final_stats.get('disease_distribution', {}))} Klassen"
        )

        # Detaillierte Verteilung anzeigen
        disease_dist = final_stats.get("disease_distribution", {})
        if disease_dist:
            logger.info("🏷️ Klassenverteilung:")
            for disease, count in sorted(disease_dist.items()):
                logger.info(f"  {disease}: {count} Bilder")

        logger.info("🎉 Qdrant-Database Setup erfolgreich abgeschlossen!")

        # Test-Suche durchführen
        await test_similarity_search(qdrant_handler, train_path)

    except Exception as e:
        logger.error(f"❌ Fehler beim Database-Setup: {e}")
        raise


async def test_similarity_search(
    qdrant_handler: PlantDiseaseQdrantHandler, dataset_path: Path
):
    """Testet die Ähnlichkeitssuche mit einem Beispielbild"""

    logger.info("🔍 Teste Ähnlichkeitssuche...")

    try:
        # Erstes verfügbares Bild als Test verwenden
        test_image = None
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.jpg"):
                    test_image = img_path
                    break
                if test_image:
                    break

        if test_image:
            logger.info(f"📷 Test mit Bild: {test_image}")

            similar_cases = qdrant_handler.search_similar_images(
                query_image_path=str(test_image), limit=3, min_similarity=0.0
            )

            logger.info(f"✅ Gefunden: {len(similar_cases)} ähnliche Bilder")
            for i, case in enumerate(similar_cases):
                logger.info(
                    f"  {i + 1}. {case.disease_label} (Ähnlichkeit: {case.similarity_score:.3f})"
                )
        else:
            logger.warning("⚠️ Kein Test-Bild gefunden")

    except Exception as e:
        logger.error(f"❌ Fehler beim Test der Ähnlichkeitssuche: {e}")


def check_qdrant_connection():
    """Prüft Qdrant-Verbindung"""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="localhost", port=6333)

        # Versuche Collections abzurufen
        collections = client.get_collections()
        logger.info(
            f"✅ Qdrant verbunden. Gefundene Collections: {len(collections.collections)}"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Qdrant-Verbindung fehlgeschlagen: {e}")
        logger.error(
            "💡 Tipp: Starte Qdrant mit: docker run -p 6333:6333 qdrant/qdrant"
        )
        return False


def create_sample_config():
    """Erstellt eine Beispiel-Konfigurationsdatei"""

    config = {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "plant_disease_embeddings",
        },
        "dataset": {
            "path": "data/PlantDoc",
            "train_split": "train",
            "test_split": "test",
        },
        "embeddings": {"model": "openai/clip-vit-base-patch32", "batch_size": 25},
    }

    config_path = project_root / "config" / "qdrant_setup.json"
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"📄 Beispiel-Konfiguration erstellt: {config_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Qdrant-Database Setup für Plant Disease Detection"
    )

    parser.add_argument(
        "--dataset-path",
        default="data/PlantDoc",
        help="Pfad zum Dataset (default: data/PlantDoc)",
    )

    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Collection neu erstellen (überschreibt existierende Daten)",
    )

    parser.add_argument(
        "--check-connection", action="store_true", help="Nur Qdrant-Verbindung testen"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Beispiel-Konfigurationsdatei erstellen",
    )

    args = parser.parse_args()

    # Arbeitsverzeichnis zum Projektroot wechseln
    os.chdir(project_root)

    if args.create_config:
        create_sample_config()
        return

    if args.check_connection:
        if check_qdrant_connection():
            logger.info("✅ Qdrant-Verbindung erfolgreich")
        else:
            logger.error("❌ Qdrant-Verbindung fehlgeschlagen")
        return

    # Prüfe Qdrant-Verbindung
    if not check_qdrant_connection():
        logger.error("❌ Qdrant nicht erreichbar. Setup abgebrochen.")
        return

    # Prüfe Dataset-Pfad
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"❌ Dataset-Pfad nicht gefunden: {dataset_path}")
        logger.error(
            "💡 Tipp: Lade das PlantDoc-Dataset herunter und platziere es in data/PlantDoc/"
        )
        return

    logger.info(f"📁 Dataset-Pfad: {dataset_path.resolve()}")

    # Database-Setup durchführen
    await setup_qdrant_database(str(dataset_path), args.force_recreate)


if __name__ == "__main__":
    # Überprüfe erforderliche Umgebungsvariablen
    logger.info("🔧 Starte Qdrant-Database Setup...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Setup durch Benutzer abgebrochen")
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)
