"""
Datenloader und Vorverarbeitungsklassen für die Pflanzenkrankheitserkennung
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml


class PlantDiseaseDataLoader:
    """Klasse zum Laden und Vorverarbeiten der Pflanzenkrankheitsdaten"""

    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        """
        Initialisiert den DataLoader

        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.class_names = ["healthy", "disease_A", "disease_B"]

    def _load_config(self) -> dict:
        """Lädt die Konfiguration aus der YAML-Datei"""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback-Konfiguration
            return {
                "data_path": "data/raw",
                "image_size": [224, 224],
                "batch_size": 32,
                "validation_split": 0.2,
            }

    def load_image(
        self, image_path: str, target_size: tuple = (224, 224)
    ) -> np.ndarray:
        """
        Lädt und preprocessed ein einzelnes Bild

        Args:
            image_path: Pfad zum Bild
            target_size: Zielgröße des Bildes

        Returns:
            Vorverarbeitetes Bild als numpy array
        """
        try:
            # Bild laden mit PIL
            image = Image.open(image_path)

            # Zu RGB konvertieren falls nötig
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Größe ändern
            image = image.resize(target_size)

            # Zu numpy array konvertieren und normalisieren
            image_array = np.array(image) / 255.0

            return image_array

        except Exception as e:
            print(f"Fehler beim Laden des Bildes {image_path}: {e}")
            return None

    def create_data_generators(self) -> tuple:
        """
        Erstellt Datengeratoren für Training und Validierung

        Returns:
            Tuple von (train_generator, val_generator)
        """
        # Datenaugmentation für Training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="nearest",
        )

        # Nur Normalisierung für Validierung
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Pfade definieren
        train_path = Path(self.config["data_path"]) / "train"
        val_path = Path(self.config["data_path"]) / "val"

        # Generatoren erstellen
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=tuple(self.config["image_size"]),
            batch_size=self.config["batch_size"],
            class_mode="categorical",
            classes=self.class_names,
        )

        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=tuple(self.config["image_size"]),
            batch_size=self.config["batch_size"],
            class_mode="categorical",
            classes=self.class_names,
        )

        return train_generator, val_generator

    def get_dataset_statistics(self) -> dict:
        """
        Berechnet Statistiken über den Datensatz

        Returns:
            Dictionary mit Statistiken
        """
        stats = {}
        data_path = Path(self.config["data_path"])

        for split in ["train", "val"]:
            split_stats = {}
            for class_name in self.class_names:
                class_path = data_path / split / class_name
                if class_path.exists():
                    # Anzahl Bilder zählen
                    image_count = len(
                        [
                            f
                            for f in class_path.iterdir()
                            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                        ]
                    )
                    split_stats[class_name] = image_count
                else:
                    split_stats[class_name] = 0
            stats[split] = split_stats

        return stats

    def create_metadata_csv(self, output_path: str = "data/processed/metadata.csv"):
        """
        Erstellt eine CSV-Datei mit Metadaten aller Bilder

        Args:
            output_path: Pfad für die Ausgabe-CSV
        """
        metadata = []
        data_path = Path(self.config["data_path"])

        for split in ["train", "val"]:
            for class_name in self.class_names:
                class_path = data_path / split / class_name
                if class_path.exists():
                    for image_file in class_path.glob("*.jpg"):
                        metadata.append(
                            {
                                "image_path": str(image_file),
                                "label": class_name,
                                "dataset_split": split,
                                "filename": image_file.name,
                                "relative_path": str(
                                    image_file.relative_to(data_path.parent)
                                ),
                            }
                        )

        # DataFrame erstellen und speichern
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        print(f"Metadaten gespeichert in: {output_path}")
        print(f"Anzahl Bilder: {len(df)}")

        return df


def preprocess_image_for_model(
    image_path: str, target_size: tuple = (224, 224)
) -> np.ndarray:
    """
    Standalone-Funktion zur Bildvorverarbeitung für Modellvorhersagen

    Args:
        image_path: Pfad zum Bild
        target_size: Zielgröße des Bildes

    Returns:
        Vorverarbeitetes Bild bereit für Modellvorhersage
    """
    loader = PlantDiseaseDataLoader()
    image = loader.load_image(image_path, target_size)

    if image is not None:
        # Batch-Dimension hinzufügen
        return np.expand_dims(image, axis=0)
    else:
        return None


if __name__ == "__main__":
    # Beispiel für die Verwendung
    loader = PlantDiseaseDataLoader()

    # Statistiken anzeigen
    stats = loader.get_dataset_statistics()
    print("Datensatz-Statistiken:")
    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        for class_name, count in split_stats.items():
            print(f"  {class_name}: {count} Bilder")

    # Metadaten erstellen
    # loader.create_metadata_csv()
