"""
Tests für den DataLoader
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

import sys

sys.path.append("../src")
from data_loader import PlantDiseaseDataLoader, preprocess_image_for_model


class TestPlantDiseaseDataLoader(unittest.TestCase):
    """Test-Klasse für PlantDiseaseDataLoader"""

    def setUp(self):
        """Setup vor jedem Test"""
        # Temporäres Verzeichnis erstellen
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Test-Datenstruktur erstellen
        self.create_test_data_structure()

        # Test-Konfiguration erstellen
        self.config = {
            "data_path": str(self.temp_path / "data" / "raw"),
            "image_size": [224, 224],
            "batch_size": 2,
            "validation_split": 0.2,
        }

        # Temporäre Config-Datei erstellen
        self.config_path = self.temp_path / "test_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        """Cleanup nach jedem Test"""
        shutil.rmtree(self.temp_dir)

    def create_test_data_structure(self):
        """Erstellt eine Test-Datenstruktur mit dummy Bildern"""
        # Verzeichnisse erstellen
        for split in ["train", "val"]:
            for category in ["healthy", "disease_A", "disease_B"]:
                category_path = self.temp_path / "data" / "raw" / split / category
                category_path.mkdir(parents=True, exist_ok=True)

                # Dummy-Bilder erstellen
                for i in range(3):
                    # RGB-Bild erstellen
                    dummy_image = Image.new("RGB", (224, 224), color=(i * 50, 100, 150))
                    dummy_image.save(category_path / f"test_image_{i}.jpg")

    def test_initialization(self):
        """Test der DataLoader-Initialisierung"""
        loader = PlantDiseaseDataLoader(str(self.config_path))

        self.assertEqual(loader.config["data_path"], self.config["data_path"])
        self.assertEqual(loader.config["batch_size"], self.config["batch_size"])
        self.assertEqual(len(loader.class_names), 3)

    def test_load_image(self):
        """Test des Bildladens"""
        loader = PlantDiseaseDataLoader(str(self.config_path))

        # Test-Bild-Pfad
        test_image_path = (
            self.temp_path / "data" / "raw" / "train" / "healthy" / "test_image_0.jpg"
        )

        # Bild laden
        image_array = loader.load_image(str(test_image_path))

        self.assertIsNotNone(image_array)
        self.assertEqual(image_array.shape, (224, 224, 3))
        self.assertTrue(
            np.all(image_array >= 0) and np.all(image_array <= 1)
        )  # Normalisierung prüfen

    def test_load_nonexistent_image(self):
        """Test mit nicht-existierendem Bild"""
        loader = PlantDiseaseDataLoader(str(self.config_path))

        image_array = loader.load_image("nonexistent.jpg")
        self.assertIsNone(image_array)

    def test_create_data_generators(self):
        """Test der Datengerator-Erstellung"""
        loader = PlantDiseaseDataLoader(str(self.config_path))

        train_gen, val_gen = loader.create_data_generators()

        # Generatoren sollten erstellt werden
        self.assertIsNotNone(train_gen)
        self.assertIsNotNone(val_gen)

        # Batch-Größe prüfen
        self.assertEqual(train_gen.batch_size, self.config["batch_size"])
        self.assertEqual(val_gen.batch_size, self.config["batch_size"])

        # Klassennamen prüfen
        expected_classes = {"disease_A": 0, "disease_B": 1, "healthy": 2}
        self.assertEqual(train_gen.class_indices, expected_classes)

    def test_get_dataset_statistics(self):
        """Test der Datensatz-Statistiken"""
        loader = PlantDiseaseDataLoader(str(self.config_path))

        stats = loader.get_dataset_statistics()

        # Strukturprüfung
        self.assertIn("train", stats)
        self.assertIn("val", stats)

        # Anzahl Bilder prüfen (3 pro Kategorie)
        for split in ["train", "val"]:
            for category in ["healthy", "disease_A", "disease_B"]:
                self.assertEqual(stats[split][category], 3)

    def test_create_metadata_csv(self):
        """Test der Metadaten-CSV-Erstellung"""
        loader = PlantDiseaseDataLoader(str(self.config_path))

        output_path = self.temp_path / "metadata.csv"
        df = loader.create_metadata_csv(str(output_path))

        # CSV-Datei sollte existieren
        self.assertTrue(output_path.exists())

        # DataFrame sollte die richtige Anzahl Zeilen haben
        expected_rows = 2 * 3 * 3  # 2 splits * 3 categories * 3 images
        self.assertEqual(len(df), expected_rows)

        # Spalten prüfen
        expected_columns = [
            "image_path",
            "label",
            "dataset_split",
            "filename",
            "relative_path",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)


class TestPreprocessImageForModel(unittest.TestCase):
    """Test-Klasse für preprocess_image_for_model Funktion"""

    def setUp(self):
        """Setup vor jedem Test"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Test-Bild erstellen
        self.test_image_path = self.temp_path / "test_image.jpg"
        test_image = Image.new("RGB", (300, 200), color=(100, 150, 200))
        test_image.save(self.test_image_path)

    def tearDown(self):
        """Cleanup nach jedem Test"""
        shutil.rmtree(self.temp_dir)

    def test_preprocess_valid_image(self):
        """Test der Bildvorverarbeitung mit gültigem Bild"""
        processed = preprocess_image_for_model(str(self.test_image_path))

        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (1, 224, 224, 3))  # Batch-Dimension
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))

    def test_preprocess_nonexistent_image(self):
        """Test mit nicht-existierendem Bild"""
        processed = preprocess_image_for_model("nonexistent.jpg")
        self.assertIsNone(processed)

    def test_preprocess_custom_target_size(self):
        """Test mit benutzerdefinierter Zielgröße"""
        target_size = (128, 128)
        processed = preprocess_image_for_model(str(self.test_image_path), target_size)

        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (1, 128, 128, 3))


if __name__ == "__main__":
    # Test-Suite ausführen
    unittest.main(verbosity=2)
