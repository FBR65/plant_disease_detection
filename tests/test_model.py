"""
Tests für das Modell
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml

import sys

sys.path.append("../src")
from model import PlantDiseaseClassifier, create_simple_cnn_model


class TestPlantDiseaseClassifier(unittest.TestCase):
    """Test-Klasse für PlantDiseaseClassifier"""

    def setUp(self):
        """Setup vor jedem Test"""
        # Temporäres Verzeichnis erstellen
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Test-Konfiguration erstellen
        self.config = {
            "model_architecture": "ResNet50",
            "input_shape": [224, 224, 3],
            "num_classes": 3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2,  # Wenige Epochen für Tests
            "dropout_rate": 0.5,
        }

        # Temporäre Config-Datei erstellen
        self.config_path = self.temp_path / "test_model_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        """Cleanup nach jedem Test"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test der Klassifikator-Initialisierung"""
        classifier = PlantDiseaseClassifier(str(self.config_path))

        self.assertEqual(classifier.config["model_architecture"], "ResNet50")
        self.assertEqual(classifier.config["num_classes"], 3)
        self.assertIsNone(classifier.model)

    def test_initialization_with_fallback_config(self):
        """Test mit fehlender Konfigurationsdatei (Fallback)"""
        classifier = PlantDiseaseClassifier("nonexistent_config.yaml")

        # Fallback-Konfiguration sollte verwendet werden
        self.assertEqual(classifier.config["model_architecture"], "ResNet50")
        self.assertEqual(classifier.config["num_classes"], 3)

    def test_create_model_resnet50(self):
        """Test der Modellgenerierung mit ResNet50"""
        classifier = PlantDiseaseClassifier(str(self.config_path))
        model = classifier.create_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(classifier.model)

        # Input-Shape prüfen
        expected_input_shape = (None, 224, 224, 3)
        self.assertEqual(model.input_shape, expected_input_shape)

        # Output-Shape prüfen
        expected_output_shape = (None, 3)
        self.assertEqual(model.output_shape, expected_output_shape)

    def test_create_model_efficientnet(self):
        """Test der Modellgenerierung mit EfficientNet"""
        classifier = PlantDiseaseClassifier(str(self.config_path))
        model = classifier.create_model(architecture="EfficientNetB0")

        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 3))

    def test_create_model_vgg16(self):
        """Test der Modellgenerierung mit VGG16"""
        classifier = PlantDiseaseClassifier(str(self.config_path))
        model = classifier.create_model(architecture="VGG16")

        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 3))

    def test_create_model_unknown_architecture(self):
        """Test mit unbekannter Architektur"""
        classifier = PlantDiseaseClassifier(str(self.config_path))

        with self.assertRaises(ValueError):
            classifier.create_model(architecture="UnknownArchitecture")

    def test_create_callbacks(self):
        """Test der Callback-Erstellung"""
        classifier = PlantDiseaseClassifier(str(self.config_path))

        save_path = self.temp_path / "test_model.h5"
        callbacks = classifier.create_callbacks(str(save_path))

        self.assertIsInstance(callbacks, list)
        self.assertEqual(
            len(callbacks), 3
        )  # ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

        # Callback-Typen prüfen
        callback_types = [type(cb).__name__ for cb in callbacks]
        self.assertIn("ModelCheckpoint", callback_types)
        self.assertIn("EarlyStopping", callback_types)
        self.assertIn("ReduceLROnPlateau", callback_types)

    def test_predict_without_model(self):
        """Test der Vorhersage ohne geladenes Modell"""
        classifier = PlantDiseaseClassifier(str(self.config_path))

        # Dummy-Eingabe
        dummy_input = np.random.rand(1, 224, 224, 3)

        with self.assertRaises(ValueError):
            classifier.predict(dummy_input)

    def test_predict_with_model(self):
        """Test der Vorhersage mit Modell"""
        classifier = PlantDiseaseClassifier(str(self.config_path))
        classifier.create_model()

        # Dummy-Eingabe
        dummy_input = np.random.rand(1, 224, 224, 3)

        result = classifier.predict(dummy_input)

        self.assertIsInstance(result, dict)
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("class_probabilities", result)

        # Klassen prüfen
        expected_classes = ["healthy", "disease_A", "disease_B"]
        self.assertIn(result["predicted_class"], expected_classes)

        # Konfidenz prüfen
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Klassenwahrscheinlichkeiten prüfen
        probs = result["class_probabilities"]
        self.assertEqual(len(probs), 3)

        # Wahrscheinlichkeiten sollten sich zu 1 summieren
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_save_and_load_model(self):
        """Test des Modellspeicherns und -ladens"""
        classifier = PlantDiseaseClassifier(str(self.config_path))
        classifier.create_model()

        # Modell speichern
        save_path = self.temp_path / "test_model.h5"
        classifier.save_model(str(save_path))

        # Datei sollte existieren
        self.assertTrue(save_path.exists())

        # Neuen Klassifikator erstellen und Modell laden
        new_classifier = PlantDiseaseClassifier(str(self.config_path))
        new_classifier.load_model(str(save_path))

        self.assertIsNotNone(new_classifier.model)

        # Beide Modelle sollten gleiche Vorhersagen machen
        dummy_input = np.random.rand(1, 224, 224, 3)

        result1 = classifier.predict(dummy_input)
        result2 = new_classifier.predict(dummy_input)

        self.assertEqual(result1["predicted_class"], result2["predicted_class"])
        self.assertAlmostEqual(result1["confidence"], result2["confidence"], places=5)

    def test_save_model_without_model(self):
        """Test des Speicherns ohne Modell"""
        classifier = PlantDiseaseClassifier(str(self.config_path))

        with self.assertRaises(ValueError):
            classifier.save_model("test_model.h5")


class TestCreateSimpleCnnModel(unittest.TestCase):
    """Test-Klasse für create_simple_cnn_model Funktion"""

    def test_create_simple_cnn_default_params(self):
        """Test der einfachen CNN-Erstellung mit Standardparametern"""
        model = create_simple_cnn_model()

        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 3))

        # Modell sollte kompiliert sein
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

    def test_create_simple_cnn_custom_params(self):
        """Test der einfachen CNN-Erstellung mit benutzerdefinierten Parametern"""
        input_shape = (128, 128, 3)
        num_classes = 5

        model = create_simple_cnn_model(input_shape, num_classes)

        self.assertEqual(model.input_shape, (None, 128, 128, 3))
        self.assertEqual(model.output_shape, (None, 5))

    def test_simple_cnn_predict(self):
        """Test der Vorhersage mit einfachem CNN"""
        model = create_simple_cnn_model()

        # Dummy-Eingabe
        dummy_input = np.random.rand(1, 224, 224, 3)

        # Vorhersage sollte funktionieren
        prediction = model.predict(dummy_input, verbose=0)

        self.assertEqual(prediction.shape, (1, 3))

        # Wahrscheinlichkeiten sollten sich zu 1 summieren
        total_prob = np.sum(prediction[0])
        self.assertAlmostEqual(total_prob, 1.0, places=5)


if __name__ == "__main__":
    # Test-Suite ausführen
    unittest.main(verbosity=2)
