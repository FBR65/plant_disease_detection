"""
Modellarchitekturen für die Pflanzenkrankheitserkennung
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import yaml
from pathlib import Path


class PlantDiseaseClassifier:
    """Klassifikationsmodell für Pflanzenkrankheiten"""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialisiert das Modell

        Args:
            config_path: Pfad zur Modellkonfiguration
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.history = None

    def _load_config(self, config_path: str) -> dict:
        """Lädt die Modellkonfiguration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback-Konfiguration
            return {
                "model_architecture": "ResNet50",
                "input_shape": [224, 224, 3],
                "num_classes": 3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "dropout_rate": 0.5,
            }

    def create_model(self, architecture: str = None) -> tf.keras.Model:
        """
        Erstellt das Modell basierend auf der gewählten Architektur

        Args:
            architecture: Modellarchitektur ('ResNet50', 'EfficientNetB0', 'VGG16')

        Returns:
            Tensorflow/Keras Modell
        """
        if architecture is None:
            architecture = self.config["model_architecture"]

        input_shape = tuple(self.config["input_shape"])
        num_classes = self.config["num_classes"]
        dropout_rate = self.config["dropout_rate"]

        # Basis-Modell auswählen
        if architecture == "ResNet50":
            base_model = ResNet50(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        elif architecture == "EfficientNetB0":
            base_model = EfficientNetB0(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        elif architecture == "VGG16":
            base_model = VGG16(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        else:
            raise ValueError(f"Unbekannte Architektur: {architecture}")

        # Basis-Modell zunächst einfrieren
        base_model.trainable = False

        # Klassifikationsschichten hinzufügen
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(512, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(256, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Modell kompilieren
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss="categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

        self.model = model
        return model

    def create_callbacks(
        self, model_save_path: str = "models/classification_model/best_model.h5"
    ) -> list:
        """
        Erstellt Callbacks für das Training

        Args:
            model_save_path: Pfad zum Speichern des besten Modells

        Returns:
            Liste von Keras Callbacks
        """
        callbacks = [
            ModelCheckpoint(
                filepath=model_save_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                mode="max",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1
            ),
        ]

        return callbacks

    def train(
        self, train_generator, val_generator, epochs: int = None
    ) -> tf.keras.callbacks.History:
        """
        Trainiert das Modell

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Anzahl Epochen (falls None, aus Config laden)

        Returns:
            Training history
        """
        if self.model is None:
            self.create_model()

        if epochs is None:
            epochs = self.config["epochs"]

        callbacks = self.create_callbacks()

        # Training starten
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        self.history = history
        return history

    def fine_tune(self, train_generator, val_generator, unfreeze_layers: int = 50):
        """
        Fine-tuning des vortrainierten Modells

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            unfreeze_layers: Anzahl der Schichten, die aufgetaut werden sollen
        """
        if self.model is None:
            raise ValueError("Modell muss zuerst erstellt und trainiert werden")

        # Letzte Schichten des Basis-Modells auftauen
        base_model = self.model.layers[0]
        base_model.trainable = True

        # Nur die letzten N Schichten trainierbar machen
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        # Mit niedrigerer Lernrate neu kompilieren
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["learning_rate"] / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

        # Fine-tuning mit weniger Epochen
        fine_tune_epochs = 10
        callbacks = self.create_callbacks(
            "models/classification_model/fine_tuned_model.h5"
        )

        history_fine = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        return history_fine

    def predict(self, image: tf.Tensor) -> dict:
        """
        Vorhersage für ein einzelnes Bild

        Args:
            image: Vorverarbeitetes Bild (Batch-Dimension erforderlich)

        Returns:
            Dictionary mit Vorhersage und Konfidenz
        """
        if self.model is None:
            raise ValueError("Modell muss zuerst geladen werden")

        predictions = self.model.predict(image)
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = tf.reduce_max(predictions[0]).numpy()

        class_names = ["healthy", "disease_A", "disease_B"]

        return {
            "predicted_class": class_names[predicted_class],
            "confidence": float(confidence),
            "class_probabilities": {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            },
        }

    def load_model(self, model_path: str):
        """
        Lädt ein gespeichertes Modell

        Args:
            model_path: Pfad zum gespeicherten Modell
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Modell geladen von: {model_path}")

    def save_model(self, save_path: str):
        """
        Speichert das aktuelle Modell

        Args:
            save_path: Pfad zum Speichern des Modells
        """
        if self.model is None:
            raise ValueError("Kein Modell zum Speichern vorhanden")

        # Verzeichnis erstellen falls nicht vorhanden
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        self.model.save(save_path)
        print(f"Modell gespeichert in: {save_path}")


def create_simple_cnn_model(
    input_shape: tuple = (224, 224, 3), num_classes: int = 3
) -> tf.keras.Model:
    """
    Erstellt ein einfaches CNN-Modell von Grund auf

    Args:
        input_shape: Eingabeform der Bilder
        num_classes: Anzahl der Klassen

    Returns:
        Einfaches CNN-Modell
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Beispiel für die Verwendung
    classifier = PlantDiseaseClassifier()
    model = classifier.create_model()

    print("Modell erstellt:")
    model.summary()
