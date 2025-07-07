"""
Trainingsscript für das Pflanzenkrankheits-Klassifikationsmodell
"""

import argparse
import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

# Eigene Module importieren
from data_loader import PlantDiseaseDataLoader
from model import PlantDiseaseClassifier


def plot_training_history(
    history, save_path: str = "models/classification_model/training_history.png"
):
    """
    Plottet die Trainingsgeschichte

    Args:
        history: Keras training history
        save_path: Pfad zum Speichern des Plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0, 0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0, 0].set_title("Model Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history["loss"], label="Training Loss")
    axes[0, 1].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 1].set_title("Model Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Top-K Accuracy (falls vorhanden)
    if "top_k_categorical_accuracy" in history.history:
        axes[1, 0].plot(
            history.history["top_k_categorical_accuracy"],
            label="Training Top-K Accuracy",
        )
        axes[1, 0].plot(
            history.history["val_top_k_categorical_accuracy"],
            label="Validation Top-K Accuracy",
        )
        axes[1, 0].set_title("Top-K Categorical Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Top-K Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Learning Rate (falls verfügbar)
    if "lr" in history.history:
        axes[1, 1].plot(history.history["lr"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()

    # Verzeichnis erstellen falls nicht vorhanden
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Trainingshistorie gespeichert in: {save_path}")


def train_model(
    config_path: str = "config/model_config.yaml",
    data_config_path: str = "config/dataset_config.yaml",
    fine_tune: bool = False,
):
    """
    Hauptfunktion für das Modelltraining

    Args:
        config_path: Pfad zur Modellkonfiguration
        data_config_path: Pfad zur Datenkonfiguration
        fine_tune: Ob Fine-tuning durchgeführt werden soll
    """
    print("🌱 Starte Training des Pflanzenkrankheits-Klassifikationsmodells...")

    # Data Loader initialisieren
    print("📊 Lade Daten...")
    data_loader = PlantDiseaseDataLoader(data_config_path)

    # Datensatz-Statistiken anzeigen
    stats = data_loader.get_dataset_statistics()
    print("\n📈 Datensatz-Statistiken:")
    total_images = 0
    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        split_total = 0
        for class_name, count in split_stats.items():
            print(f"  {class_name}: {count} Bilder")
            split_total += count
        print(f"  Gesamt {split}: {split_total} Bilder")
        total_images += split_total
    print(f"\nGesamtanzahl Bilder: {total_images}")

    # Datengeratoren erstellen
    print("\n🔄 Erstelle Datengeratoren...")
    train_generator, val_generator = data_loader.create_data_generators()

    print(f"Training Batches: {len(train_generator)}")
    print(f"Validation Batches: {len(val_generator)}")
    print(f"Klassennamen: {train_generator.class_indices}")

    # Modell erstellen
    print("\n🏗️ Erstelle Modell...")
    classifier = PlantDiseaseClassifier(config_path)
    model = classifier.create_model()

    print("\n📋 Modellarchitektur:")
    model.summary()

    # Training starten
    print("\n🚀 Starte Training...")
    history = classifier.train(train_generator, val_generator)

    # Trainingshistorie plotten
    print("\n📊 Erstelle Trainingsplots...")
    plot_training_history(history)

    # Fine-tuning (optional)
    if fine_tune:
        print("\n🎯 Starte Fine-tuning...")
        history_fine = classifier.fine_tune(train_generator, val_generator)

        # Fine-tuning Historie plotten
        plot_training_history(
            history_fine, "models/classification_model/fine_tuning_history.png"
        )

    # Finales Modell speichern
    classifier.save_model("models/classification_model/final_model.h5")

    # Evaluierung auf Validierungsdaten
    print("\n📊 Evaluiere Modell auf Validierungsdaten...")
    val_loss, val_accuracy, val_top_k = model.evaluate(val_generator, verbose=1)

    print("\n✅ Training abgeschlossen!")
    print(f"📊 Finale Validation Accuracy: {val_accuracy:.4f}")
    print(f"📊 Finale Validation Loss: {val_loss:.4f}")
    print(f"📊 Finale Top-K Accuracy: {val_top_k:.4f}")

    # Ergebnisse in Datei speichern
    results = {
        "final_val_accuracy": float(val_accuracy),
        "final_val_loss": float(val_loss),
        "final_top_k_accuracy": float(val_top_k),
        "total_epochs": len(history.history["accuracy"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "dataset_stats": stats,
    }

    results_path = "models/classification_model/training_results.yaml"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"📄 Trainingsergebnisse gespeichert in: {results_path}")

    return classifier, history


def main():
    """Hauptfunktion mit Kommandozeilen-Interface"""
    parser = argparse.ArgumentParser(
        description="Training des Pflanzenkrankheits-Klassifikationsmodells"
    )

    parser.add_argument(
        "--model-config",
        default="config/model_config.yaml",
        help="Pfad zur Modellkonfiguration",
    )

    parser.add_argument(
        "--data-config",
        default="config/dataset_config.yaml",
        help="Pfad zur Datenkonfiguration",
    )

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tuning nach dem initialen Training durchführen",
    )

    parser.add_argument(
        "--architecture",
        choices=["ResNet50", "EfficientNetB0", "VGG16"],
        help="Modellarchitektur (überschreibt Config)",
    )

    args = parser.parse_args()

    # Konfiguration ggf. anpassen
    if args.architecture:
        print(f"🏗️ Verwende Architektur: {args.architecture}")
        # Config laden und anpassen
        with open(args.model_config, "r") as f:
            config = yaml.safe_load(f)
        config["model_architecture"] = args.architecture

        # Temporäre Config-Datei erstellen
        temp_config_path = "config/temp_model_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
        args.model_config = temp_config_path

    try:
        # Training starten
        classifier, history = train_model(
            config_path=args.model_config,
            data_config_path=args.data_config,
            fine_tune=args.fine_tune,
        )

        print("\n🎉 Training erfolgreich abgeschlossen!")

    except Exception as e:
        print(f"\n❌ Fehler beim Training: {e}")
        sys.exit(1)

    finally:
        # Temporäre Config-Datei aufräumen
        if args.architecture and Path("config/temp_model_config.yaml").exists():
            Path("config/temp_model_config.yaml").unlink()


if __name__ == "__main__":
    main()
