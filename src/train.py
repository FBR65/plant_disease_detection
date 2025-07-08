"""
Trainingsscript für das Pflanzenkrankheits-Klassifikationsmodell
"""

import argparse
import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import json
import os

# Eigene Module importieren
try:
    from .advanced_augmentation import create_balanced_dataloader
except ImportError:
    # Fallback für direkten Aufruf
    from advanced_augmentation import create_balanced_dataloader


class PlantDiseaseClassifier(nn.Module):
    """PyTorch-basierter Klassifikator für Pflanzenkrankheiten"""

    def __init__(self, num_classes, architecture="resnet50", pretrained=True):
        super(PlantDiseaseClassifier, self).__init__()

        if architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier[1] = nn.Linear(
                self.backbone.classifier[1].in_features, num_classes
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x):
        return self.backbone(x)


def plot_training_history(history, save_path="reports/training_history.png"):
    """Plottet die Trainingsgeschichte"""
    # Wenn relativer Pfad, dann zum Projektroot hinzufügen
    if not os.path.isabs(save_path):
        project_root = Path(__file__).parent.parent
        save_path = project_root / save_path

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Verlust
    ax1.plot(history["train_loss"], label="Training Loss", marker="o")
    ax1.plot(history["val_loss"], label="Validation Loss", marker="s")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Genauigkeit
    ax2.plot(history["train_acc"], label="Training Accuracy", marker="o")
    ax2.plot(history["val_acc"], label="Validation Accuracy", marker="s")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Trainingshistorie gespeichert in: {save_path}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trainiert das Modell für eine Epoche"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            {
                "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "Acc": f"{100.0 * correct / total:.2f}%",
            }
        )

    return running_loss / len(dataloader), 100.0 * correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validiert das Modell für eine Epoche"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {
                    "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    return running_loss / len(dataloader), 100.0 * correct / total


def train_model(
    data_path="data/PlantDoc",
    epochs=25,
    batch_size=32,
    learning_rate=0.001,
    architecture="resnet50",
    save_path="models/classification_model",
):
    """
    Hauptfunktion für das Modelltraining mit PyTorch
    """
    print("🌱 Starte PyTorch Training des Pflanzenkrankheits-Klassifikationsmodells...")

    # Arbeitsverzeichnis ermitteln (Projektroot)
    project_root = Path(__file__).parent.parent

    # Absolute Pfade erstellen
    if not os.path.isabs(data_path):
        data_path = project_root / data_path
    if not os.path.isabs(save_path):
        save_path = project_root / save_path

    print(f"📁 Datenverzeichnis: {data_path}")
    print(f"💾 Speicherverzeichnis: {save_path}")

    # Gerät auswählen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Verwende Gerät: {device}")

    # GPU-Informationen anzeigen
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(f"⚡ CUDA Version: {torch.version.cuda}")
        
        # GPU-Memory-Optimierungen
        torch.cuda.empty_cache()  # Cache leeren
        torch.backends.cudnn.benchmark = True  # Optimiere für feste Input-Größen
        
        # Batch-Size-Empfehlung basierend auf GPU-Memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if batch_size < 16 and gpu_memory_gb >= 6:
            print(f"💡 Tipp: Mit {gpu_memory_gb:.1f}GB GPU-Memory könnten Sie batch_size={min(32, batch_size * 4)} verwenden")
    else:
        print("⚠️  Keine GPU verfügbar - verwende CPU")

    # Datenlader erstellen
    print("📊 Lade Daten...")
    train_loader, train_dataset = create_balanced_dataloader(
        data_dir=data_path,
        split="train",
        batch_size=batch_size,
        augment_minority_classes=True,
        synthetic_factor=3,
        use_weighted_sampling=True,
    )

    val_loader, val_dataset = create_balanced_dataloader(
        data_dir=data_path,
        split="test",  # Verwende test als validation
        batch_size=batch_size,
        augment_minority_classes=False,
        use_weighted_sampling=False,
    )

    num_classes = len(train_dataset.classes)
    print(f"📊 Gefundene Klassen: {num_classes}")
    print(f"📊 Trainingssamples: {len(train_dataset)}")
    print(f"📊 Validierungssamples: {len(val_dataset)}")

    # Modell erstellen
    print(f"🏗️ Erstelle Modell ({architecture})...")
    model = PlantDiseaseClassifier(num_classes=num_classes, architecture=architecture)
    model = model.to(device)

    # Verlustfunktion und Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    # Training
    print("🚀 Starte Training...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n📅 Epoche {epoch + 1}/{epochs}")
        print("-" * 50)

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validierung
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step(val_loss)

        # History speichern
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Training - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Bestes Modell speichern
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✅ Neues bestes Modell! Validierungsgenauigkeit: {val_acc:.2f}%")

            # Modell speichern
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "model.pth")

            # Klassen speichern
            with open(save_path / "classes.json", "w") as f:
                json.dump(train_dataset.classes, f, indent=2)

            print(f"💾 Modell gespeichert in: {save_path / 'model.pth'}")

    # Trainingshistorie plotten
    plot_training_history(history)

    print(f"\n🎉 Training abgeschlossen!")
    print(f"🏆 Beste Validierungsgenauigkeit: {best_val_acc:.2f}%")

    return model, history


def main():
    """Hauptfunktion mit Kommandozeilen-Interface"""
    parser = argparse.ArgumentParser(
        description="Training des Pflanzenkrankheits-Klassifikationsmodells"
    )

    parser.add_argument(
        "--data-path", default="data/PlantDoc", help="Pfad zum Datensatz"
    )

    parser.add_argument(
        "--epochs", type=int, default=25, help="Anzahl der Trainings-Epochen"
    )

    parser.add_argument("--batch-size", type=int, default=32, help="Batch-Größe")

    parser.add_argument("--learning-rate", type=float, default=0.001, help="Lernrate")

    parser.add_argument(
        "--architecture",
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Modellarchitektur",
    )

    parser.add_argument(
        "--save-path",
        default="models/classification_model",
        help="Pfad zum Speichern des Modells",
    )

    args = parser.parse_args()

    # Training starten
    model, history = train_model(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        save_path=args.save_path,
    )

    print("✅ Training erfolgreich abgeschlossen!")


if __name__ == "__main__":
    main()
