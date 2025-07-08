"""
Datenexploration für Pflanzenkrankheitserkennung
"""

import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def explore_dataset_structure(base_path="data/PlantDoc/"):
    """Analysiert die Struktur des PlantDoc-Datasets"""

    results = {}

    for split in ["train", "test"]:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"Warnung: {split_path} existiert nicht!")
            continue

        print(f"\n=== {split.upper()} DATASET ===")

        image_paths_by_class = defaultdict(list)
        all_image_paths = []
        all_labels = []

        # Durch alle Klassen-Ordner iterieren
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                # Alle Bilder in diesem Klassen-Ordner finden
                for image_name in os.listdir(class_path):
                    if image_name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                    ):
                        image_path = os.path.join(class_path, image_name)
                        image_paths_by_class[class_name].append(image_path)
                        all_image_paths.append(image_path)
                        all_labels.append(class_name)

        # Statistiken für diesen Split
        print(f"Anzahl Klassen: {len(image_paths_by_class)}")
        print(f"Gesamtzahl Bilder: {len(all_image_paths)}")
        print("\nBilder pro Klasse:")

        for class_name, paths in sorted(image_paths_by_class.items()):
            print(f"  {class_name}: {len(paths)} Bilder")

        # DataFrame erstellen
        df = pd.DataFrame(
            {"image_path": all_image_paths, "label": all_labels, "split": split}
        )

        results[split] = {
            "df": df,
            "image_paths_by_class": image_paths_by_class,
            "class_counts": df["label"].value_counts(),
        }

    return results


def plot_class_distribution(dataset_info, save_path=None):
    """Visualisiert die Verteilung der Klassen in Train/Test"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, (split, info) in enumerate(dataset_info.items()):
        class_counts = info["class_counts"]

        # Balkendiagramm
        axes[i].bar(range(len(class_counts)), class_counts.values)
        axes[i].set_title(f"{split.upper()} - Klassenverteilung")
        axes[i].set_xlabel("Klassen")
        axes[i].set_ylabel("Anzahl Bilder")
        axes[i].set_xticks(range(len(class_counts)))
        axes[i].set_xticklabels(class_counts.index, rotation=45, ha="right")

        # Statistiken anzeigen
        print(f"\n{split.upper()} Statistiken:")
        print(f"  Durchschnitt: {class_counts.mean():.1f} Bilder/Klasse")
        print(f"  Median: {class_counts.median():.1f} Bilder/Klasse")
        print(f"  Min: {class_counts.min()} Bilder")
        print(f"  Max: {class_counts.max()} Bilder")

    plt.tight_layout()

    if save_path:
        # Verzeichnis erstellen falls nicht vorhanden
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

    plt.show()


def analyze_train_test_split(dataset_info):
    """Analysiert die Konsistenz zwischen Train und Test"""

    if "train" not in dataset_info or "test" not in dataset_info:
        print("Train oder Test-Daten fehlen!")
        return

    train_classes = set(dataset_info["train"]["class_counts"].index)
    test_classes = set(dataset_info["test"]["class_counts"].index)

    print("=== TRAIN/TEST KONSISTENZ ===")
    print(f"Klassen nur in Train: {train_classes - test_classes}")
    print(f"Klassen nur in Test: {test_classes - train_classes}")
    print(f"Gemeinsame Klassen: {len(train_classes & test_classes)}")

    # Verhältnis Train/Test pro Klasse
    print("\n=== TRAIN/TEST VERHÄLTNIS ===")
    common_classes = train_classes & test_classes

    ratios = []
    for class_name in sorted(common_classes):
        train_count = dataset_info["train"]["class_counts"][class_name]
        test_count = dataset_info["test"]["class_counts"][class_name]
        ratio = train_count / test_count if test_count > 0 else float("inf")
        ratios.append(ratio)
        print(
            f"{class_name}: {train_count} Train, {test_count} Test (Verhältnis: {ratio:.2f})"
        )

    if ratios:
        print(
            f"\nDurchschnittliches Train/Test-Verhältnis: {sum(ratios) / len(ratios):.2f}"
        )


def analyze_image_sizes(dataset_info, sample_size=100):
    """Analysiert die Bildgrößen im Dataset"""

    sizes = []

    for split, info in dataset_info.items():
        print(f"\n=== {split.upper()} BILDGRÖSSEN ===")

        sample_paths = (
            info["df"]["image_path"].sample(min(sample_size, len(info["df"]))).tolist()
        )

        widths, heights = [], []

        for img_path in sample_paths:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"Fehler bei {img_path}: {e}")

        if widths and heights:
            print(f"Durchschnittliche Breite: {np.mean(widths):.1f} px")
            print(f"Durchschnittliche Höhe: {np.mean(heights):.1f} px")
            print(f"Breite: {min(widths)} - {max(widths)} px")
            print(f"Höhe: {min(heights)} - {max(heights)} px")

            # Histogramm
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(widths, bins=20, alpha=0.7)
            plt.title(f"{split} - Breite")
            plt.xlabel("Pixel")

            plt.subplot(1, 2, 2)
            plt.hist(heights, bins=20, alpha=0.7)
            plt.title(f"{split} - Höhe")
            plt.xlabel("Pixel")

            plt.tight_layout()
            plt.show()


def analyze_class_imbalance(dataset_info):
    """Analysiert Klassenungleichgewicht und gibt Empfehlungen"""

    if "train" not in dataset_info:
        return

    train_counts = dataset_info["train"]["class_counts"]

    print("\n=== KLASSENUNGLEICHGEWICHT ANALYSE ===")

    max_count = train_counts.max()
    min_count = train_counts.min()
    median_count = train_counts.median()

    print(f"Größte Klasse: {max_count} Samples")
    print(f"Kleinste Klasse: {min_count} Samples")
    print(f"Median: {median_count} Samples")
    print(f"Ungleichgewichts-Ratio: {max_count / min_count:.1f}:1")

    # Kategorisierung der Klassen
    minority_classes = []
    majority_classes = []
    balanced_classes = []

    threshold_low = median_count * 0.5
    threshold_high = median_count * 1.5

    for class_name, count in train_counts.items():
        if count < threshold_low:
            minority_classes.append((class_name, count))
        elif count > threshold_high:
            majority_classes.append((class_name, count))
        else:
            balanced_classes.append((class_name, count))

    print(f"\n📊 Klassenkategorisierung:")
    print(f"Minderheitsklassen (< {threshold_low:.0f}): {len(minority_classes)}")
    for name, count in sorted(minority_classes, key=lambda x: x[1]):
        print(f"  ⚠️  {name}: {count} (benötigt +{int(median_count - count)} Samples)")

    print(
        f"\nAusgewogene Klassen ({threshold_low:.0f}-{threshold_high:.0f}): {len(balanced_classes)}"
    )
    for name, count in balanced_classes:
        print(f"  ✅ {name}: {count}")

    print(f"\nMehrheitsklassen (> {threshold_high:.0f}): {len(majority_classes)}")
    for name, count in sorted(majority_classes, key=lambda x: x[1], reverse=True):
        print(f"  📈 {name}: {count}")

    # Empfehlungen
    print(f"\n💡 EMPFEHLUNGEN:")

    if len(minority_classes) > 0:
        print("🔄 DATENAUGMENTATION:")
        print("  - Synthetische Bilder für Minderheitsklassen generieren")
        print("  - Faktor 2-5x Augmentation für kleine Klassen")
        print("  - Starke geometrische + Farb-Transformationen")

        print("\n⚖️  SAMPLING STRATEGIEN:")
        print("  - WeightedRandomSampler für Klassenbalancing")
        print("  - Oversampling von Minderheitsklassen")
        print("  - Focal Loss für schwierige Klassen")

    if max_count / min_count > 10:
        print("\n🚨 KRITISCHES UNGLEICHGEWICHT:")
        print("  - Ratio > 10:1 erkannt")
        print("  - Dringend Balancing erforderlich")
        print("  - Evaluation-Metriken: Precision/Recall statt Accuracy")

    return {
        "minority_classes": minority_classes,
        "majority_classes": majority_classes,
        "balanced_classes": balanced_classes,
        "imbalance_ratio": max_count / min_count,
    }


def recommend_image_preprocessing(dataset_info):
    """Empfiehlt Bildvorverarbeitung basierend auf Größenanalyse"""

    print("\n=== BILDVORVERARBEITUNG EMPFEHLUNGEN ===")

    print("📐 GRÖSSEN-VEREINHEITLICHUNG:")
    print("  1. Resize auf 256x256 (Zwischengröße)")
    print("  2. RandomCrop/CenterCrop auf 224x224 (Standard)")
    print("  3. Aspect Ratio beibehalten wo möglich")

    print("\n🎨 NORMALISIERUNG:")
    print("  - ImageNet-Statistiken verwenden:")
    print("    mean=[0.485, 0.456, 0.406]")
    print("    std=[0.229, 0.224, 0.225]")

    print("\n🔄 AUGMENTATION-PIPELINE:")
    print("  Training:")
    print("    - HorizontalFlip (0.5)")
    print("    - Rotation (±30°)")
    print("    - ColorJitter (brightness, contrast)")
    print("    - RandomCrop nach Resize")
    print("  Test:")
    print("    - Nur Resize + CenterCrop")
    print("    - Keine Augmentation")

    print("\n⚡ PERFORMANCE-OPTIMIERUNG:")
    print("  - Bilder im RAM cachen bei kleinen Datasets")
    print("  - Multi-Threading für Datenladung")
    print("  - Pin Memory bei GPU-Training")


def main():
    """Hauptfunktion für die Datenexploration"""  # Dataset analysieren
    print("🔍 Analysiere Dataset-Struktur...")
    dataset_info = explore_dataset_structure()

    if not dataset_info:
        print("❌ Keine Daten gefunden!")
        print(
            "💡 Tipp: Stelle sicher, dass das PlantDoc-Dataset in 'data/PlantDoc/' liegt"
        )
        print("📁 Erwartete Struktur:")
        print("   data/PlantDoc/train/{klasse1,klasse2,...}/")
        print("   data/PlantDoc/test/{klasse1,klasse2,...}/")
        return

    # Klassenverteilung visualisieren
    print("\n📊 Erstelle Klassenverteilung...")
    plot_class_distribution(dataset_info, save_path="reports/class_distribution.png")

    # Train/Test-Konsistenz prüfen
    print("\n🔄 Prüfe Train/Test-Konsistenz...")
    analyze_train_test_split(dataset_info)

    # Bildgrößen analysieren
    print("\n📐 Analysiere Bildgrößen...")
    analyze_image_sizes(dataset_info)

    # Klassenungleichgewicht analysieren
    print("\n⚖️ Analysiere Klassenungleichgewicht...")
    imbalance_info = analyze_class_imbalance(dataset_info)

    # Empfehlungen für Bildvorverarbeitung
    recommend_image_preprocessing(dataset_info)

    # Klassenungleichgewicht analysieren
    print("\n⚖️ Analysiere Klassenungleichgewicht...")
    imbalance_analysis = analyze_class_imbalance(dataset_info)

    # Bildvorverarbeitung empfehlen
    print("\n🖼️ Empfehlungen zur Bildvorverarbeitung...")
    recommend_image_preprocessing(dataset_info)

    print("\n✅ Datenexploration abgeschlossen!")


if __name__ == "__main__":
    main()
