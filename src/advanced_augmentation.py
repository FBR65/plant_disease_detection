"""
Erweiterte Datenaugmentation für Pflanzenkrankheitserkennung
Löst Probleme mit Klassenungleichgewicht und Bildgrößen-Variation
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import yaml
import cv2


class AdvancedPlantDataset(Dataset):
    """
    Erweitertes Dataset mit Augmentation und Klassenbalancing
    """

    def __init__(
        self,
        data_dir,
        split="train",
        config_path=None,
        augment_minority_classes=True,
        synthetic_factor=3,
    ):
        """
        Args:
            data_dir: Pfad zum Datenverzeichnis
            split: 'train' oder 'test'
            config_path: Pfad zur Konfigurationsdatei
            augment_minority_classes: Ob Minderheitsklassen verstärkt werden sollen
            synthetic_factor: Faktor für synthetische Bilder (3 = 3x mehr Augmentations)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment_minority_classes = augment_minority_classes
        self.synthetic_factor = synthetic_factor

        # Konfiguration laden
        self.config = self._load_config(config_path)

        # Daten sammeln
        self.samples = []
        self.class_counts = defaultdict(int)
        self._collect_samples()

        # Klassen-Mapping erstellen
        self.classes = sorted(list(self.class_counts.keys()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Klassengewichte für Weighted Sampling berechnen
        self.class_weights = self._calculate_class_weights()
        self.sample_weights = self._calculate_sample_weights()

        # Augmentation basierend auf Klassenungleichgewicht
        if augment_minority_classes and split == "train":
            self._create_synthetic_samples()

        # Transformationen definieren
        self.transform = self._get_transforms()

        print(f"\n=== {split.upper()} DATASET STATISTIKEN ===")
        print(f"Originale Samples: {len(self.samples)}")
        if hasattr(self, "synthetic_samples"):
            print(f"Synthetische Samples: {len(self.synthetic_samples)}")
            print(f"Gesamt: {len(self.samples) + len(self.synthetic_samples)}")

        # Klassenverteilung anzeigen
        final_counts = Counter()
        for _, class_name, _ in self.get_all_samples():
            final_counts[class_name] += 1

        print("\nFinale Klassenverteilung:")
        for class_name in sorted(final_counts.keys()):
            print(f"  {class_name}: {final_counts[class_name]} Samples")

    def _load_config(self, config_path):
        """Lädt Konfiguration"""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        # Standard-Konfiguration
        return {
            "image_size": 224,
            "augmentation": {
                "rotation_range": 30,
                "zoom_range": 0.2,
                "brightness_range": 0.2,
                "horizontal_flip": True,
            },
        }

    def _collect_samples(self):
        """Sammelt alle Bilddateien"""
        split_dir = self.data_dir / self.split

        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name

                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((str(img_path), class_name, "original"))
                        self.class_counts[class_name] += 1

    def _calculate_class_weights(self):
        """Berechnet Gewichte für jede Klasse (inverse Häufigkeit)"""
        total_samples = sum(self.class_counts.values())
        weights = {}

        for class_name, count in self.class_counts.items():
            weights[class_name] = total_samples / (len(self.class_counts) * count)

        return weights

    def _calculate_sample_weights(self):
        """Berechnet Gewichte für jedes Sample"""
        weights = []
        for _, class_name, _ in self.samples:
            weights.append(self.class_weights[class_name])
        return weights

    def _create_synthetic_samples(self):
        """Erstellt synthetische Samples für Minderheitsklassen"""
        self.synthetic_samples = []

        # Bestimme die maximale Klassengröße
        max_count = max(self.class_counts.values())

        for class_name, count in self.class_counts.items():
            if count < max_count * 0.5:  # Nur für Klassen mit < 50% der max. Größe
                # Wie viele synthetische Samples benötigt werden
                target_count = min(max_count, count * self.synthetic_factor)
                needed = target_count - count

                if needed > 0:
                    # Original-Samples dieser Klasse
                    class_samples = [s for s in self.samples if s[1] == class_name]

                    # Synthetische Samples erstellen
                    for i in range(needed):
                        # Zufälliges Original-Sample als Basis
                        base_sample = class_samples[i % len(class_samples)]
                        synthetic_sample = (base_sample[0], base_sample[1], "synthetic")
                        self.synthetic_samples.append(synthetic_sample)

                print(
                    f"Klasse '{class_name}': {count} → {count + needed} (+{needed} synthetisch)"
                )

    def get_all_samples(self):
        """Gibt alle Samples zurück (original + synthetic)"""
        all_samples = self.samples.copy()
        if hasattr(self, "synthetic_samples"):
            all_samples.extend(self.synthetic_samples)
        return all_samples

    def _get_transforms(self):
        """Definiert Transformationen basierend auf albumentations"""

        if self.split == "train":
            return A.Compose(
                [
                    # Größe vereinheitlichen
                    A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR),
                    A.RandomCrop(height=224, width=224),
                    # Geometrische Augmentationen
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(limit=30, p=0.5),
                    # Farb-/Helligkeit-Augmentationen
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=0.3,
                    ),
                    A.CLAHE(clip_limit=2.0, p=0.3),
                    # Rauschen und Blur
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    # Pflanzen-spezifische Augmentationen
                    A.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
                    ),
                    A.RandomShadow(p=0.3),
                    A.RandomSunFlare(p=0.1),
                    # Normalisierung (ImageNet Stats)
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            # Test/Validation: Nur Resize und Normalisierung
            return A.Compose(
                [
                    A.Resize(height=256, width=256),
                    A.CenterCrop(height=224, width=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.get_all_samples())

    def __getitem__(self, idx):
        all_samples = self.get_all_samples()
        img_path, class_name, sample_type = all_samples[idx]

        # Bild laden
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"Fehler beim Laden von {img_path}: {e}")
            # Fallback: schwarzes Bild
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Stärkere Augmentation für synthetische Samples
        if sample_type == "synthetic" and self.split == "train":
            # Zusätzliche starke Augmentationen für synthetische Bilder
            strong_aug = A.Compose(
                [
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.3),
                ]
            )
            image = strong_aug(image=image)["image"]

        # Standard-Transformationen anwenden
        if self.transform:
            image = self.transform(image=image)["image"]

        label = self.class_to_idx[class_name]

        return image, label

    def get_weighted_sampler(self):
        """Erstellt einen WeightedRandomSampler für Klassenbalancing"""
        all_weights = []
        all_samples = self.get_all_samples()

        for _, class_name, _ in all_samples:
            all_weights.append(self.class_weights[class_name])

        return WeightedRandomSampler(
            weights=all_weights, num_samples=len(all_weights), replacement=True
        )


def create_balanced_dataloader(
    data_dir,
    split="train",
    batch_size=32,
    num_workers=4,
    use_weighted_sampling=True,
    augment_minority_classes=True,
    synthetic_factor=3,
):
    """
    Erstellt einen ausbalancierten DataLoader

    Args:
        data_dir: Pfad zu den Daten
        split: 'train' oder 'test'
        batch_size: Batch-Größe
        num_workers: Anzahl Worker-Threads
        use_weighted_sampling: Ob Weighted Sampling verwendet werden soll
        augment_minority_classes: Ob Minderheitsklassen augmentiert werden sollen
        synthetic_factor: Faktor für synthetische Datenerzeugung
    """

    dataset = AdvancedPlantDataset(
        data_dir=data_dir,
        split=split,
        augment_minority_classes=augment_minority_classes,
        synthetic_factor=synthetic_factor,
    )

    # Sampler wählen
    sampler = None
    shuffle = True

    if split == "train" and use_weighted_sampling:
        sampler = dataset.get_weighted_sampler()
        shuffle = False  # Bei Sampler kein shuffle

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if split == "train" else False,
    )

    return dataloader, dataset


def analyze_dataloader_balance(dataloader, dataset, num_batches=10):
    """Analysiert die Klassenverteilung in einem DataLoader"""

    class_counts = defaultdict(int)
    total_samples = 0

    print(f"\n=== DATALOADER BALANCE ANALYSE ({num_batches} Batches) ===")

    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break

        for label in labels:
            class_name = dataset.idx_to_class[label.item()]
            class_counts[class_name] += 1
            total_samples += 1

    print(f"Analysierte Samples: {total_samples}")
    print("Klassenverteilung in Batches:")

    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    return class_counts


if __name__ == "__main__":
    # Test der neuen Datenaugmentation

    print("🔄 Teste erweiterte Datenaugmentation...")

    # Balanced DataLoader erstellen
    train_loader, train_dataset = create_balanced_dataloader(
        data_dir="data/PlantDoc",
        split="train",
        batch_size=16,
        use_weighted_sampling=True,
        augment_minority_classes=True,
        synthetic_factor=3,
    )

    # Test DataLoader
    test_loader, test_dataset = create_balanced_dataloader(
        data_dir="data/PlantDoc",
        split="test",
        batch_size=16,
        use_weighted_sampling=False,
        augment_minority_classes=False,
    )

    # Balance analysieren
    analyze_dataloader_balance(train_loader, train_dataset)

    print("\n✅ Erweiterte Datenaugmentation getestet!")
