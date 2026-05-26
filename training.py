#!/usr/bin/env python3
"""
training.py - Script d'entraînement et de fine-tuning de modèles MobileNetV2
pour la classification d'images dermatologiques (HAM10000).

Ce script entraîne deux modèles :
  1. Modèle binaire (bénin/malin) - Classifieur principal
  2. Modèle multi-classes (7 types de lésions) - Classifieur secondaire

Le dataset HAM10000 est disponible sous licence CC BY-NC 4.0.
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224  # Taille d'entrée standard pour MobileNetV2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS_BINARY = 20
NUM_EPOCHS_MULTICLASS = 25
BINARY_MODEL_PATH = "models/binary_model.pt"
MULTICLASS_MODEL_PATH = "models/multiclass_model.pt"
METADATA_PATH = "content/ham10000/HAM10000_metadata.csv"
IMAGES_DIR = "content/ham10000"

# Classes du dataset HAM10000
# bkl:  Benign Keratosis-like lesions
# df:   Dermatitis
# mel:  Melanoma
# meln: Melanocytic nevi
# nv:   Melanocytic nevi (legacy label)
# vasc: Vascular lesions
# derm: Dermatofibroma

# Mapping binaire : bénin vs malin
# Bénin : bkl, df, nv, derm, vasc
# Malin : mel, meln
BENIGN_CLASSES = {"bkl", "df", "nv", "derm", "vascular"}
MALIGN_CLASSES = {"mel", "meln"}

# Pour le modèle multi-classes, on regroupe en 7 classes
# On peut aussi regrouper mel et meln ensemble pour simplifier
MULTICLASS_CLASSES = ["bkl", "df", "mel", "meln", "nv", "vascular", "derm"]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Détermine les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed défini : {seed}")


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Charge le fichier CSV de métadonnées HAM10000."""
    print(f"[INFO] Chargement des métadonnées depuis : {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"[INFO] {len(df)} lignes chargées")
    return df


def get_binary_label(dx: str) -> int:
    """
    Retourne 0 pour bénin, 1 pour malin.
    """
    if dx in BENIGN_CLASSES:
        return 0
    elif dx in MALIGN_CLASSES:
        return 1
    else:
        # Par défaut, considérer comme bénin si classe inconnue
        return 0


def find_image_path(dx: str, image_id: str, images_dir: str) -> str:
    """
    Cherche le chemin de l'image dans les sous-répertoires d'images.
    Le dataset HAM10000 est divisé en plusieurs répertoires.
    """
    # Essayer les différents répertoires d'images
    possible_dirs = [
        os.path.join(images_dir, "HAM10000_images_part_1"),
        os.path.join(images_dir, "HAM10000_images_part_2"),
        os.path.join(images_dir, "ISIC_images"),
        os.path.join(images_dir, "images"),
    ]

    filename = f"{image_id}.jpg"

    for d in possible_dirs:
        filepath = os.path.join(d, filename)
        if os.path.exists(filepath):
            return filepath

    # Essayer avec .jpeg
    filename = f"{image_id}.jpeg"
    for d in possible_dirs:
        filepath = os.path.join(d, filename)
        if os.path.exists(filepath):
            return filepath

    return ""


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class HAM10000Dataset(Dataset):
    """
    Dataset PyTorch pour les images HAM10000.
    Charge les images à la volée depuis le disque.
    """

    def __init__(self, image_paths: List[str], labels: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Charger l'image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Impossible de charger {image_path} : {e}")
            # Retourner une image noire en cas d'erreur
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="black")

        # Redimensionner si nécessaire
        if image.size != (IMG_SIZE, IMG_SIZE):
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Construction du dataset
# ---------------------------------------------------------------------------

def build_datasets(
    df: pd.DataFrame, images_dir: str, test_size: float = 0.2, val_size: float = 0.1
):
    """
    Construit les datasets train/val/test pour les deux tâches.
    Retourne :
      - binary_data : datasets pour le classifieur binaire
      - multiclass_data : datasets pour le classifieur multi-classes
    """
    print("\n[INFO] Construction des datasets...")

    # Trouver les chemins d'images et les labels
    image_paths = []
    binary_labels = []
    multiclass_labels = []

    for _, row in df.iterrows():
        image_id = row["image_id"]
        dx = row["dx"]

        path = find_image_path(dx, image_id, images_dir)
        if path == "":
            print(f"[WARNING] Image non trouvée : {image_id}")
            continue

        image_paths.append(path)
        binary_labels.append(get_binary_label(dx))
        multiclass_labels.append(MULTICLASS_CLASSES.index(dx) if dx in MULTICLASS_CLASSES else 0)

    image_paths = np.array(image_paths)
    binary_labels = np.array(binary_labels)
    multiclass_labels = np.array(multiclass_labels)

    print(f"[INFO] {len(image_paths)} images trouvées")

    # Séparation train/val/test
    indices = np.arange(len(image_paths))
    train_indices, temp_indices = train_test_split(
        indices, test_size=test_size, random_state=SEED, stratify=binary_labels
    )
    val_size_adj = val_size / (1 - test_size)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=val_size_adj, random_state=SEED, stratify=binary_labels[temp_indices]
    )

    # Sauvegarder les indices pour reproductibilité
    np.save("models/train_indices.npy", train_indices)
    np.save("models/val_indices.npy", val_indices)
    np.save("models/test_indices.npy", test_indices)

    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Binary datasets
    binary_train_paths = image_paths[train_indices]
    binary_train_labels = binary_labels[train_indices]
    binary_val_paths = image_paths[val_indices]
    binary_val_labels = binary_labels[val_indices]
    binary_test_paths = image_paths[test_indices]
    binary_test_labels = binary_labels[test_indices]

    # Multi-class datasets
    multi_train_paths = image_paths[train_indices]
    multi_train_labels = multiclass_labels[train_indices]
    multi_val_paths = image_paths[val_indices]
    multi_val_labels = multiclass_labels[val_indices]
    multi_test_paths = image_paths[test_indices]
    multi_test_labels = multiclass_labels[test_indices]

    binary_data = {
        "train": HAM10000Dataset(binary_train_paths, binary_train_labels, train_transform),
        "val": HAM10000Dataset(binary_val_paths, binary_val_labels, val_test_transform),
        "test": HAM10000Dataset(binary_test_paths, binary_test_labels, val_test_transform),
    }

    multi_data = {
        "train": HAM10000Dataset(multi_train_paths, multi_train_labels, train_transform),
        "val": HAM10000Dataset(multi_val_paths, multi_val_labels, val_test_transform),
        "test": HAM10000Dataset(multi_test_paths, multi_test_labels, val_test_transform),
    }

    print(f"[INFO] Binary - Train: {len(binary_train_paths)}, Val: {len(binary_val_paths)}, Test: {len(binary_test_paths)}")
    print(f"[INFO] Multi-class - Train: {len(multi_train_paths)}, Val: {len(multi_val_paths)}, Test: {len(multi_test_paths)}")

    return binary_data, multi_data


# ---------------------------------------------------------------------------
# Modèle de classification binaire
# ---------------------------------------------------------------------------

def train_binary_model(binary_data: dict, num_epochs: int = NUM_EPOCHS_BINARY):
    """
    Entraîne un MobileNetV2 fine-tuné pour la classification binaire
    (bénin vs malin).
    """
    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT DU MODÈLE BINAIRE (bénin/malin)")
    print("=" * 60)

    # Charger MobileNetV2 pré-entraîné
    print("[INFO] Chargement de MobileNetV2 pré-entraîné...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Geler les couches de feature extraction
    for param in model.features.parameters():
        param.requires_grad = False

    # Remplacer le classifieur par défaut
    num_finetune = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_finetune, 128),
        nn.ReLU(),
        nn.Linear(128, 2),  # 2 classes : bénin / malin
    )

    model = model.to(DEVICE)
    print(f"[INFO] Modèle sur {DEVICE}")

    # DataLoader
    train_loader = DataLoader(
        binary_data["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        binary_data["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    # N'optimiser que les couches finales
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {epoch_loss:.4f} Val-Loss: {val_loss:.4f} "
            f"Train-Acc: {epoch_acc:.4f} Val-Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  >> Nouveau meilleur modèle ! Accuracy validation: {val_acc:.4f}")

    # Sauvegarder le meilleur modèle
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "model_type": "binary",
            "num_classes": 2,
            "input_size": IMG_SIZE,
            "architecture": "MobileNetV2",
            "best_val_accuracy": best_acc,
        },
        BINARY_MODEL_PATH,
    )
    print(f"\n[INFO] Modèle binaire sauvegardé dans : {BINARY_MODEL_PATH}")
    print(f"[INFO] Meilleure accuracy validation : {best_acc:.4f}")

    return model, best_acc


# ---------------------------------------------------------------------------
# Modèle de classification multi-classes
# ---------------------------------------------------------------------------

def train_multiclass_model(multi_data: dict, num_epochs: int = NUM_EPOCHS_MULTICLASS):
    """
    Entraîne un MobileNetV2 fine-tuné pour la classification multi-classes
    (7 types de lésions dermatologiques).
    """
    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT DU MODÈLE MULTI-CLASSES (7 types)")
    print("=" * 60)

    NUM_CLASSES = len(MULTICLASS_CLASSES)

    # Charger MobileNetV2 pré-entraîné
    print("[INFO] Chargement de MobileNetV2 pré-entraîné...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Geler les couches de feature extraction
    for param in model.features.parameters():
        param.requires_grad = False

    # Remplacer le classifieur
    num_finetune = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_finetune, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),  # 7 classes
    )

    model = model.to(DEVICE)
    print(f"[INFO] Modèle sur {DEVICE}")
    print(f"[INFO] Classes : {MULTICLASS_CLASSES}")

    # DataLoader
    train_loader = DataLoader(
        multi_data["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        multi_data["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {epoch_loss:.4f} Val-Loss: {val_loss:.4f} "
            f"Train-Acc: {epoch_acc:.4f} Val-Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  >> Nouveau meilleur modèle ! Accuracy validation: {val_acc:.4f}")

    # Sauvegarder le meilleur modèle
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "model_type": "multiclass",
            "num_classes": NUM_CLASSES,
            "input_size": IMG_SIZE,
            "architecture": "MobileNetV2",
            "classes": MULTICLASS_CLASSES,
            "best_val_accuracy": best_acc,
        },
        MULTICLASS_MODEL_PATH,
    )
    print(f"\n[INFO] Modèle multi-classes sauvegardé dans : {MULTICLASS_MODEL_PATH}")
    print(f"[INFO] Meilleure accuracy validation : {best_acc:.4f}")

    return model, best_acc


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, data_dict: dict, model_name: str):
    """
    Évalue le modèle sur l'ensemble de test.
    """
    print(f"\n[INFO] Évaluation de {model_name} sur l'ensemble de test...")

    test_loader = DataLoader(
        data_dict["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"[INFO] Test Accuracy : {acc:.4f}")
    print("\n" + classification_report(all_labels, all_preds, digits=4))

    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning MobileNetV2 sur HAM10000")
    parser.add_argument(
        "--mode",
        choices=["binary", "multiclass", "all"],
        default="all",
        help="Mode d'entraînement",
    )
    parser.add_argument(
        "--epochs-binary",
        type=int,
        default=NUM_EPOCHS_BINARY,
        help="Nombre d'époques pour le modèle binaire",
    )
    parser.add_argument(
        "--epochs-multiclass",
        type=int,
        default=NUM_EPOCHS_MULTICLASS,
        help="Nombre d'époques pour le modèle multi-classes",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=METADATA_PATH,
        help="Chemin vers le CSV de métadonnées",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=IMAGES_DIR,
        help="Chemin vers les répertoires d'images",
    )
    args = parser.parse_args()

    set_seed(SEED)

    # Charger les métadonnées
    df = load_metadata(args.metadata_path)

    # Filtrer les images non trouvées
    df["image_path"] = df.apply(
        lambda row: find_image_path(row["dx"], row["image_id"], args.images_dir),
        axis=1,
    )
    df = df[df["image_path"] != ""]
    print(f"[INFO] {len(df)} images valides après filtrage")

    if len(df) == 0:
        print("[ERROR] Aucune image trouvée. Vérifiez le chemin des images.")
        return

    # Construire les datasets
    binary_data, multi_data = build_datasets(df, args.images_dir)

    # Entraîner le modèle binaire
    if args.mode in ("binary", "all"):
        binary_model, binary_acc = train_binary_model(
            binary_data, num_epochs=args.epochs_binary
        )
        evaluate_model(binary_model, binary_data, "Modèle Binaire")

    # Entraîner le modèle multi-classes
    if args.mode in ("multiclass", "all"):
        multi_model, multi_acc = train_multiclass_model(
            multi_data, num_epochs=args.epochs_multiclass
        )
        evaluate_model(multi_model, multi_data, "Modèle Multi-Classes")

    # Créer un README pour le dossier models
    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)

    readme_content = """# Modèles Entraînés

## Dataset
- **Nom** : HAM10000 (The Skin Lesion Images against Human Annotation Dataset)
- **Source** : Harvard Dataverse
- **Licence** : Creative Commons Attribution Non-Commercial 4.0 (CC BY-NC 4.0)
- **Nombre d'images** : 10 018 images dermatologiques
- **Classes** : 7 types de lésions cutanées

## Modèle Binaire (bénin/malin)
- **Architecture** : MobileNetV2 pré-entraîné
- **Type** : Classification binaire
- **Poids sauvegardés** : `binary_model.pt`
- **Usage** : Premier filtre pour déterminer si une lésion est suspecte

## Modèle Multi-Classes (7 types de lésions)
- **Architecture** : MobileNetV2 pré-entraîné
- **Type** : Classification multi-classes
- **Poids sauvegardés** : `multiclass_model.pt`
- **Classes** : bkl, df, mel, meln, nv, vascular, derm
- **Usage** : Identification précise du type de lésion si le modèle binaire détecte une anomalie

## Pipeline d'Inférence
1. Image brute → Preprocessing (redimensionnement, normalisation)
2. Preprocessing → Modèle Binaire (bénin/malin)
3. Si "malignant" → Modèle Multi-Classes (identification du type)
4. Résultats retournés au client via API REST

## Métriques
- **Accuracy** : Métrique principale pour l'évaluation
- **F1-Score** : Métrique secondaire pour les classes déséquilibrées
- **AUC-ROC** : Métrique pour la calibration des probabilités

## Reproductibilité
- Seed : 42
- Taille d'entrée : 224x224
- Optimiseur : Adam (lr=1e-4)
- Scheduler : StepLR (step_size=5, gamma=0.5)
- Batch size : 32
"""

    os.makedirs("models", exist_ok=True)
    with open("models/README.md", "w") as f:
        f.write(readme_content)
    print("[INFO] README.md créé dans models/")

    print("\n[INFO] Tous les modèles ont été sauvegardés avec succès.")
    print(f"  - {BINARY_MODEL_PATH}")
    print(f"  - {MULTICLASS_MODEL_PATH}")


if __name__ == "__main__":
    main()
