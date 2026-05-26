# Modèles Entraînés

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
