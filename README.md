# <span style="color: #1976D2;">Projet TRIGRAMME — Orchestration d'un modèle CNN Dermatologique sur Kubernetes</span>

## Présentation du Projet & Technologies

Ce projet a pour but de déployer, orchestrer et monitorer en production un modèle de Deep Learning (**CNN**) spécialisé dans la classification d'images médicales, en respectant les standards industriels du **MLOps**.

L'application sert d'outil de tri automatisé pour un service de dermatologie : elle analyse des photographies de lésions cutanées (dataset HAM10000) afin de détecter les cas suspects et d'aider les médecins à prioriser les urgences.

---

## Stack Technologique

Le projet s'articule autour d'un écosystème technologique moderne pour assurer la transition du modèle de la phase d'entraînement à la phase de production :

* **Deep Learning :** `PyTorch` & `Torchvision` pour l'architecture du CNN (**MobileNetV2**).
* **Développement API :** `Python` (Flask / FastAPI) pour la création des micro-services (Preprocessing et Inférence).
* **Conteneurisation :** `Docker` pour packager le code, le runtime PyTorch et les poids du modèle dans des images légères et isolées.
* **Orchestration :** `Kubernetes (Minikube)` pour gérer le cycle de vie des conteneurs, le routage réseau interne et l'auto-guerison des pods.
* **CI/CD :** `GitHub Actions` pour automatiser les tests unitaires et le build/push des images Docker sur le `Docker Hub`.
* **Monitoring :** `Prometheus` pour la collecte des métriques de performance (latence d'inférence, taux d'erreur).

---

## Architecture des Micro-services

Pour optimiser les ressources, l'application est découpée en **trois briques** qui communiquent au sein du cluster Kubernetes via le protocole HTTP et le DNS interne :

1. **Service Preprocessing :** Reçoit l'image brute, la décompresse, la redimensionne au format standardisé du CNN (224x224 pixels) et applique la normalisation.
2. **Service Inférence (Le cœur du CNN) :** Charge le modèle en mémoire. Il exécute d'abord un filtre binaire (bénin/malin). Si la lésion est suspecte, il active en cascade un second classifieur multi-classes (7 types de maladies) pour affiner le diagnostic.
3. **Service Monitoring :** Enregistre les métriques d'utilisation en temps réel et expose un endpoint lisible par Prometheus.

---

## Gestion des Quotas Kubernetes

Parce que le Deep Learning est gourmand en ressources, l'architecture est optimisée pour respecter un quota strict de **3500m CPU** et **5 Gi de RAM** :

* **Choix du CNN :** Utilisation de **MobileNetV2**. Ses poids ne pèsent que 14 Mo sur disque (contre 20 Mo pour EfficientNet), ce qui limite l'empreinte mémoire d'exécution avec le runtime PyTorch à environ 1 Gi.
* **Stratégie de Déploiement :** Utilisation de la stratégie **`Recreate`**. Pour éviter que le cluster ne sature lors des mises à jour de code (effet de *surge*), les anciens pods sont coupés avant de démarrer les nouveaux, maintenant la consommation stable à **2,75 Gi** au total.

---

## Structure du Dépôt

```text
.
├── .github/workflows/ci-cd.yml   # Pipeline d'automatisation GitHub Actions
├── models/
│   ├── binary_model.pt           # Poids du CNN entraîné (MobileNetV2)
│   └── train_indices.npy         # Indices pour la reproductibilité
├── ham10000/                     # Images des lésions (Ignoré par Git)
├── .gitignore                    # Fichiers exclus du dépôt (Données lourdes, scripts locaux)
└── README.md                     # Documentation du projet