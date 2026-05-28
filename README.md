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
* **Monitoring :** `Service de Monitoring Custom` Afin de respecter scrupuleusement les contraintes de quotas du namespace (3500m CPU / 5Gi RAM), nous n'avons pas déployé un serveur Prometheus complet (trop lourd en mémoire). À la place, un service de monitoring In-Memory sur mesure a été développé avec FastAPI et NumPy. Il collecte les métriques en temps réel via un endpoint /log et calcule dynamiquement le volume, le taux de succès et la latence P95 sur l'endpoint /metrics.
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
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # Pipeline d'automatisation Multi-Architecture GitHub Actions
├── k8s/
│   ├── quota.yaml                # Configuration du ResourceQuota (3500m CPU / 5Gi RAM)
│   └── limitrange.yaml           # Bornes de ressources par conteneur (Max 2000m CPU / 2Gi RAM)
├── models/
│   ├── binary_model.pt           # Poids du premier modèle CNN (Classification Binaire)
│   └── multiclass_model.pt       # Poids du deuxième modèle CNN (Classification Multi-classes)
├── scripts/
│   └── load_test.py              # Script officiel d'évaluation de charge et de stress
├── ADR.md                        # Architecture Decision Record (Séances 1 à 5 complet)
├── pipeline.yaml                 # Manifest global incluant Monitoring, Preprocessing et Inférence
├── .gitignore                    # Fichiers exclus (Dossier data/, résidus d'environnements virtuels)
└── README.md                     # Documentation et guide de déploiement rapide sur Minikube                 # Documentation du projet

```

# Déploiement

## Prérequis

mkdir data/images

## Instructions
Récupérer les images à ce lien: 
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
ou via ces commandes
cd data
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p /content/ham10000 --unzip
Prenez un échantillon d'images et les mettre dans images

git clone https://github.com/IHassane/Orchestration_machine_learning.git

minikube start --driver=docker --cpus=4 --memory=6144
minikube addons enable metrics-server

# Création du namespace obligatoire
kubectl create namespace projet-trigramme

# Si erreurs de téléchargement -> Injection des clés de registre  (ImagePullBackOff)
kubectl create secret docker-registry dockerhub-cred \
  --docker-username="" \
  --docker-password="" \
  --docker-email="" \
  -n projet-trigramme

# 1. Application immédiate des quotas et limites du cas d'usage 1
kubectl apply -f k8s/quota.yaml
kubectl apply -f k8s/limitrange.yaml

# 2. Déploiement simultané des microservices (Inference, Preprocessing, Monitoring)
kubectl apply -f k8s/pipeline-ml.yaml

# Attendre que tous les pods affichent le statut Running
kubectl get pods -n projet-trigramme

# Ouvrir le tunnel réseau local (Conserver ce terminal ouvert pendant les tests)
minikube service preprocessing-svc -n projet-trigramme --url

python scripts/load_test.py --case images --level nominal --url http://127.0.0.1:<PORT_DYNAMIQUE>/predict

## Si besoin
pip install requests