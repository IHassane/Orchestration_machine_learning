import os
import io
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np

app = FastAPI(title="HAM10000 Preprocessing Service")

# Récupération de l'URL du service d'inférence depuis les variables d'environnement K8s
INFERENCE_SERVICE_URL = os.getenv(
    "INFERENCE_SERVICE_URL", 
    "http://inference-svc.projet-trigramme.svc.cluster.local/predict"
)

IMG_SIZE = 224

# Moyenne et écart-type ImageNet (issus de ton script d'entraînement)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_bytes: bytes) -> list:
    """
    Reproduit exactement le pipeline de validation de ton script d'entraînement :
    1. Chargement de l'image en RGB
    2. Redimensionnement en 224x224 (Bilinear)
    3. Conversion en Tenseur (0-1)
    4. Normalisation ImageNet
    """
    # 1. Chargement de l'image brute (comme dans ton HAM10000Dataset)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Redimensionnement (Bilinear)
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    
    # 3. Conversion en array numpy et passage à l'échelle [0, 1] (équivalent de ToTensor)
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # 4. Normalisation (équivalent de transforms.Normalize)
    # Formule : (image - mean) / std
    img_array = (img_array - MEAN) / STD
    
    # 5. Réorganisation des dimensions de (H, W, C) à (C, H, W) attendu par PyTorch
    img_array = img_array.transpose(2, 0, 1)
    
    # Ajout de la dimension de batch (1, C, H, W) et conversion en liste pour le JSON
    return img_array.tolist()

@app.post("/predict")
async def forward_prediction(file: UploadFile = File(...)):
    """
    Endpoint principal qui reçoit l'image du script de charge,
    la prépare, et bascule la requête vers l'inférence.
    """
    # Vérification du format (ton ADR et ton cours mentionnent uniquement le JPEG)
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Seules les images JPEG sont acceptées.")
    
    try:
        # Lecture et traitement CPU de l'image
        image_bytes = await file.read()
        processed_tensor = preprocess_image(image_bytes)
        
        # Envoi du tenseur au format JSON au service d'inférence
        response = requests.post(
            INFERENCE_SERVICE_URL,
            json={"tensor": processed_tensor},
            timeout=10
        )
        
        # Si le service d'inférence répond avec une erreur, on la relaie
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Erreur du service d'inférence : {response.text}"
            )
            
        # Renvoie la réponse finale (JSON complet avec prediction et confidence)
        return response.json()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du preprocessing : {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint utile pour Kubernetes (Liveness/Readiness probes)"""
    return {"status": "healthy"}

import time
import requests

# URL DNS interne K8s du monitoring
MONITORING_URL = "http://monitoring-svc.projet-trigramme.svc.cluster.local:9090/log"

# ... dans ton endpoint /predict :
t0 = time.monotonic()
# [Logique de preprocessing + appel à l'inférence]
status_code = 200 # ou le code d'erreur intercepté
latency = time.monotonic() - t0

# Envoi asynchrone ou direct au monitoring (en tâche de fond pour ne pas bloquer l'inférence)
try:
    requests.post(MONITORING_URL, json={"status_code": status_code, "latency": latency}, timeout=1)
except Exception:
    pass # Le monitoring ne doit jamais faire planter l'application principale