import os
import io
import time
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# --- AJOUT DES CORS ---
from fastapi.middleware.cors import CORSMiddleware
# ----------------------
from PIL import Image
import numpy as np

app = FastAPI(title="HAM10000 Preprocessing Service")

# --- AJOUT DES CORS ---
# On autorise toutes les origines (pratique en dev / local), 
# tous les headers et toutes les méthodes HTTP (POST, GET, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------

# Récupération des URLs depuis les variables d'environnement
INFERENCE_SERVICE_URL = os.getenv(
    "INFERENCE_SERVICE_URL", 
    "http://inference-svc.projet-trigramme.svc.cluster.local/predict"
)

# Utilise ton fallback local ou K8s selon l'environnement
MONITORING_URL = os.getenv(
    "MONITORING_URL", 
    "http://monitoring-svc.projet-trigramme.svc.cluster.local:9090/log"
)

IMG_SIZE = 224

# Moyenne et écart-type ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_bytes: bytes) -> list:
    """Pipeline de validation identique à l'entraînement."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = img_array.transpose(2, 0, 1)
    return img_array.tolist()

def send_metrics_to_monitoring(status_code: int, latency: float):
    """Fonction exécutée en arrière-plan pour notifier le monitoring."""
    try:
        requests.post(
            MONITORING_URL, 
            json={"status_code": status_code, "latency": latency}, 
            timeout=1
        )
    except Exception as e:
        # On log l'erreur en console mais on ne crash pas le preprocessing
        print(f"[Monitoring Link Failed]: {e}")

@app.post("/predict")
async def forward_prediction(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint principal : prépare l'image, appelle l'inférence,
    et délègue le logging au monitoring en tâche de fond.
    """
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Seules les images JPEG sont acceptées.")
    
    t0 = time.monotonic()
    status_code = 200 # Initialisation par défaut
    
    try:
        # Lecture et traitement de l'image
        image_bytes = await file.read()
        processed_tensor = preprocess_image(image_bytes)
        
        # Envoi du tenseur au service d'inférence
        response = requests.post(
            INFERENCE_SERVICE_URL,
            json={"tensor": processed_tensor},
            timeout=10
        )
        
        status_code = response.status_code

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Erreur du service d'inférence : {response.text}"
            )
            
        return response.json()
        
    except HTTPException as http_err:
        status_code = http_err.status_code
        raise http_err
    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail=f"Erreur lors du preprocessing : {str(e)}")
        
    finally:
        # Calcul de la latence globale du Preprocessing
        latency = time.monotonic() - t0
        # Ajout de la notification au monitoring dans les tâches de fond de FastAPI
        background_tasks.add_task(send_metrics_to_monitoring, status_code, latency)

@app.get("/health")
def health_check():
    return {"status": "healthy"}