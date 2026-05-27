import os
import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HAM10000 Inference Service")

# Forcer l'utilisation du CPU dans le cluster Kubernetes (conforme aux specs K8s/Minikube)
DEVICE = torch.device("cpu")
print("[INFO] Service d'inférence configuré sur : CPU")

# Configuration des classes issues de ton entraînement
MULTICLASS_CLASSES = ["bkl", "df", "mel", "meln", "nv", "vascular", "derm"]
SEUIL_MALIGNANT = 0.5  # Seuil de décision défini dans ton ADR

# Variables globales pour stocker les modèles en mémoire
binary_model = None
multiclass_model = None

class InferenceRequest(BaseModel):
    tensor: list  # Reçoit le tenseur sérialisé en liste JSON depuis le preprocessing

def load_mobilenet_v2_binary():
    model = models.mobilenet_v2(weights=None)
    
    # Structure exacte de ton entraînement binaire : 128 neurones, pas de 2e dropout
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    
    path = "models/binary_model.pt"
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

def load_mobilenet_v2_multiclass():
    model = models.mobilenet_v2(weights=None)
    
    # Structure exacte de ton entraînement multi-classes : 256 neurones + 2e dropout
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(MULTICLASS_CLASSES))
    )
    
    path = "models/multiclass_model.pt"
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

@app.on_event("startup")
def startup_event():
    """Chargement unique des modèles au démarrage du conteneur (Gain de perf massif)."""
    global binary_model, multiclass_model
    try:
        print("[INFO] Chargement des modèles MobileNetV2 en mémoire...")
        binary_model = load_mobilenet_v2_binary()
        multiclass_model = load_mobilenet_v2_multiclass()
        print("[INFO] Modèles chargés avec succès et prêts pour l'inférence.")
    except Exception as e:
        print(f"[ERREUR CRITIQUE] Impossible de charger les modèles : {str(e)}")
        # On ne lève pas d'exception ici pour laisser le conteneur démarrer,
        # le liveness/readiness probe gérera le statut.

@app.post("/predict")
async def predict(payload: InferenceRequest):
    if binary_model is None or multiclass_model is None:
        raise HTTPException(status_code=503, detail="Modèles non disponibles ou en cours de chargement.")
    
    try:
        # 1. Reconversion de la liste JSON en Tenseur PyTorch
        # Le preprocessing envoie une forme (1, C, H, W)
        # .unsqueeze(0) transforme la forme (3, 224, 224) en (1, 3, 224, 224)
        input_tensor = torch.tensor(payload.tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 2. Inférence Binaire (Filtre Principal)
        with torch.no_grad():
            binary_outputs = binary_model(input_tensor)
            binary_probs = torch.softmax(binary_outputs, dim=1)[0]
            
        # Supposons que l'index 1 correspond à 'malignant' dans ton LabelEncoder binaire
        malignant_prob = binary_probs[1].item()
        benign_prob = binary_probs[0].item()
        
        result = {
            "is_suspect": malignant_prob >= SEUIL_MALIGNANT,
            "binary_scores": {
                "benign": round(benign_prob, 4),
                "malignant": round(malignant_prob, 4)
            },
            "routing_triggered": False,
            "final_diagnosis": "benign" if malignant_prob < SEUIL_MALIGNANT else "malignant_unclassified"
        }
        
        # 3. Logique de Routage (Règle Métier de l'ADR)
        if result["is_suspect"]:
            result["routing_triggered"] = True
            with torch.no_grad():
                multi_outputs = multiclass_model(input_tensor)
                multi_probs = torch.softmax(multi_outputs, dim=1)[0]
                
            # Extraire la classe dominante du modèle secondaire
            top_idx = torch.argmax(multi_probs).item()
            result["final_diagnosis"] = MULTICLASS_CLASSES[top_idx]
            result["multiclass_scores"] = {
                MULTICLASS_CLASSES[i]: round(multi_probs[i].item(), 4) for i in range(len(MULTICLASS_CLASSES))
            }
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'inférence : {str(e)}")

@app.get("/health")
def health_check():
    """Indique à Kubernetes si le conteneur est opérationnel et si les modèles sont bien là."""
    if binary_model is not None and multiclass_model is not None:
        return {"status": "healthy", "models_loaded": True}
    return {"status": "starting", "models_loaded": False}