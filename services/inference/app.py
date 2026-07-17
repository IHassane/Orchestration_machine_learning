import os
import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configuration du processeur (CPU pour Kubernetes/Minikube)
DEVICE = torch.device("cpu")
print("[INFO] Service d'inférence configuré sur : CPU")

# CORRECTIF : Les vraies classes définies dans training.py
MULTICLASS_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SEUIL_MALIGNANT = 0.5

# Variables globales pour les modèles
binary_model = None
multiclass_model = None

class InferenceRequest(BaseModel):
    tensor: list  # Reçoit le tenseur sérialisé en liste JSON

def load_mobilenet_v2_binary():
    model = models.mobilenet_v2(weights=None)
    
    # CORRECTIF : Doit correspondre EXACTEMENT à la structure de training.py
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1280, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    path = "models/binary_model.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de modèle binaire est introuvable au chemin : {path}")
        
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
    
    # CORRECTIF : Doit correspondre EXACTEMENT à la structure multiclasse de training.py
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(MULTICLASS_CLASSES))
    )
    
    path = "models/multiclass_model.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de modèle multiclasse est introuvable au chemin : {path}")
        
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

# CORRECTIF : Remplacement du startup_event déprécié par le gestionnaire moderne Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global binary_model, multiclass_model
    try:
        print("[INFO] Chargement des modèles MobileNetV2 en mémoire...")
        binary_model = load_mobilenet_v2_binary()
        multiclass_model = load_mobilenet_v2_multiclass()
        print("[INFO] Modèles chargés avec succès et prêts pour l'inférence.")
    except Exception as e:
        # En cas d'erreur de chargement, on affiche clairement l'erreur de dimension ou de chemin
        print(f"[ERREUR CRITIQUE] Impossible de charger les modèles : {str(e)}")
        # Optionnel : lever l'exception pour forcer le crash du conteneur en local et voir le bug direct
        # raise e 
    yield
    # Nettoyage à l'extinction
    binary_model = None
    multiclass_model = None

# On initialise FastAPI avec le lifespan
app = FastAPI(title="HAM10000 Inference Service", lifespan=lifespan)

@app.post("/predict")
async def predict(payload: InferenceRequest):
    if binary_model is None or multiclass_model is None:
        raise HTTPException(status_code=503, detail="Modèles non disponibles ou en cours de chargement.")
    
    try:
        # Reconversion en tenseur PyTorch (batch size = 1)
        input_tensor = torch.tensor(payload.tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 1. Inférence Binaire
        with torch.no_grad():
            binary_outputs = binary_model(input_tensor)
            binary_probs = torch.softmax(binary_outputs, dim=1)[0]
            
        # Index 1 = Malignant (selon training.py)
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
        
        # 2. Logique de Routage
        if result["is_suspect"]:
            result["routing_triggered"] = True
            with torch.no_grad():
                multi_outputs = multiclass_model(input_tensor)
                multi_probs = torch.softmax(multi_outputs, dim=1)[0]
                
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
    if binary_model is not None and multiclass_model is not None:
        return {"status": "healthy", "models_loaded": True}
    return {"status": "starting", "models_loaded": False}