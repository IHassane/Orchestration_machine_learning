import pytest
from fastapi.testclient import TestClient
import torch
import numpy as np

# On désactive le chargement automatique des modèles lors de l'import pour le runner CI
import services.inference.app as inference_mod
inference_mod.load_mobilenet_v2_binary = lambda: torch.nn.Sequential()
inference_mod.load_mobilenet_v2_multiclass = lambda: torch.nn.Sequential()

from services.inference.app import app

client = TestClient(app)

def test_health_endpoint_models_not_loaded():
    inference_mod.binary_model = None
    inference_mod.multiclass_model = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "starting", "models_loaded": False}

def test_health_endpoint_healthy():
    inference_mod.binary_model = torch.nn.Sequential()
    inference_mod.multiclass_model = torch.nn.Sequential()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "models_loaded": True}

def test_predict_models_unavailable():
    inference_mod.binary_model = None
    inference_mod.multiclass_model = None
    fake_tensor = np.random.randn(3, 224, 224).tolist()
    response = client.post("/predict", json={"tensor": fake_tensor})
    assert response.status_code == 503
    assert "Modèles non disponibles" in response.json()["detail"]

def test_predict_benign_case(mocker):
    # Setup des faux modèles fonctionnels
    mock_binary = mocker.MagicMock()
    mock_multiclass = mocker.MagicMock()
    
    # Simulation des outputs de MobileNet (Logits)
    # Pour le cas bénin : forte valeur sur l'index 0, faible sur l'index 1 [benign, malignant]
    mock_binary.return_value = torch.tensor([[2.0, -2.0]]) 
    
    inference_mod.binary_model = mock_binary
    inference_mod.multiclass_model = mock_multiclass

    fake_tensor = np.random.randn(3, 224, 224).tolist()
    response = client.post("/predict", json={"tensor": fake_tensor})
    
    assert response.status_code == 200
    res = response.json()
    assert res["is_suspect"] is False
    assert res["routing_triggered"] is False
    assert res["final_diagnosis"] == "benign"
    assert res["binary_scores"]["benign"] > 0.5

def test_predict_malignant_routing_case(mocker):
    mock_binary = mocker.MagicMock()
    mock_multiclass = mocker.MagicMock()
    
    # Pour le cas malin : faible valeur sur index 0, forte sur index 1 (seuil dépassé)
    mock_binary.return_value = torch.tensor([[-2.0, 2.0]])
    # Pour le modèle multiclasse : on met une forte valeur sur l'index 2 (qui correspond à "mel")
    mock_multiclass.return_value = torch.tensor([[-1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0]])

    inference_mod.binary_model = mock_binary
    inference_mod.multiclass_model = mock_multiclass

    fake_tensor = np.random.randn(3, 224, 224).tolist()
    response = client.post("/predict", json={"tensor": fake_tensor})
    
    assert response.status_code == 200
    res = response.json()
    assert res["is_suspect"] is True
    assert res["routing_triggered"] is True
    assert res["final_diagnosis"] == "mel"  # Index 2 de MULTICLASS_CLASSES
    assert "multiclass_scores" in res

def test_predict_internal_server_error(mocker):
    mock_binary = mocker.MagicMock()
    # On force le modèle à lever une exception pour tester le bloc 'except Exception'
    mock_binary.side_effect = RuntimeError("PyTorch Tensor Error")
    
    inference_mod.binary_model = mock_binary
    inference_mod.multiclass_model = torch.nn.Sequential()

    fake_tensor = np.random.randn(3, 224, 224).tolist()
    response = client.post("/predict", json={"tensor": fake_tensor})
    
    assert response.status_code == 500
    assert "Erreur lors de l'inférence" in response.json()["detail"]