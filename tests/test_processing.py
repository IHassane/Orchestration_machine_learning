import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import requests

# On importe l'application FastAPI depuis ton dossier services
from services.preprocessing.app import app, preprocess_image

client = TestClient(app)

def test_health_endpoint():
    """Vérifie que le health check de Kubernetes répond 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_preprocess_image_logic():
    """Vérifie que la fonction de traitement mathématique de l'image renvoie la bonne structure."""
    # 1. Créer une fausse image RGB en mémoire (ex: 100x100)
    img = Image.new("RGB", (100, 100), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()
    
    # 2. Appeler la fonction de preprocessing directe
    tensor_list = preprocess_image(img_bytes)
    
    # 3. Vérifications des dimensions attendues (Batch de 1 supprimé ici car réorganisé dans l'inférence via unsqueeze)
    # img_array.tolist() renvoie une liste à 3 dimensions (C, H, W) -> (3, 224, 224)
    assert isinstance(tensor_list, list)
    assert len(tensor_list) == 3          # 3 canaux (R, G, B)
    assert len(tensor_list[0]) == 224     # Hauteur
    assert len(tensor_list[0][0]) == 224  # Largeur

def test_predict_rejects_non_jpeg():
    """Vérifie que l'endpoint /predict bloque les fichiers png ou texte (Code 400)."""
    files = {"file": ("test.png", b"fake_png_data", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Seules les images JPEG sont acceptées" in response.json()["detail"]

def test_predict_success_with_mock(mocker):
    """Teste le flux nominal de /predict en simulant (mockant) l'Inférence et le Monitoring."""
    # 1. Créer une vraie mini-image JPEG en mémoire pour le test
    img = Image.new("RGB", (50, 50), color="green")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    # 2. Mocker globalement requests.post
    mock_post = mocker.patch("services.preprocessing.app.requests.post")
    
    # On configure le comportement par défaut (pour l'appel à l'inférence)
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "is_suspect": False,
        "binary_scores": {"benign": 0.99, "malignant": 0.01},
        "routing_triggered": False,
        "final_diagnosis": "benign"
    }

    # 3. Envoyer la requête de test au Preprocessing
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    # 4. Assertions sur la réponse HTTP
    assert response.status_code == 200
    assert response.json()["final_diagnosis"] == "benign"
    assert response.json()["is_suspect"] is False
    
    # 5. Vérifier que requests.post a été appelé exactement 2 fois
    assert mock_post.call_count == 2
    
    # Vérification du premier appel (Inférence)
    first_call_url = mock_post.call_args_list[0][0][0]
    assert "predict" in first_call_url
    
    # Vérification du deuxième appel (Monitoring, envoyé en arrière-plan)
    second_call_url = mock_post.call_args_list[1][0][0]
    assert "log" in second_call_url

def test_predict_inference_failure_relayed(mocker):
    """Vérifie que si l'Inférence tombe en panne (ex: 503), le Preprocessing relaie l'erreur."""
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    
    mock_inference = mocker.patch("services.preprocessing.app.requests.post")
    mock_inference.return_value.status_code = 503
    mock_inference.return_value.text = "Service Unavailable"

    files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 503
    assert "Erreur du service d'inférence" in response.json()["detail"]

def test_predict_inference_network_crash(mocker):
    # On simule un crash réseau complet (Pas de code HTTP, juste une coupure)
    mock_post = mocker.patch("services.preprocessing.app.requests.post")
    mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

    import io
    from PIL import Image
    img = Image.new("RGB", (50, 50), color="yellow")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")

    files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
    response = client.post("/predict", files=files)
    
    # On vérifie que ton code attrape l'erreur proprement (500 ou 503 selon ton code)
    assert response.status_code in [500, 503, 400]