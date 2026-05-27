import pytest
from fastapi.testclient import TestClient
from services.monitoring.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    client.post("/reset")
    yield
    client.post("/reset")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_metrics_empty():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 0
    assert data["success_rate_pct"] == 100.0
    assert "Aucune requête enregistrée" in data["message"]

def test_log_and_metrics_aggregation():
    response_1 = client.post("/log", json={"status_code": 200, "latency": 0.100})
    assert response_1.status_code == 200
    assert response_1.json() == {"status": "logged"}

    response_2 = client.post("/log", json={"status_code": 500, "latency": 0.200})
    assert response_2.status_code == 200

    response_metrics = client.get("/metrics")
    assert response_metrics.status_code == 200
    
    data = response_metrics.json()
    assert data["total_requests"] == 2
    assert data["success_rate_pct"] == 50.0
    assert data["http_200_count"] == 1
    assert data["errors_count"] == 1
    assert data["latency_avg_s"] == 0.1500
    assert data["latency_p95_s"] > 0