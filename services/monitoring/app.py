from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List

app = FastAPI(title="HAM10000 Monitoring Service")

# Stockage en mémoire des latences et des statuts pour rester ultra-léger
latencies: List[float] = []
status_codes: List[int] = []

class MetricPayload(BaseModel):
    status_code: int
    latency: float

@app.post("/log")
async def log_metric(payload: MetricPayload):
    """
    Endpoint appelé par le preprocessing ou l'inférence 
    pour enregistrer les résultats de chaque requête.
    """
    latencies.append(payload.latency)
    status_codes.append(payload.status_code)
    return {"status": "logged"}

@app.get("/metrics")
def get_metrics():
    """
    Endpoint GET /metrics exigé par le sujet pour la démo live.
    Calcule et expose le volume, le taux de succès, et les latences (Moyenne, P95).
    """
    total_requests = len(latencies)
    if total_requests == 0:
        return {
            "total_requests": 0,
            "success_rate_pct": 100.0,
            "latency_avg_s": 0.0,
            "latency_p95_s": 0.0,
            "message": "Aucune requête enregistrée pour le moment."
        }
    
    # Calcul des succès (HTTP 200)
    success_count = status_codes.count(200)
    success_rate = (success_count / total_requests) * 100
    
    # Calculs statistiques des latences
    latency_avg = np.mean(latencies)
    latency_p95 = np.percentile(latencies, 95)
    
    return {
        "total_requests": total_requests,
        "success_rate_pct": round(success_rate, 2),
        "latency_avg_s": round(float(latency_avg), 4),
        "latency_p95_s": round(float(latency_p95), 4),
        "http_200_count": success_count,
        "errors_count": total_requests - success_count
    }

@app.post("/reset")
def reset_metrics():
    """Pratique pour vider les métriques entre deux paliers du challenge de charge."""
    latencies.clear()
    status_codes.clear()
    return {"status": "metrics cleared"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}