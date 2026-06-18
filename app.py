"""
FastAPI application for Water Potability prediction.

Loads model from Blob Storage (latest), MLflow Model Registry, or local fallback.
    GET  /
    GET  /health
    POST /predict
    POST /predict-with-stats
    GET  /prediction-stats
    GET  /model-info

Environment variables:
    MLFLOW_TRACKING_URI  — MLflow server URI (cloud)
    MODEL_NAME           — registered model name (default: water_potability_model)
    MODEL_STAGE_OR_ALIAS — stage to fetch (default: Production)
    DATASET_VERSION      — dataset version deployed
    AZURE_STORAGE_ACCOUNT — storage account name (for Blob model download)
    AZURE_STORAGE_KEY     — storage account key (for Blob model download)
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------------------------
# Configuration from environment
# ------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MODEL_NAME = os.getenv("MODEL_NAME", "water_potability_model")
MODEL_STAGE_OR_ALIAS = os.getenv("MODEL_STAGE_OR_ALIAS", "Production")
DATASET_VERSION = os.getenv("DATASET_VERSION", "unknown")
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "")
STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY", "")
STORAGE_CONTAINER = "datasets"

# ------------------------------------------------------------------
# In-memory prediction stats
# NOTE: For production, replace with database or Azure Table Storage.
# ------------------------------------------------------------------
_prediction_stats = {
    "total_requests_since_start": 0,
    "potable_count": 0,
    "not_potable_count": 0,
    "probability_sum": 0.0,
    "last_prediction_at": None,
}
_daily_counts: dict = {}
_model_version_counts: dict = {}

# ------------------------------------------------------------------
# Load model from Blob Storage (latest) or local fallback
# ------------------------------------------------------------------
sklearn_pipeline = None
_model_info = {
    "model_name": MODEL_NAME,
    "model_stage_or_alias": MODEL_STAGE_OR_ALIAS,
    "dataset_version": DATASET_VERSION,
    "loaded_model_uri": None,
    "api_started_timezone": "Asia/Jakarta",
}

def _download_from_blob(blob_path: str, local_path: str) -> bool:
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={STORAGE_KEY};EndpointSuffix=core.windows.net"
        client = BlobServiceClient.from_connection_string(conn_str)
        blob = client.get_blob_client(container=STORAGE_CONTAINER, blob=blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            blob.download_blob().readinto(f)
        return True
    except Exception as e:
        print(f"Blob download failed ({blob_path}): {e}")
        return False

if STORAGE_ACCOUNT and STORAGE_KEY:
    model_path = "outputs/model.joblib"
    meta_path = "outputs/model_metadata.json"
    if _download_from_blob("models/latest/model.joblib", model_path):
        try:
            sklearn_pipeline = joblib.load(model_path)
            _model_info["loaded_model_uri"] = f"blob://{STORAGE_CONTAINER}/models/latest/model.joblib"
            if _download_from_blob("models/latest/model_metadata.json", meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                _model_info["dataset_version"] = meta.get("dataset_version", DATASET_VERSION)
            print(f"Loaded model from Blob: models/latest/model.joblib")
        except Exception as e:
            print(f"Failed to load model from Blob: {e}")
            sklearn_pipeline = None
else:
    print("AZURE_STORAGE_ACCOUNT not set — skipping Blob download.")

# Fallback: load local model.joblib
if sklearn_pipeline is None:
    local_model_path = "outputs/model.joblib"
    if os.path.exists(local_model_path):
        try:
            sklearn_pipeline = joblib.load(local_model_path)
            _model_info["loaded_model_uri"] = local_model_path
            meta_path = "outputs/model_metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                _model_info["dataset_version"] = meta.get("dataset_version", DATASET_VERSION)
            print(f"Loaded model from local file: {local_model_path}")
        except Exception as e:
            print(f"Failed to load local model: {e}")
    else:
        print(f"Local model file not found at: {local_model_path}")

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="Water Potability API",
    version="1.0.0",
    description="MLOps pipeline — FastAPI with MLflow Cloud + Azure",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prediction logger
# ------------------------------------------------------------------
from prediction_logger import create_prediction_logger

_prediction_logger = create_prediction_logger()


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------
class WaterPotabilityDataItem(BaseModel):
    ph: Optional[float] = None
    Hardness: Optional[float] = None
    Solids: Optional[float] = None
    Chloramines: Optional[float] = None
    Sulfate: Optional[float] = None
    Conductivity: Optional[float] = None
    Organic_carbon: Optional[float] = None
    Trihalomethanes: Optional[float] = None
    Turbidity: Optional[float] = None


# ------------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------------
def _predict_item(item: WaterPotabilityDataItem):
    """Run pipeline on a single input item.

    Returns:
        (prediction: int, probability: float)
    """
    arr = np.array(
        [
            item.ph,
            item.Hardness,
            item.Solids,
            item.Chloramines,
            item.Sulfate,
            item.Conductivity,
            item.Organic_carbon,
            item.Trihalomethanes,
            item.Turbidity,
        ],
        dtype=float,
    ).reshape(1, -1)

    pred = int(sklearn_pipeline.predict(arr)[0])

    try:
        proba = sklearn_pipeline.predict_proba(arr)[0]
        probability = float(proba[1])  # probability of class 1 (Potable)
    except Exception:
        probability = float(pred)  # fallback if no predict_proba

    return pred, probability


def _confidence_label(probability: float) -> str:
    if probability >= 0.9:
        return "High"
    elif probability >= 0.7:
        return "Medium"
    else:
        return "Low"


def _update_stats(prediction: int, probability: float):
    _prediction_stats["total_requests_since_start"] += 1
    if prediction == 1:
        _prediction_stats["potable_count"] += 1
    else:
        _prediction_stats["not_potable_count"] += 1
    _prediction_stats["probability_sum"] += probability
    now = datetime.now(timezone.utc)
    _prediction_stats["last_prediction_at"] = now.isoformat()

    day_key = now.strftime("%Y-%m-%d")
    _daily_counts[day_key] = _daily_counts.get(day_key, 0) + 1

    mv = _model_version_number or 0
    _model_version_counts[mv] = _model_version_counts.get(mv, 0) + 1


def _get_stats() -> dict:
    s = _prediction_stats
    total = s["total_requests_since_start"]
    potable_pct = (s["potable_count"] / total * 100) if total > 0 else 0.0
    not_potable_pct = (s["not_potable_count"] / total * 100) if total > 0 else 0.0
    avg_confidence = (s["probability_sum"] / total) if total > 0 else 0.0
    return {
        "total_requests_since_start": total,
        "potable_count": s["potable_count"],
        "not_potable_count": s["not_potable_count"],
        "potable_percentage": round(potable_pct, 2),
        "not_potable_percentage": round(not_potable_pct, 2),
        "avg_confidence": round(avg_confidence, 4),
        "current_model_version": _model_version_number,
        "daily_counts": dict(sorted(_daily_counts.items())),
        "model_version_counts": dict(sorted(_model_version_counts.items())),
        "last_prediction_at": s["last_prediction_at"],
    }


# ------------------------------------------------------------------
# Jakarta time helper
# ------------------------------------------------------------------
def _jakarta_timestamp() -> str:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Asia/Jakarta")
        return datetime.now(tz).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


# ==================================================================
# ENDPOINTS
# ==================================================================

@app.get("/")
def root():
    """Root — service status."""
    return {
        "status": "running",
        "service": "water-potability-api",
        "model_name": _model_info["model_name"],
        "model_version": _model_info["model_version"],
        "dataset_version": _model_info["dataset_version"],
    }


@app.get("/health")
def health():
    """Health check — returns model load status."""
    return {
        "status": "healthy" if sklearn_pipeline is not None else "degraded",
        "model_loaded": sklearn_pipeline is not None,
        "model_version": _model_info["model_version"],
    }


@app.post("/predict")
def predict(item: WaterPotabilityDataItem):
    """Predict potability for a single water sample.

    Returns enhanced response with prediction label, probability,
    confidence level, model version, and timestamp.
    """
    if sklearn_pipeline is None:
        return {"error": "Model not loaded"}

    pred, probability = _predict_item(item)
    _update_stats(pred, probability)
    _prediction_logger.log_prediction(
        timestamp=_jakarta_timestamp(),
        prediction=pred,
        probability=probability,
        confidence=_confidence_label(probability),
        model_version=_model_version_number or 0,
        input_data=item.model_dump(),
    )

    return {
        "prediction": pred,
        "label": "Potable" if pred == 1 else "Not Potable",
        "probability": round(probability, 4),
        "confidence": _confidence_label(probability),
        "model_version": _model_version_number,
        "timestamp": _jakarta_timestamp(),
    }


@app.post("/predict-with-stats")
def predict_with_stats(item: WaterPotabilityDataItem):
    """Predict and return enriched response with model metadata
    and cumulative prediction statistics.
    """
    if sklearn_pipeline is None:
        return {"error": "Model not loaded"}

    pred, probability = _predict_item(item)
    _update_stats(pred, probability)
    _prediction_logger.log_prediction(
        timestamp=_jakarta_timestamp(),
        prediction=pred,
        probability=probability,
        confidence=_confidence_label(probability),
        model_version=_model_version_number or 0,
        input_data=item.model_dump(),
    )

    return {
        "prediction": pred,
        "label": "Potable" if pred == 1 else "Not Potable",
        "probability": round(probability, 4),
        "confidence": _confidence_label(probability),
        "timestamp": _jakarta_timestamp(),
        "input_timestamp": _jakarta_timestamp(),
        "model_name": _model_info["model_name"],
        "dataset_version": _model_info["dataset_version"],
        "prediction_stats": _get_stats(),
    }


@app.get("/prediction-stats")
def prediction_stats():
    """Return cumulative prediction statistics since container start.

    NOTE: For production, store stats in a database or Azure Table Storage
    to persist across container restarts.
    """
    return _get_stats()


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name": _model_info["model_name"],
        "dataset_version": _model_info["dataset_version"],
        "loaded_model_uri": _model_info["loaded_model_uri"],
        "api_started_timezone": _model_info["api_started_timezone"],
    }
