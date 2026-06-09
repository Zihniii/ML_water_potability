"""
FastAPI application for Water Potability prediction.

Loads model from MLflow Model Registry and serves:
    GET  /
    POST /predict
    POST /predict-with-stats
    GET  /prediction-stats
    GET  /model-info

Environment variables:
    MLFLOW_TRACKING_URI  — MLflow server URI (cloud)
    MODEL_NAME           — registered model name (default: water_potability_model)
    MODEL_STAGE_OR_ALIAS — stage to fetch (default: Production)
    DATASET_VERSION      — dataset version deployed
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import mlflow
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

# ------------------------------------------------------------------
# In-memory prediction stats
# NOTE: For production, replace with database or Azure Table Storage.
# ------------------------------------------------------------------
_prediction_stats = {
    "total_requests_since_start": 0,
    "potable_count": 0,
    "not_potable_count": 0,
    "last_prediction_at": None,
}

# ------------------------------------------------------------------
# Load model from MLflow Model Registry
# ------------------------------------------------------------------
sklearn_pipeline = None
_model_info = {
    "model_name": MODEL_NAME,
    "model_stage_or_alias": MODEL_STAGE_OR_ALIAS,
    "mlflow_run_id": None,
    "dataset_version": DATASET_VERSION,
    "loaded_model_uri": None,
    "api_started_timezone": "Asia/Jakarta",
}

if MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()

        # Get latest model version in the specified stage
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE_OR_ALIAS])
        if latest_versions:
            model_version = latest_versions[0]
            model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
            sklearn_pipeline = mlflow.sklearn.load_model(model_uri)

            _model_info["mlflow_run_id"] = model_version.run_id
            _model_info["loaded_model_uri"] = model_uri

            # Try to load dataset_version from the run's model_metadata artifact
            try:
                artifact_uri = f"runs:/{model_version.run_id}/model_metadata.json"
                metadata = mlflow.artifacts.load_dict(artifact_uri)
                ds_version = metadata.get("dataset_version")
                if ds_version:
                    _model_info["dataset_version"] = ds_version
            except Exception:
                pass  # fallback to env var

            print(f"Loaded model: {model_uri}")
        else:
            print(f"No model found for '{MODEL_NAME}' in stage '{MODEL_STAGE_OR_ALIAS}'")
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
else:
    print("MLFLOW_TRACKING_URI not set — model will not be loaded.")

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


class PredictionStatsResponse(BaseModel):
    total_requests_since_start: int
    potable_count: int
    not_potable_count: int
    potable_percentage: float
    not_potable_percentage: float
    last_prediction_at: Optional[str]


# ------------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------------
def _predict_item(item: WaterPotabilityDataItem) -> int:
    """Run pipeline on a single input item and return prediction (0 or 1)."""
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
    return int(sklearn_pipeline.predict(arr)[0])


def _update_stats(prediction: int):
    """Update in-memory prediction counters."""
    _prediction_stats["total_requests_since_start"] += 1
    if prediction == 1:
        _prediction_stats["potable_count"] += 1
    else:
        _prediction_stats["not_potable_count"] += 1
    _prediction_stats["last_prediction_at"] = (
        datetime.now(timezone.utc).isoformat()
    )


def _get_stats() -> dict:
    s = _prediction_stats
    total = s["total_requests_since_start"]
    potable_pct = (s["potable_count"] / total * 100) if total > 0 else 0.0
    not_potable_pct = (s["not_potable_count"] / total * 100) if total > 0 else 0.0
    return {
        "total_requests_since_start": total,
        "potable_count": s["potable_count"],
        "not_potable_count": s["not_potable_count"],
        "potable_percentage": round(potable_pct, 2),
        "not_potable_percentage": round(not_potable_pct, 2),
        "last_prediction_at": s["last_prediction_at"],
    }


# ------------------------------------------------------------------
# Jakarta time helper
# ------------------------------------------------------------------
def _jakarta_timestamp() -> str:
    """Return current timestamp in Asia/Jakarta timezone."""
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
        "dataset_version": _model_info["dataset_version"],
    }


@app.post("/predict")
def predict(item: WaterPotabilityDataItem):
    """Predict potability for a single water sample."""
    if sklearn_pipeline is None:
        return {"error": "Model not loaded"}

    pred = _predict_item(item)
    _update_stats(pred)

    return {
        "prediction": pred,
        "label": "Potable" if pred == 1 else "Not Potable",
    }


@app.post("/predict-with-stats")
def predict_with_stats(item: WaterPotabilityDataItem):
    """
    Predict and return enriched response with model metadata
    and cumulative prediction statistics.
    """
    if sklearn_pipeline is None:
        return {"error": "Model not loaded"}

    pred = _predict_item(item)
    _update_stats(pred)

    return {
        "prediction": pred,
        "label": "Potable" if pred == 1 else "Not Potable",
        "input_timestamp": _jakarta_timestamp(),
        "model_name": _model_info["model_name"],
        "model_stage_or_alias": _model_info["model_stage_or_alias"],
        "mlflow_run_id": _model_info["mlflow_run_id"],
        "dataset_version": _model_info["dataset_version"],
        "prediction_stats": _get_stats(),
    }


@app.get("/prediction-stats", response_model=PredictionStatsResponse)
def prediction_stats():
    """
    Return cumulative prediction statistics since container start.

    NOTE: For production, store stats in a database or Azure Table Storage
    to persist across container restarts.
    """
    return _get_stats()


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name": _model_info["model_name"],
        "model_stage_or_alias": _model_info["model_stage_or_alias"],
        "mlflow_run_id": _model_info["mlflow_run_id"],
        "dataset_version": _model_info["dataset_version"],
        "loaded_model_uri": _model_info["loaded_model_uri"],
        "api_started_timezone": _model_info["api_started_timezone"],
    }
