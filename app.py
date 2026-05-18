import logging
from typing import Optional

import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from config import settings

# =========================================================
# Load MLflow model
# =========================================================
try:
    path_mlflow_model = "./model_for_production/"
    sklearn_pipeline = mlflow.sklearn.load_model(path_mlflow_model)
    print(f"Successfully loaded model from: {path_mlflow_model}")

except Exception as e:
    print(f"Local model load failed: {e}")

    try:
        path_mlflow_model = "/data/model_for_production/"
        sklearn_pipeline = mlflow.sklearn.load_model(path_mlflow_model)
        print(f"Successfully loaded model from: {path_mlflow_model}")

    except Exception as e:
        print(f"Docker model load failed: {e}")
        sklearn_pipeline = None


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(
    title="Water Potability API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


# =========================================================
# Request schema
# =========================================================
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


# =========================================================
# Prediction helper
# =========================================================
def predict_pipeline(data_sample):
    pred_sample = sklearn_pipeline.predict(data_sample)
    return pred_sample


# =========================================================
# Root endpoint
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Water Potability API is running",
        "docs": "/docs",
    }


# =========================================================
# Info endpoint
# =========================================================
@app.get("/info")
def get_app_info():
    return {
        "app_name": settings.app_name,
        "version": settings.version,
    }


# =========================================================
# Prediction endpoint
# =========================================================
@app.post("/predict")
def predict(wpd_item: WaterPotabilityDataItem):

    if sklearn_pipeline is None:
        return {
            "error": "Model failed to load"
        }

    wpd_arr = np.array(
        [
            wpd_item.ph,
            wpd_item.Hardness,
            wpd_item.Solids,
            wpd_item.Chloramines,
            wpd_item.Sulfate,
            wpd_item.Conductivity,
            wpd_item.Organic_carbon,
            wpd_item.Trihalomethanes,
            wpd_item.Turbidity,
        ],
        dtype=float,
    ).reshape(1, -1)

    logging.info(f"Input sample: {wpd_arr}")

    pred_sample = predict_pipeline(wpd_arr)

    logging.info(f"Prediction result: {pred_sample}")

    return {
        "Potability": int(pred_sample[0])
    }