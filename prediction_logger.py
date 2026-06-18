"""
Prediction logging abstraction.

Stores prediction records (timestamp, prediction, probability, confidence,
model version) to Azure Blob Storage, with a local fallback.

Each prediction is a JSON line appended to a date-partitioned blob:
    predictions/<YYYY>/<MM>/<DD>/predictions.jsonl

Environment variables:
    AZURE_STORAGE_CONNECTION_STRING  — for Azure Blob logging
    AZURE_STORAGE_ACCOUNT           — alternative auth
    AZURE_STORAGE_CONTAINER         — container name (default: predictions)
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PREDICTIONS_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "datasets")


class PredictionLogger(ABC):
    """Abstract base for prediction loggers."""

    @abstractmethod
    def log_prediction(
        self,
        timestamp: str,
        prediction: int,
        probability: float,
        confidence: str,
        model_version: int,
        input_data: dict,
    ):
        ...


class BlobPredictionLogger(PredictionLogger):
    """Logs predictions as JSON Lines to Azure Blob Storage.

    Path: predictions/<YYYY>/<MM>/<DD>/predictions.jsonl
    """

    def __init__(self):
        self._container_client = None
        self._use_azure = False
        self._init_azure()

    def _init_azure(self):
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
        account_key = os.getenv("AZURE_STORAGE_KEY")
        container_name = PREDICTIONS_CONTAINER

        try:
            if conn_str:
                from azure.storage.blob import BlobServiceClient
                client = BlobServiceClient.from_connection_string(conn_str)
                self._container_client = client.get_container_client(container_name)
                self._use_azure = True
            elif account_name and account_key:
                from azure.storage.blob import BlobServiceClient
                conn_str = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
                client = BlobServiceClient.from_connection_string(conn_str)
                self._container_client = client.get_container_client(container_name)
                self._use_azure = True
            elif account_name:
                from azure.identity import DefaultAzureCredential
                from azure.storage.blob import BlobServiceClient
                credential = DefaultAzureCredential()
                account_url = f"https://{account_name}.blob.core.windows.net"
                client = BlobServiceClient(account_url, credential=credential)
                self._container_client = client.get_container_client(container_name)
                self._use_azure = True
        except Exception as e:
            logger.warning("BlobPredictionLogger init failed: %s", e)

        if not self._use_azure:
            logger.info("BlobPredictionLogger: local-only mode (no Azure credentials)")

    def log_prediction(
        self,
        timestamp: str,
        prediction: int,
        probability: float,
        confidence: str,
        model_version: int,
        input_data: dict,
    ):
        record = {
            "timestamp": timestamp,
            "prediction": prediction,
            "probability": round(probability, 4),
            "confidence": confidence,
            "model_version": model_version,
            "input": {k: v for k, v in input_data.items() if v is not None},
        }
        line = json.dumps(record, default=str) + "\n"

        if self._use_azure:
            try:
                now = datetime.now(timezone.utc)
                blob_path = (
                    f"predictions/{now.strftime('%Y/%m/%d')}/predictions.jsonl"
                )
                blob_client = self._container_client.get_blob_client(blob_path)

                try:
                    existing = blob_client.download_blob().readall()
                    blob_client.upload_blob(existing + line.encode(), overwrite=True)
                except Exception:
                    blob_client.upload_blob(line.encode(), overwrite=True)
            except Exception as e:
                logger.error("Failed to log prediction to Blob: %s", e)
        else:
            logger.debug("Prediction: %s", line.strip())


class LoggingPredictionLogger(PredictionLogger):
    """Fallback logger — writes predictions to Python logger."""

    def log_prediction(
        self,
        timestamp: str,
        prediction: int,
        probability: float,
        confidence: str,
        model_version: int,
        input_data: dict,
    ):
        logger.info(
            "pred=%s prob=%.4f conf=%s model_v=%s ts=%s",
            prediction, probability, confidence, model_version, timestamp,
        )


def create_prediction_logger() -> PredictionLogger:
    """Factory: returns BlobPredictionLogger if Azure is configured,
    otherwise LoggingPredictionLogger."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    if conn_str or account_name:
        return BlobPredictionLogger()
    return LoggingPredictionLogger()
