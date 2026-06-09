"""
Training script for Water Potability MLOps pipeline.

Reads dataset, trains a RandomForestClassifier with SimpleImputer,
logs metrics and model to MLflow cloud, registers model.

Environment variables:
    DATASET_PATH        — path to CSV (default: dataset/water_potability.csv)
    DATASET_VERSION     — version string passed from CI
    MLFLOW_TRACKING_URI — MLflow server URI (cloud)
    MLFLOW_EXPERIMENT_NAME — MLflow experiment name
    MODEL_NAME          — registered model name
"""
import json
import os
import sys
from datetime import datetime, timezone

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATASET_PATH = os.getenv("DATASET_PATH", "dataset/water_potability.csv")
DATASET_VERSION = os.getenv("DATASET_VERSION", "local")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "water-potability-mlops")
MODEL_NAME = os.getenv("MODEL_NAME", "water_potability_model")
TARGET_COLUMN = "Potability"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------------
    # Validate MLflow is configured
    # ------------------------------------------------------------------
    if not MLFLOW_TRACKING_URI:
        print("ERROR: MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)

    training_started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Reading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    row_count, feature_count = df.shape
    print(f"Dataset shape: {row_count} rows, {feature_count} columns")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # Build pipeline: SimpleImputer + RandomForest
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
        )),
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # ------------------------------------------------------------------
    # Build metadata
    # ------------------------------------------------------------------
    model_metadata = {
        "dataset_path": DATASET_PATH,
        "dataset_version": DATASET_VERSION,
        "training_started_at": training_started_at,
        "training_completed_at": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "feature_count": feature_count - 1,  # exclude target
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "class_weight": "balanced",
        "imputer_strategy": "median",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        },
    }

    # Save metadata to file (for artifact logging)
    os.makedirs("outputs", exist_ok=True)
    metadata_path = "outputs/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train_{DATASET_VERSION}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # -- Log params --
        mlflow.log_param("dataset_path", DATASET_PATH)
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("training_started_at", training_started_at)
        mlflow.log_param("row_count", row_count)
        mlflow.log_param("feature_count", feature_count - 1)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # -- Log metrics --
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # -- Log model_metadata.json as artifact --
        mlflow.log_artifact(metadata_path)

        # -- Log model and register --
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Model registered as '{MODEL_NAME}'")

    print(f"\nTraining complete. Run ID: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
