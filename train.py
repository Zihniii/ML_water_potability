"""
Training script for Water Potability MLOps pipeline.

YAML-driven training with support for multiple classifiers, imputers,
preprocessors, transformers, and PCA. Supports grid search via optional
hyperparameter config.

Usage:
    python train.py
    python train.py --config config/train.yaml
    python train.py --config config/train.yaml --hyperparameters config/hyperparameters.yaml

Environment variables (override YAML values):
    DATASET_PATH            — path to CSV
    DATASET_VERSION         — version string from CI
    MLFLOW_TRACKING_URI     — MLflow server URI (required in cloud mode)
    MLFLOW_EXPERIMENT_NAME  — experiment name
    MODEL_NAME              — registered model name
    PROMOTION_METRIC        — metric used for model promotion
"""
import argparse
import json
import os
import sys
import time
import warnings
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional / conditional imports (handled gracefully on missing packages)
# ---------------------------------------------------------------------------
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
except ImportError:
    pass

from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
)
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

_LGBM_AVAILABLE = False
try:
    import lightgbm as lgbm

    _LGBM_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Valid options for config validation
# ---------------------------------------------------------------------------
VALID_CLASSIFIERS = {"ada_boost", "log_reg", "random_forest", "svc", "light_gbm"}
VALID_IMPUTERS = {"simple", "knn", "iterative"}
VALID_PREPROCESSORS = {"none", "std", "min_max", "norm", "poly", "robust"}
VALID_TRANSFORMERS = {"none", "power_box_cox", "power_yeo_johnson"}

# ---------------------------------------------------------------------------
# Default classifier hyperparameter search spaces (used when hyperparameter
# config is provided).   Keys match classifier_type values.
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMETER_SPACES: Dict[str, Dict[str, list]] = {
    "ada_boost": {
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "classifier__n_estimators": [50, 100, 200],
    },
    "log_reg": {
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
        "classifier__C": [0.1, 0.5, 1, 2],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [None, 10, 25],
        "classifier__min_samples_leaf": [1, 5],
        "classifier__min_samples_split": [2, 5],
    },
    "svc": {
        "classifier__C": [0.5, 1, 2],
        "classifier__kernel": ["linear", "rbf"],
    },
    "light_gbm": {
        "classifier__num_leaves": [31, 63],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__n_estimators": [100, 300],
        "classifier__reg_lambda": [0.1, 1.0],
        "classifier__min_data_in_leaf": [20, 50],
    },
}

DEFAULT_IMPUTER_PARAM_SPACES: Dict[str, Dict[str, list]] = {
    "simple": {"imputer__strategy": ["mean", "median"]},
    "knn": {
        "imputer__n_neighbors": [5, 7],
        "imputer__weights": ["uniform", "distance"],
    },
    "iterative": {"imputer__initial_strategy": ["mean", "median"]},
}

DEFAULT_PREPROCESSOR_PARAM_SPACES: Dict[str, Optional[Dict[str, list]]] = {
    "std": None,
    "min_max": None,
    "norm": {"preprocessor__norm": ["l1", "l2"]},
    "poly": {
        "preprocessor__degree": [2],
        "preprocessor__interaction_only": [True, False],
        "preprocessor__include_bias": [False],
    },
    "robust": None,
}

DEFAULT_PCA_PARAM_SPACES: Dict[str, list] = {}


# ===================================================================
# Config
# ===================================================================
class TrainConfig:
    """Holds all training configuration, loaded from YAML + env overrides."""

    def __init__(self, config_path: Optional[str] = None, hyperparam_path: Optional[str] = None):
        self.config_path = config_path
        self.hyperparam_path = hyperparam_path

        # Pipeline architecture (from YAML, overridable by env)
        self.file_csv: str = "dataset/water_potability.csv"
        self.target_column: str = "Potability"
        self.test_size: float = 0.2
        self.random_state: int = 42
        self.classifier_type: str = "random_forest"
        self.imputer_type: str = "simple"
        self.preprocessor_type: str = "none"
        self.transformer_type: str = "none"
        self.is_pca: bool = False
        self.use_blob_dataset: bool = False

        # MLflow / runtime (env first, then YAML)
        self.dataset_version: str = "local"
        self.mlflow_tracking_uri: str = ""
        self.mlflow_experiment_name: str = "water-potability-mlops"
        self.model_name: str = "water_potability_model"
        self.promotion_metric: str = "test_f1_score"

        # Hyperparameter search spaces (loaded from hyperparam_path or defaults)
        self.classifier_params: Optional[Dict[str, list]] = None
        self.imputer_params: Optional[Dict[str, list]] = None
        self.preprocessor_params: Optional[Dict[str, list]] = None
        self.pca_params: Optional[Dict[str, list]] = None
        self.is_grid_search: bool = False

        self._load_yaml()
        self._apply_env_overrides()

    def _load_yaml(self):
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            self.file_csv = data.get("file_csv", self.file_csv)
            self.target_column = data.get("target_column", self.target_column)
            self.test_size = float(data.get("test_size", self.test_size))
            self.random_state = int(data.get("random_state", self.random_state))
            self.classifier_type = data.get("classifier_type", self.classifier_type)
            self.imputer_type = data.get("imputer_type", self.imputer_type)
            self.preprocessor_type = data.get("preprocessor_type", self.preprocessor_type)
            self.transformer_type = data.get("transformer_type", self.transformer_type)
            self.is_pca = bool(data.get("is_pca", self.is_pca))
            self.use_blob_dataset = bool(data.get("use_blob_dataset", self.use_blob_dataset))

        # Try loading hyperparameter YAML
        if self.hyperparam_path and os.path.exists(self.hyperparam_path):
            self._load_hyperparameters()

    def _load_hyperparameters(self):
        with open(self.hyperparam_path, "r") as f:
            hp_data = yaml.safe_load(f) or {}

        clf_type = self.classifier_type
        if clf_type in hp_data:
            clf_section = hp_data[clf_type]
            if isinstance(clf_section, dict):
                prefixed = {}
                for k, v in clf_section.items():
                    prefixed[f"classifier__{k}"] = v
                self.classifier_params = prefixed
                self.is_grid_search = True

        # Imputer grid search config (optional)
        if self.imputer_type in hp_data:
            imputer_section = hp_data[self.imputer_type]
            if isinstance(imputer_section, dict):
                prefixed = {}
                for k, v in imputer_section.items():
                    prefixed[f"imputer__{k}"] = v
                self.imputer_params = prefixed
                self.is_grid_search = True

        # Preprocessor grid search config (optional)
        prep_type = self.preprocessor_type
        if prep_type in hp_data:
            prep_section = hp_data[prep_type]
            if isinstance(prep_section, dict):
                prefixed = {}
                for k, v in prep_section.items():
                    prefixed[f"preprocessor__{k}"] = v
                self.preprocessor_params = prefixed
                self.is_grid_search = True

        # PCA grid search config (optional)
        if "pca" in hp_data:
            pca_section = hp_data["pca"]
            if isinstance(pca_section, dict):
                prefixed = {}
                for k, v in pca_section.items():
                    prefixed[f"pca__{k}"] = v
                self.pca_params = prefixed
                self.is_grid_search = True

    def _apply_env_overrides(self):
        env_map = {
            "DATASET_PATH": "file_csv",
            "DATASET_VERSION": "dataset_version",
            "MLFLOW_TRACKING_URI": "mlflow_tracking_uri",
            "MLFLOW_EXPERIMENT_NAME": "mlflow_experiment_name",
            "MODEL_NAME": "model_name",
            "PROMOTION_METRIC": "promotion_metric",
        }
        for env_key, attr in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                setattr(self, attr, val)

    def validate(self):
        errors = []

        if self.classifier_type not in VALID_CLASSIFIERS:
            errors.append(
                f"Unknown classifier_type '{self.classifier_type}'. "
                f"Valid: {', '.join(sorted(VALID_CLASSIFIERS))}"
            )
        if self.classifier_type == "light_gbm" and not _LGBM_AVAILABLE:
            errors.append(
                "Classifier 'light_gbm' requires lightgbm package. "
                "Install with: pip install lightgbm"
            )
        if self.imputer_type not in VALID_IMPUTERS:
            errors.append(
                f"Unknown imputer_type '{self.imputer_type}'. "
                f"Valid: {', '.join(sorted(VALID_IMPUTERS))}"
            )
        if self.preprocessor_type not in VALID_PREPROCESSORS:
            errors.append(
                f"Unknown preprocessor_type '{self.preprocessor_type}'. "
                f"Valid: {', '.join(sorted(VALID_PREPROCESSORS))}"
            )
        if self.transformer_type not in VALID_TRANSFORMERS:
            errors.append(
                f"Unknown transformer_type '{self.transformer_type}'. "
                f"Valid: {', '.join(sorted(VALID_TRANSFORMERS))}"
            )
        if not os.path.exists(self.file_csv):
            errors.append(f"Dataset file not found: {self.file_csv}")
        if self.test_size <= 0 or self.test_size >= 1:
            errors.append(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.random_state < 0:
            errors.append(f"random_state must be non-negative, got {self.random_state}")

        if errors:
            print("Configuration errors:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)


# ===================================================================
# PipelineBuilder
# ===================================================================
class PipelineBuilder:
    """Dynamically builds an sklearn Pipeline from a TrainConfig."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self._imputer = None
        self._preprocessor = None
        self._transformer = None
        self._pca = None
        self._classifier = None
        self._steps: list = []
        self._param_grid: Dict[str, list] = {}

    def build(self) -> Tuple[Pipeline, Dict[str, list]]:
        self._build_imputer()
        self._build_preprocessor()
        self._build_transformer()
        self._build_pca()
        self._build_classifier()
        self._assemble()
        return Pipeline(self._steps), self._param_grid

    def _build_imputer(self):
        t = self.config.imputer_type
        if t == "simple":
            self._imputer = SimpleImputer()
        elif t == "knn":
            self._imputer = KNNImputer()
        elif t == "iterative":
            self._imputer = IterativeImputer(random_state=self.config.random_state)

        # Use provided params, or default search space
        params = self.config.imputer_params
        if params is None and self.config.is_grid_search:
            params = DEFAULT_IMPUTER_PARAM_SPACES.get(t)
        if params:
            self._param_grid.update(params)

    def _build_preprocessor(self):
        t = self.config.preprocessor_type
        if t == "std":
            self._preprocessor = StandardScaler()
        elif t == "min_max":
            self._preprocessor = MinMaxScaler(feature_range=(1, 2), clip=True)
        elif t == "norm":
            self._preprocessor = Normalizer()
        elif t == "poly":
            self._preprocessor = PolynomialFeatures()
        elif t == "robust":
            self._preprocessor = RobustScaler()

        params = self.config.preprocessor_params
        if params is None and self.config.is_grid_search and t != "none":
            space = DEFAULT_PREPROCESSOR_PARAM_SPACES.get(t)
            if space:
                self._param_grid.update(space)
        elif params:
            self._param_grid.update(params)

    def _build_transformer(self):
        t = self.config.transformer_type
        if t == "power_box_cox":
            self._transformer = PowerTransformer(method="box-cox")
            # Box-Cox requires positive values — auto-add MinMax
            if self.config.preprocessor_type == "none":
                self._preprocessor = MinMaxScaler(feature_range=(1, 2), clip=True)
        elif t == "power_yeo_johnson":
            self._transformer = PowerTransformer(method="yeo-johnson")
            # Yeo-Johnson works with negatives — auto-add StandardScaler
            if self.config.preprocessor_type == "none":
                self._preprocessor = StandardScaler()

    def _build_pca(self):
        if self.config.is_pca:
            self._pca = PCA()
            params = self.config.pca_params
            if params:
                self._param_grid.update(params)

    def _build_classifier(self):
        t = self.config.classifier_type
        rs = self.config.random_state

        if t == "ada_boost":
            self._classifier = AdaBoostClassifier(algorithm="SAMME", random_state=rs)
        elif t == "log_reg":
            self._classifier = LogisticRegression(max_iter=500, solver="saga", random_state=rs)
        elif t == "random_forest":
            self._classifier = RandomForestClassifier(random_state=rs, n_jobs=-1)
        elif t == "svc":
            self._classifier = SVC(probability=True)
        elif t == "light_gbm":
            self._classifier = lgbm.LGBMClassifier(
                boosting_type="gbdt", objective="binary", verbosity=-1, random_state=rs
            )

        params = self.config.classifier_params
        if params is None and self.config.is_grid_search:
            params = DEFAULT_HYPERPARAMETER_SPACES.get(t)
        if params:
            self._param_grid.update(params)

    def _assemble(self):
        self._steps.append(("imputer", self._imputer))
        if self._preprocessor is not None:
            self._steps.append(("preprocessor", self._preprocessor))
        if self._transformer is not None:
            self._steps.append(("transformer", self._transformer))
        if self._pca is not None:
            self._steps.append(("pca", self._pca))
        self._steps.append(("classifier", self._classifier))


# ===================================================================
# Training logic
# ===================================================================
def _load_dataset(file_csv: str, target_column: str, test_size: float, random_state: int):
    df = pd.read_csv(file_csv)
    row_count, feature_count = df.shape
    print(f"Dataset shape: {row_count} rows, {feature_count} columns")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, row_count, feature_count


def _evaluate(pipeline, X_train, y_train, X_test, y_test):
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": round(float(accuracy_score(y_train, y_train_pred)), 4),
        "train_f1_score": round(float(f1_score(y_train, y_train_pred, zero_division=0)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_test_pred)), 4),
        "test_f1_score": round(float(f1_score(y_test, y_test_pred, zero_division=0)), 4),
        "test_precision": round(float(precision_score(y_test, y_test_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_test_pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_test_proba)), 4),
    }
    return metrics


def _build_model_metadata(
    config: TrainConfig,
    pipeline,
    metrics: Dict[str, float],
    row_count: int,
    feature_count: int,
    cv_best_params: Optional[Dict[str, Any]] = None,
    training_duration_seconds: float = 0.0,
) -> Dict[str, Any]:
    meta = {
        "dataset_path": config.file_csv,
        "dataset_version": config.dataset_version,
        "classifier_type": config.classifier_type,
        "imputer_type": config.imputer_type,
        "preprocessor_type": config.preprocessor_type,
        "transformer_type": config.transformer_type,
        "is_pca": config.is_pca,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "row_count": row_count,
        "feature_count": feature_count - 1,
        "training_started_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(training_duration_seconds, 2),
        "metrics": metrics,
    }
    if cv_best_params:
        meta["best_cv_params"] = {str(k): str(v) for k, v in cv_best_params.items()}
    return meta


def _log_to_mlflow(
    config: TrainConfig,
    pipeline,
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    feature_names: list[str],
    cv_best_params: Optional[Dict[str, Any]] = None,
) -> str:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    if experiment is None:
        mlflow.create_experiment(config.mlflow_experiment_name)

    run_name = f"train_{config.dataset_version}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log pipeline config params
        mlflow.log_param("dataset_path", config.file_csv)
        mlflow.log_param("dataset_version", config.dataset_version)
        mlflow.log_param("classifier_type", config.classifier_type)
        mlflow.log_param("imputer_type", config.imputer_type)
        mlflow.log_param("preprocessor_type", config.preprocessor_type)
        mlflow.log_param("transformer_type", config.transformer_type)
        mlflow.log_param("is_pca", str(config.is_pca))
        mlflow.log_param("test_size", str(config.test_size))
        mlflow.log_param("random_state", str(config.random_state))
        mlflow.log_param("row_count", metadata["row_count"])
        mlflow.log_param("feature_count", metadata["feature_count"])

        if cv_best_params:
            mlflow.log_params({str(k): str(v) for k, v in cv_best_params.items()})

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log feature importance for tree-based classifiers
        clf = pipeline.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "feature_importances_"):
            for name, imp in zip(feature_names, clf.feature_importances_):
                mlflow.log_metric(f"feat_imp_{name}", round(float(imp), 4))

        # Build and log model signature
        input_schema = mlflow.types.Schema(
            [mlflow.types.ColSpec("double", name) for name in feature_names]
        )
        output_schema = mlflow.types.Schema(
            [mlflow.types.ColSpec("long", "prediction")]
        )
        signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)

        # Save and log metadata artifact
        os.makedirs("outputs", exist_ok=True)
        meta_path = "outputs/model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(meta_path)

        # Log model and register
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=config.model_name,
            signature=signature,
        )
        print(f"Model registered as '{config.model_name}'")

    return run_id


# ===================================================================
# Model promotion (Phase 5)
# ===================================================================
def _promote_if_better(
    config: TrainConfig,
    current_metrics: Dict[str, float],
) -> None:
    """Compare current run's metric against Production model and promote if better."""
    if not config.mlflow_tracking_uri:
        return

    try:
        client = mlflow.MlflowClient()
        latest_prod = client.get_latest_versions(config.model_name, stages=["Production"])

        if not latest_prod:
            print("No Production model found - promoting current run to Production.")
            _transition_latest_to_stage(client, config.model_name, "Production")
            return

        prod_version = latest_prod[0]
        prod_run_id = prod_version.run_id

        prod_metric = mlflow.get_run(prod_run_id).data.metrics.get(config.promotion_metric)
        current_metric = current_metrics.get(config.promotion_metric)

        if prod_metric is None:
            print(f"No {config.promotion_metric} found on Production model - promoting.")
            _transition_latest_to_stage(client, config.model_name, "Production")
            return

        print(f"\nModel promotion check:")
        print(f"  Production {config.promotion_metric}: {prod_metric:.4f}")
        print(f"  Current    {config.promotion_metric}: {current_metric:.4f}")

        if current_metric is not None and current_metric >= prod_metric:
            print("  Current model is better or equal - promoting to Production.")
            _transition_latest_to_stage(client, config.model_name, "Production")
        else:
            print("  Production model is better - skipping promotion.")

    except Exception as e:
        print(f"Model promotion skipped ({e})")


def _transition_latest_to_stage(client, model_name: str, stage: str):
    """Transition the latest registered model version to the given stage."""
    latest = client.get_latest_versions(model_name, stages=["None"])
    if not latest:
        print(f"No un-staged versions found for '{model_name}'.")
        return

    latest_version = latest[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage=stage,
    )
    print(f"  Model version {latest_version} -> {stage}")

    # Archive previous Production version (if any and if promoting to Production)
    if stage == "Production":
        all_prod = client.get_latest_versions(model_name, stages=["Production"])
        for v in all_prod:
            if v.version != latest_version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )
                print(f"  Archived previous Production model version {v.version}")


# ===================================================================
# Main
# ===================================================================
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Water Potability MLOps Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (default: env vars only)",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="Path to hyperparameter search space YAML (optional)",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config = TrainConfig(config_path=args.config, hyperparam_path=args.hyperparameters)

    # Allow hyperparameters to trigger grid search even without --hyperparameters flag
    # if a file exists at the default path
    if args.hyperparameters is None and config.is_grid_search:
        pass  # already loaded from default path via config init

    config.validate()

    if not config.mlflow_tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Classifier:  {config.classifier_type}")
    print(f"Imputer:     {config.imputer_type}")
    print(f"Preprocessor:{config.preprocessor_type}")
    print(f"Transformer: {config.transformer_type}")
    print(f"PCA:         {config.is_pca}")
    print(f"Grid search: {config.is_grid_search}")
    print(f"Dataset:     {config.file_csv}")
    print(f"Version:     {config.dataset_version}")
    print(f"{'='*60}\n")

    # ---- Download dataset from Blob if enabled ----
    if config.use_blob_dataset:
        from dataset_manager import DatasetManager

        dm = DatasetManager()
        if dm.is_azure_connected:
            version = dm.get_latest_version()
            if version:
                config.file_csv = dm.download_latest(dest_dir="dataset")[0]
                if config.dataset_version == "local":
                    config.dataset_version = version
                print(f"Using blob dataset version: {version}")
            else:
                print("No blob datasets found, falling back to local file")
        else:
            print("Azure Blob not connected, using local file")

    # ---- Load dataset ----
    X_train, X_test, y_train, y_test, row_count, feature_count = _load_dataset(
        config.file_csv, config.target_column, config.test_size, config.random_state
    )

    # ---- Build pipeline ----
    builder = PipelineBuilder(config)
    pipeline, param_grid = builder.build()

    print(f"Pipeline: {pipeline}")
    if param_grid:
        print(f"Search space: {len(param_grid)} parameter(s)")
        for k, v in param_grid.items():
            print(f"  {k}: {v}")

    # ---- Train (with or without grid search) ----
    cv_best_params = None
    train_start = time.time()

    if config.is_grid_search and param_grid:
        from sklearn.model_selection import GridSearchCV, KFold

        k_fold = KFold(n_splits=5, shuffle=True, random_state=config.random_state)
        grid = GridSearchCV(
            pipeline, param_grid, scoring="f1", cv=k_fold, n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        cv_best_params = grid.best_params_
        print(f"\nBest CV F1: {grid.best_score_:.4f}")
        print(f"Best params: {cv_best_params}")
    else:
        print("Training pipeline (no grid search)...")
        pipeline.fit(X_train, y_train)

    train_end = time.time()
    training_duration = train_end - train_start
    print(f"Training took {training_duration:.2f} seconds")

    # ---- Evaluate ----
    metrics = _evaluate(pipeline, X_train, y_train, X_test, y_test)
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- Build metadata ----
    metadata = _build_model_metadata(
        config, pipeline, metrics, row_count, feature_count, cv_best_params, training_duration
    )

    # ---- Log to MLflow ----
    feature_names = list(X_train.columns)
    run_id = _log_to_mlflow(config, pipeline, metrics, metadata, feature_names, cv_best_params)

    # ---- Save model for Docker deployment ----
    import joblib
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(pipeline, "outputs/model.joblib")
    with open("outputs/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved to outputs/model.joblib")

    # ---- Model promotion ----
    _promote_if_better(config, metrics)

    print(f"\n{'='*60}")
    print(f"Training complete. Run ID: {run_id}")
    print(f"{'='*60}")
    return run_id


if __name__ == "__main__":
    main()
