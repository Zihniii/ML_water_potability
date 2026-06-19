# Water Potability MLOps — Final Presentation Package

> Week 5 Final Presentation | MLOps Pipeline | Full Stack ML Engineering

---

## 1. PROJECT UNDERSTANDING

### 1.1 Problem Statement

Water quality testing generates millions of data points globally, but the journey from raw water sample data to actionable deployment-ready ML models is fragmented. Data scientists train in isolation, models are handed off manually to engineering, deployments break silently, and there is no traceability from prediction back to the training run that produced it.

**This project solves:** The gap between ML experimentation and production deployment by building an automated, reproducible, end-to-end MLOps pipeline.

### 1.2 Use Case — Water Potability Classification

Given 9 water quality parameters (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity), predict whether water is potable (safe to drink) — a binary classification problem.

| Feature | Unit | Range (Typical) |
|---|---|---|
| pH | — | 0–14 |
| Hardness | mg/L | 100–300 |
| Solids | ppm | 0–50,000 |
| Chloramines | ppm | 0–15 |
| Sulfate | mg/L | 100–500 |
| Conductivity | μS/cm | 100–800 |
| Organic Carbon | mg/L | 0–30 |
| Trihalomethanes | μg/L | 0–120 |
| Turbidity | NTU | 0–10 |

**Dataset:** ~3,276 samples, 9 features, sourced from MainakRepositor/Datasets.

### 1.3 Who Are the Users?

| Persona | How They Use This |
|---|---|
| **Data Scientist** | Config-driven training in YAML — try different classifiers, imputers, hyperparameters without touching code. View all experiments in MLflow UI. |
| **ML Engineer** | CI/CD pipeline that validates, trains, registers, deploys automatically. No manual handoffs. |
| **DevOps / Platform** | One `git push` triggers the full pipeline. Monitor health via `/health` and `/model-info`. |
| **End User** (via frontend) | Input water quality parameters → get prediction with confidence level. |

### 1.4 Business Value

- **Speed:** From commit to deployed model in ~5-8 minutes
- **Reproducibility:** Every training run is logged, versioned, and traceable
- **Safety:** Model registry with stage promotion prevents deploying unverified models
- **Cost:** No expensive SaaS — runs on Azure App Service Basic tier (~$13/month)
- **Zero Entra ID dependency:** Works without service principals, making it accessible to teams without Azure admin privileges

### 1.5 Technical Value

- Config-driven pipeline architecture (classifier, imputer, preprocessor, transformer, PCA all swappable via YAML)
- 5 classifier algorithms, 3 imputation strategies, 6 preprocessing methods, 2 power transforms
- Grid search across any combination with 5-fold cross-validation
- Automatic model promotion based on configurable metric (default: test F1)
- Immutable artifact versioning with timestamp-based dataset + model versioning

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture Overview

```
GitHub Push
    │
    ▼
GitHub Actions
    ├── validate (syntax + YAML)
    ├── upload dataset (timestamped + latest) ──────► Blob Storage
    ├── train.py ──────────────────────────────────► Remote MLflow (App Service)
    │   ├── load dataset from Blob                        ├── log params
    │   ├── build pipeline (config-driven)                ├── log metrics
    │   ├── train + evaluate                              ├── register model version
    │   └── save model.joblib + metadata                  └── promote to Production
    ├── upload artifacts ──────────────────────────────► Blob Storage
    │   ├── models/{timestamp}/model.joblib (immutable)
    │   ├── models/latest/model.joblib (overwritable)
    │   └── model_metadata.json
    ├── build Docker image → push to ACR
    └── deploy to App Service
            │
            ▼
    FastAPI (water-potability-api)
        ├── loads model from Blob at startup
        ├── GET  /health
        ├── POST /predict
        ├── POST /predict-with-stats
        ├── GET  /prediction-stats
        └── GET  /model-info
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|---|---|
| **GitHub Actions** | Orchestrate: validate code → train → upload artifacts → build image → deploy |
| **Blob Storage** | Single source of truth for datasets, model binaries, metadata, prediction logs |
| **MLflow Server** | Persistent experiment tracking (params, metrics, runs), model registry (versions, stages) |
| **ACR** | Docker image storage with immutable SHA tags |
| **API App Service** | FastAPI inference — loads model from Blob at startup, serves 6 REST endpoints |
| **Frontend SWA** | Next.js SPA — 9 input fields, calls /predict, displays results |

### 2.3 Data Flow

**DATASET FLOW:** GitHub repo → CI upload to Blob (timestamped + latest)

**TRAINING FLOW:** train.py → read CSV → build sklearn Pipeline → GridSearchCV (optional) → evaluate → log to MLflow → register model → promote → save outputs/

**ARTIFACT FLOW:** CI uploads model.joblib + metadata to Blob (immutable versioned + overwritable latest)

**DEPLOYMENT FLOW:** CI builds Docker image → pushes to ACR → deploys to App Service → API loads model from Blob at startup

**PREDICTION FLOW:** User → POST /predict → numpy array → pipeline.predict() → pipeline.predict_proba() → update stats → log to Blob → return result

### 2.4 Service Interactions

| Service | Protocols | Dependencies |
|---|---|---|
| FastAPI API | HTTP (REST) | Blob Storage (model load), prediction_logger → Blob |
| MLflow Server | HTTP (REST API) | SQLite (backend store), App Service filesystem |
| GitHub Actions Runner | HTTPS, Docker | ACR (image push), Azure Storage (blob upload), Azure App Service (deploy) |
| Next.js Frontend | HTTP | FastAPI API (/predict) |

### 2.5 Infrastructure Layout (Azure)

```
Resource Group: mlops-rg (southeastasia)
├── Storage Account: waterpotabilitystorage
│   ├── datasets/{timestamp}/water_potability.csv
│   ├── datasets/latest_water_potability.csv
│   ├── models/{timestamp}/model.joblib + metadata
│   ├── models/latest/model.joblib + metadata
│   └── predictions/{YYYY}/{MM}/{DD}/predictions.jsonl
├── Container Registry: waterpotabilityacr
│   ├── water-potability-api:{sha} (immutable)
│   ├── water-potability-api:{version} (dataset-version tag)
│   └── mlflow-server:latest
├── App Service: water-potability-api (FastAPI, B1)
├── App Service: mlflow-server-app (MLflow, B1, port 5001)
└── Static Web Apps: water-potability-frontend (Next.js)
```
---

## 3. END-TO-END PIPELINE WALKTHROUGH

### Stage 1: Local Development Setup

**Purpose:** Enable data scientists to train and experiment locally before pushing to CI.

**Input:** Repository clone, Python 3.11, Azure CLI

**Process:**
```bash
git clone https://github.com/Zihniii/ML_water_potability
cd ML_water_potability
python -m venv venv
pip install -r requirements.txt
cp .env.example .env
# Set MLFLOW_TRACKING_URI, AZURE_STORAGE_ACCOUNT, etc.
```

**Output:** Working local environment with all ML dependencies installed.

**Tools:** Python venv, pip, .env file

**Why:** Simple, standard Python tooling. No Docker required for local development. Docker Compose is available as an alternative.

**Alternatives considered:** Docker-only development (docker-compose is available but optional). Pure Python was chosen because data scientists iterate faster without container overhead.

---

### Stage 2: Environment Configuration

**Purpose:** Separate configuration from code — make the pipeline run differently in dev vs CI without changing source.

**Input:** `config/train.yaml`, `.env.example`, GitHub Secrets

**Process:**
- `TrainConfig` class loads from YAML first, then applies environment variable overrides
- Environment variables: `DATASET_PATH`, `DATASET_VERSION`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MODEL_NAME`, `PROMOTION_METRIC`
- In CI: all env vars come from the workflow's `env:` block and `${{ secrets.* }}`

**Output:** A fully configured `TrainConfig` object ready for training.

**Tools:** `pyyaml`, `python-dotenv`, GitHub Actions `env:` and `secrets`

**Why YAML + env overrides:** Standard 12-factor app pattern. YAML provides sensible defaults; env vars override for CI-specific values without touching committed files.

**Tradeoff:** YAML can become complex. Schema validation is manual (TrainConfig.validate()), not automatic like Pydantic/JSON Schema.

---

### Stage 3: Data Ingestion

**Purpose:** Make the dataset available to the training pipeline.

**Input:** `dataset/water_potability.csv` (committed in repo)

**Process in CI:**
```yaml
- name: Upload dataset to Blob Storage
  run: |
    az storage blob upload --account-name ${{ env.AZURE_STORAGE_ACCOUNT }} \
      --account-key ${{ secrets.AZURE_STORAGE_KEY }} \
      --container-name datasets --file dataset/water_potability.csv \
      --name "${DATASET_VERSION}/water_potability.csv"
    az storage blob upload ... --name "latest_water_potability.csv" --overwrite
```

**Output:** Versioned dataset in Blob at `datasets/{timestamp}/water_potability.csv` with `latest_water_potability.csv` alias.

**Tools:** Azure CLI (`az storage blob upload`), Python `dataset_manager.py`

**Versioning scheme:** Timestamp-based (`YYYY-MM-DD_HHMMSS`) — simple, sortable, unique.

**Why not DVC or LakeFS:** Simplicity. For a single small CSV (~3K rows), timestamped Blob upload is sufficient.

---

### Stage 4: Data Preprocessing & Feature Engineering

**Purpose:** Handle missing values, scale features, apply transformations, reduce dimensionality.

**Process (inside sklearn Pipeline):**
```python
Pipeline([
    ("imputer", SimpleImputer()),           # Handle NaN values
    ("preprocessor", StandardScaler()),      # Feature scaling (optional)
    ("transformer", PowerTransformer()),     # Power transform (optional)
    ("pca", PCA()),                          # Dimensionality reduction (optional)
    ("classifier", LogisticRegression())     # ML model
])
```

**Imputation strategies:** `simple` (mean/median), `knn` (KNN-based), `iterative` (modeling-based)

**Preprocessing options:** `std` (StandardScaler), `min_max` (MinMaxScaler), `norm` (Normalizer), `poly` (PolynomialFeatures), `robust` (RobustScaler)

**Transformers:** `power_box_cox` (requires positive values), `power_yeo_johnson` (handles negatives, auto-adds StandardScaler)

**Output:** Preprocessed numpy array ready for model training.

**Tools:** scikit-learn Pipeline, config-driven builder (`PipelineBuilder` class)

**Why sklearn Pipeline:** Composability — each step is a separate object, can be swapped independently, and the entire pipeline is serializable with `joblib`. GridSearchCV can search across any step's parameters.

**Tradeoff:** No automated feature engineering beyond polynomial features. Custom sklearn Transformers would be needed for advanced feature creation.

---

### Stage 5: Model Training

**Purpose:** Train a binary classification model on the preprocessed data.

**Input:** Preprocessed X_train (9 features), y_train (binary target)

**Process:**
```python
if config.is_grid_search and param_grid:
    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    pipeline = grid.best_estimator_
else:
    pipeline.fit(X_train, y_train)
```

**Available classifiers:**

| Classifier | Pros | Cons |
|---|---|---|
| LogisticRegression | Fast, interpretable, probabilistic | Linear decision boundary |
| RandomForestClassifier | Non-linear, handles mixed data | Slower, less interpretable |
| LGBMClassifier | Fast, accurate, gradient-boosted | Needs libgomp, harder to tune |
| AdaBoostClassifier | Boosts weak learners | Sensitive to noisy data |
| SVC (with probability) | Strong theoretical foundation | Slow, needs scaling |

**Output:** Trained sklearn Pipeline object.

**Tools:** scikit-learn GridSearchCV, KFold cross-validation (5 splits)

**Why 5-fold CV:** Industry standard. Balances bias and variance in performance estimation. F1 scoring because the dataset is imbalanced (~61% not potable vs 39% potable).

---

### Stage 6: Model Evaluation

**Purpose:** Quantify model performance on held-out test data.

**Input:** Trained pipeline, X_test, y_test

**Process:** 7 metrics computed on train and test sets.

**Metrics tracked:**

| Metric | Purpose |
|---|---|
| train_accuracy | Overfitting indicator |
| train_f1_score | Overfitting indicator for imbalanced data |
| test_accuracy | Overall correctness on unseen data |
| test_f1_score | Primary comparison metric |
| test_precision | How many predicted positives were correct |
| test_recall | How many actual positives were found |
| test_roc_auc | Threshold-independent metric |

**Why test F1 as primary metric:** The dataset is imbalanced (~39% potable). Accuracy alone is misleading (a model predicting "not potable" always achieves ~61% accuracy). F1 balances precision and recall.

**Output:** Dictionary of 7 float metrics.

**Tools:** scikit-learn metrics module (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

---

### Stage 7: Experiment Tracking

**Purpose:** Log every training run's parameters, metrics, and metadata to a persistent location for comparison and audit.

**Process (inside _log_to_mlflow):**
```python
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment(config.mlflow_experiment_name)
with mlflow.start_run(run_name=f"train_{config.dataset_version}") as run:
    mlflow.log_param("classifier_type", config.classifier_type)
    # ... 11+ params
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    # Feature importance for tree-based models
    if hasattr(clf, "feature_importances_"):
        for name, imp in zip(feature_names, clf.feature_importances_):
            mlflow.log_metric(f"feat_imp_{name}", round(float(imp), 4))
```

**What gets logged:**
- **Params (11+):** dataset_path, dataset_version, classifier_type, imputer_type, preprocessor_type, transformer_type, is_pca, test_size, random_state, row_count, feature_count, plus grid search best params
- **Metrics (7):** train/test accuracy, F1, precision, recall, ROC-AUC
- **Feature importance (N):** one metric per feature for tree-based models

**NOT logged:** Model artifacts (.joblib file) — intentionally skipped to avoid PermissionError with remote MLflow.

**Output:** MLflow run with permanent run ID.

**Tools:** MLflow tracking API (Python client)

**Why remote MLflow over local:** Local MLflow in CI is ephemeral — every run starts fresh, history is lost, version numbers reset to 1.

**Tradeoff:** Remote MLflow adds network dependency. If the server is down, training still succeeds but logging fails (caught in try/except).

---

### Stage 8: Model Registry & Versioning

**Purpose:** Assign a version number to every trained model, track which version is in Production, auto-promote better models.

**Register version:**
```python
client = mlflow.MlflowClient()
try:
    client.create_registered_model(config.model_name)
except mlflow.exceptions.RestException:
    pass  # already exists
source = f"blob://models/{config.dataset_version}/model.joblib"
new_version = client.create_model_version(name=config.model_name, source=source, run_id=run_id)
model_version = new_version.version
```

**Promote if better:**
```python
def _promote_if_better(config, metrics):
    client = mlflow.MlflowClient()
    latest_prod = client.get_latest_versions(config.model_name, stages=["Production"])
    current_metric = metrics[config.promotion_metric]  # default: test_f1_score
    if not latest_prod:
        _transition_latest_to_stage(client, config.model_name, "Production")
        return
    prod_metric = ...  # fetch Production's metric
    if current_metric >= prod_metric:
        _transition_latest_to_stage(client, config.model_name, "Production")
```

**Stage lifecycle:** None → Production (when promoted) | Production → Archived (when replaced)

**Output:** Registered model with version number (v1, v2, v3...) in MLflow Model Registry.

**Tools:** MLflow Model Registry API (MlflowClient)

**Tradeoff:** The Blob source URI is metadata only — MLflow cannot serve the model from it. But the API doesn't use MLflow for serving anyway; it loads from Blob directly.

---

### Stage 9: Artifact Management

**Purpose:** Persist the trained model binary and its metadata for deployment and audit.

**Process (save locally, CI uploads to Blob):**
```python
joblib.dump(pipeline, "outputs/model.joblib")
json.dump(metadata, open("outputs/model_metadata.json", "w"), indent=2)
```

**Metadata includes:** dataset_version, classifier_type, imputer_type, all config params, all 7 metrics, training_started_at, training_duration_seconds, model_version, best_cv_params (if grid search)

**Blob layout:**
```
models/
  latest/
    model.joblib            (overwritten each CI run)
    model_metadata.json     (overwritten each CI run)
  2026-06-18_062804/
    model.joblib            (immutable — never overwritten)
    model_metadata.json     (immutable — never overwritten)
```

**Output:** Versioned model artifacts in Azure Blob Storage.

**Tools:** joblib, Azure CLI (`az storage blob upload`)

**Why joblib over pickle:** Optimized for large numpy arrays, less disk space, faster than pickle.

**Why Blob over MLflow artifacts:** The PermissionError challenge — direct Blob upload is simpler and avoids depending on MLflow's artifact plumbing.

---

### Stage 10: Containerization

**Purpose:** Package the API into a deployable Docker image.

**API Dockerfile:**
```dockerfile
FROM python:3.11-slim
RUN apt-get update -y && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt . && RUN pip install --no-cache-dir -r requirements.txt
COPY ./*.py . && COPY ./dataset/ ./dataset/ && COPY ./outputs/ ./outputs/
EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
```

**MLflow server Dockerfile:**
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.12.2
RUN mkdir -p /home/mlflow/artifacts
EXPOSE 5001
CMD ["mlflow","server","--host","0.0.0.0","--port","5001", "--backend-store-uri","sqlite:////home/mlflow/mlflow.db", "--default-artifact-root","/home/mlflow/artifacts"]
```

**Output:** Docker image `water-potability-api:{sha}` in ACR.

**Tools:** Docker, Azure Container Registry

**Why python:3.11-slim:** ~120MB base vs ~800MB for full Python. LightGBM needs `libgomp1` — one additional apt package.

---

### Stage 11: CI/CD Pipeline

**Purpose:** Automate the entire ML workflow from code push to production deployment.

**Job 1: validate** (syntax + YAML check) — Fast-fail gate (<30s)

**Job 2: mlops-pipeline** (needs validate) — Steps:

| Step | What Happens | Time |
|---|---|---|
| Checkout | Clone repo | 10s |
| Generate version | `date -u +'%Y-%m-%d_%H%M%S'` | 1s |
| Upload dataset | 2 blob uploads | 10s |
| Setup Python | actions/setup-python@v5 | 30s |
| Install deps | pip install -r requirements.txt | 60s |
| Train model | python train.py → remote MLflow | 90-180s |
| Upload model | 4 blob uploads | 15s |
| Login to ACR | docker login | 5s |
| Build & push | docker build + 2 pushes | 120-180s |
| Deploy to App Service | azure/webapps-deploy@v3 | 60-120s |

**Total estimated time:** 6-10 minutes

**Output:** Live API endpoint with newly trained model.

**Tools:** GitHub Actions, Azure CLI, Docker

**Why GitHub Actions:** Zero infrastructure, tight GitHub integration, generous free tier (2000 min/month).

**Path-based triggers only:** Only runs when relevant files change (`.py`, `.yaml`, `Dockerfile`). Frontend changes don't trigger the ML pipeline.

---

### Stage 12: Deployment

**Purpose:** Expose the trained model as a production REST API.

**Input:** Docker image in ACR (`water-potability-api:{sha}`)

**Process:**
```yaml
- name: Deploy API to App Service
  uses: azure/webapps-deploy@v3
  with:
    publish-profile: ${{ secrets.AZURE_API_PUBLISH_PROFILE }}
    images: ${{ env.ACR_NAME }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
    app-settings: |
      MODEL_NAME=${{ env.MODEL_NAME }}
      MODEL_STAGE_OR_ALIAS=${{ env.MODEL_STAGE_OR_ALIAS }}
      DATASET_VERSION=${{ steps.version.outputs.DATASET_VERSION }}
```

**What happens:** App Service pulls the Docker image from ACR, starts the container, applies env vars. On startup, `app.py` downloads `models/latest/model.joblib` from Blob.

**Requirements:** WEBSITES_PORT=5000, publish profile configured, ACR credentials set.

**Output:** Live API at `https://water-potability-api.azurewebsites.net`

**Tools:** `azure/webapps-deploy@v3` GitHub Action

**Why App Service over Container Apps:** Simpler setup. Publish-profile based auth avoids service principal dependency.

**Tradeoff:** Less control over container runtime. No auto-scaling on B1 tier.

---

### Stage 13: Frontend Deployment

**Purpose:** Provide a user interface for water quality predictions.

**Input:** Next.js source in `water-potability-frontend/`

**Process:**
```yaml
- name: Build
  run: npm run build
  env:
    NEXT_PUBLIC_API_URL: ${{ vars.BACKEND_API_URL || secrets.BACKEND_API_URL || '...' }}
- name: Deploy to Azure Static Web Apps
  uses: Azure/static-web-apps-deploy@v1
  with:
    azure_static_web_apps_api_token: ${{ secrets.AZURE_SWA_DEPLOY_TOKEN }}
    skip_app_build: true
    app_location: water-potability-frontend
    output_location: out
```

**Frontend features:** 9 input fields, animated water drop visualization, color-coded results (blue = potable, red = not potable), dark theme.

**Output:** Live frontend at Azure Static Web Apps URL.

**Tools:** Next.js (static export), Azure Static Web Apps

---

### Stage 14: Monitoring & Operations

**Purpose:** Observe model performance, API health, and prediction statistics.

**Endpoints:**

| Endpoint | Purpose | Example Response |
|---|---|---|
| `GET /health` | Liveness check | `{"status": "healthy", "model_loaded": true}` |
| `GET /model-info` | Model metadata | Model name, version, classifier, training metrics, API uptime |
| `GET /prediction-stats` | Live prediction stats | Daily counts, avg confidence, version distribution |
| `GET /` | Service status | Model name, version, dataset version |

**Prediction logging:** Every prediction logged to Blob as JSON Lines: `predictions/2026/06/19/predictions.jsonl`

**In-memory stats (limitation acknowledged):** Reset on container restart. Docs recommend Azure Table Storage for production.

**Tools:** FastAPI (built-in), Azure Blob Storage (prediction logs)
---

## 4. CHALLENGES & SOLUTIONS

### Challenge 1: `PermissionError: [Errno 13]` on MLflow artifact logging

**Symptom:** During CI training, `mlflow.log_artifact()` and `mlflow.sklearn.log_model()` threw `PermissionError: [Errno 13] Permission denied: '/home/mlflow/artifacts'`.

**Root Cause:** MLflow Server was pointing its `--default-artifact-root` to the local filesystem (`/home/mlflow/artifacts`). When the CI training job (different machine, no network filesystem) called `log_artifact()`, it tried to write to that same local path — impossible because the runner and server are different machines.

**Resolution:** Removed all `mlflow.log_artifact()` and `mlflow.sklearn.log_model()` calls from `train.py`. CI uploads the model binary directly to Blob Storage. The MLflow server only tracks params, metrics, and registry.

**Lesson:** Remote MLflow with local artifact store = PermissionError. Two solutions: (a) configure MLflow with Azure Blob Storage as artifact store (`mlflow server ... --default-artifact-root wasbs://...`), or (b) skip MLflow artifact logging entirely and upload manually. We chose (b) for simplicity.

---

### Challenge 2: App Service crashing — blank `appCommandLine`

**Symptom:** After deploying the MLflow image to App Service, the container kept crashing with no visible logs.

**Root Cause:** `az webapp config set --acr-use-identity false` set `appCommandLine` to a single space character `" "`. This overrode the Dockerfile's CMD instruction, so the container started with an empty command and immediately exited.

**Resolution:** Cleared `appCommandLine` via Invoke-RestMethod with Azure Management API: `PUT .../config/web?api-version=2023-12-01` with body `{"appCommandLine": null}`.

**Lesson:** Never set `appCommandLine` unless you intentionally want to override CMD. Use `az webapp config show` to inspect current config before making changes.

---

### Challenge 3: Port mismatch — MLflow runs on 5000, App Service expects 5000

**Symptom:** Container started but health checks failed. MLflow server was unreachable at port 5000.

**Root Cause:** MLflow server runs on port 5000 by default. App Service's WEBSITES_PORT was set to the same value (5000 on Linux, 8000 on Windows). Both were correct — but confusion arose when we changed the MLflow Dockerfile to use port 5001 and forgot to update WEBSITES_PORT.

**Resolution:** Set `WEBSITES_PORT=5001` for the MLflow App Service to match the Dockerfile's EXPOSE 5001.

**Lesson:** WEBSITES_PORT is the critical bridge between Docker container port and Azure App Service routing. Keep it in sync with the Dockerfile's EXPOSE.

---

### Challenge 4: Ephemeral SQLite — runs lost on restart

**Symptom:** After an App Service restart, all past MLflow runs were gone. Only a fresh `Default` experiment remained.

**Root Cause:** SQLite database was stored at the container's local filesystem (default `mlruns.db`). App Service restarts create a fresh container with no persisted filesystem.

**Resolution:** Enabled `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true` and configured MLflow to store SQLite at a persistent path: `--backend-store-uri sqlite:////home/mlflow/mlflow.db`.

**Lesson:** Never use default SQLite paths in App Service. Always persist via the Azure Files mount that App Service provides when `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`.

---

### Challenge 5: ACR credentials — login failure in CI

**Symptom:** `docker login $ACR_NAME.azurecr.io` failed with authentication error.

**Root Cause:** ACR admin user was disabled by default. No service principal could be created (no Entra ID permissions).

**Resolution:** Enabled ACR admin user in Azure Portal → Settings → Admin user → Enable. Used the generated `username` and `password` as GitHub Secrets.

**Lesson:** Without Entra ID, ACR admin credentials are the only option. Rotate them periodically since they're static.

---

### Challenge 6: Azure CLI timeouts and resource locking

**Symptom:** `az webapp config set`, `az storage blob upload`, etc., intermittently timed out or returned "Another operation is in progress."

**Root Cause:** Azure Resource Manager applies locks per resource. Rapid successive operations can trigger contention.

**Resolution:** Used `Invoke-RestMethod` with bearer token for App Service config changes (faster, no CLI overhead). For blob uploads, added retry logic and single-threaded uploads.

**Lesson:** The REST API is faster and more reliable than Azure CLI for simple operations. For blob operations, wait for each upload to complete before starting the next.

---

### Challenge 7: WASBS crash — MLflow with Azure Blob artifact store

**Symptom:** When we tried `--default-artifact-root wasbs://container@storageaccount.blob.core.windows.net/`, the MLflow server crashed at startup with ImportError for azure-storage-blob.

**Root Cause:** `mlflow server` requires `azure-storage-blob` Python package to use wasbs:// URIs. Our Dockerfile only installed `mlflow`. Additionally, the authentication flow (connection string vs SAS vs DefaultAzureCredential) was unclear — MLflow doesn't pass credentials via env vars consistently.

**Resolution:** Abandoned wasbs:// artifact store. We don't need MLflow to manage artifacts; we handle them manually via CI → Blob.

**Lesson:** If you're not using MLflow's artifact serving features, skip the artifact store entirely. Manual Blob management is simpler, more reliable, and gives full control over naming conventions.

---

### Challenge 8: MLflow version resolution — `version` was null in model-info

**Symptom:** The `/model-info` endpoint returned `"version": null`.

**Root Cause:** Before the fix, `train.py` didn't call `mlflow.create_model_version()` — it only tracked params/metrics. Without a registered version, the API couldn't report a version number.

**Resolution:** Added `mlflow.create_model_version()` in `train.py` right after training. The model version (e.g., "v3") is stored in the metadata JSON, and `app.py` reads it when returning `/model-info`.

**Lesson:** MLflow separates run tracking (params/metrics) from model registry (versions/stages). Creating a version is an explicit API call — it doesn't happen automatically.

---

## 5. MLOPS PRINCIPLES APPLIED

| Principle | How We Apply It |
|---|---|
| **Reproducibility** | Config-driven training (YAML), versioned datasets, immutable artifacts, MLflow tracking |
| **Versioning** | Timestamp-based dataset versions, sequential model versions (v1, v2, v3), explicit metadata |
| **Automation** | CI/CD from commit to deployment — no manual steps after code push |
| **Monitoring** | /health (liveness), /model-info (metadata), /prediction-stats (performance), prediction logging to Blob |
| **Experiment Tracking** | Remote MLflow with SQLite persistence, params + metrics + registry for every run |
| **Model Registry** | Versioned models with stage promotion (None → Production → Archived), metric-based auto-promotion |
| **Infrastructure as Code** | GitHub Actions YAML (pipeline definition), Dockerfile (environment), config/ (pipeline config) |
| **Testing & Validation** | validate job (syntax + YAML), GridSearchCV (validation), test/train split |
| **Containerization** | Docker for API and MLflow, ACR for image storage, App Service for deployment |
| **CI/CD Integration** | GitHub Actions orchestrates the full pipeline — GitHub is the single source of truth |
| **Zero-Entra ID** | No Azure AD required. Storage account keys, ACR admin credentials, publish profile for deployment |

---

## 6. SECURITY & AUTHENTICATION

### How Authentication Works (No Entra ID)

| Service | Auth Method | Where Stored |
|---|---|---|
| Azure Storage | Account key in GitHub Secret | `AZURE_STORAGE_KEY` |
| Azure Container Registry | ACR admin username + password | `ACR_USERNAME`, `ACR_PASSWORD` |
| App Service Deployment | Publish profile (scm endpoint) | `AZURE_API_PUBLISH_PROFILE` |
| Azure CLI (local) | Interactive login via `az login` | Cached locally |
| GitHub Actions Runner | N/A (trusted runner) | N/A |

### Security Considerations
- Storage account keys are static and powerful (full access to all blobs)
- ACR admin passwords are static — rotate periodically
- No RBAC, no managed identities, no OAuth2
- GitHub Secrets encryption protects credentials at rest
- App Service publish profile is equivalent to full deployment access

### Why This Matters
This system works without any Entra ID permissions — meaning any developer with an Azure subscription (even a student subscription) can replicate it. It's more accessible than architectures requiring service principals.

---

## 7. COST ANALYSIS

| Service | Tier | Approx Monthly Cost |
|---|---|---|
| App Service (API) | B1 (1 vCPU, 1.75 GB RAM) | ~$13 |
| App Service (MLflow) | B1 (1 vCPU, 1.75 GB RAM) | ~$13 |
| Blob Storage | Standard LRS (hot) | ~$0.02/GB |
| Container Registry | Basic | ~$5 |
| Static Web Apps | Free tier | $0 |
| GitHub Actions | Free (2000 min/month) | $0 |
| **Total** | | **~$31/month** |

**Cost per training run:** ~$0.01 (6 min runner time)
**Cost per API call:** Negligible (~$0.000001 per request)

---

## 8. LIMITATIONS & FUTURE WORK

### Current Limitations
1. **No automated retraining trigger** — Training runs on every push to specific files, not on data drift detection
2. **No A/B testing** — Deploy staged version to production without splitting traffic
3. **Prediction stats are in-memory** — Reset on container restart. Logged to Blob but not served from there
4. **No model monitoring beyond stats** — No accuracy decay detection, data drift monitoring, or alerting
5. **No automated rollback** — Previous version exists in registry but isn't auto-deployed on failure
6. **Static credentials** — No managed identities, no credential rotation automation
7. **Single region** — Southeast Asia only. No multi-region deployment
8. **No feature store** — Features computed and stored inline, not in a centralized store
9. **Test coverage is zero** — No unit tests for train.py or app.py
10. **No model explanation** — Only feature importance. No SHAP/LIME integration

### Future Work Roadmap

**Phase 2 (Operational Excellence)**
- Feature store integration (Azure Feature Store or Feast)
- Data drift detection with Evidently AI or Great Expectations
- A/B testing and canary deployments
- Automated rollback on metric degradation

**Phase 3 (Scale & Reliability)**
- Unit and integration test suite
- Multi-region deployment
- Managed identity migration (when Entra ID becomes available)
- Auto-scaling rules
- Performance benchmark suite

**Phase 4 (Advanced MLOps)**
- Automated pipeline retriggering on data drift
- Model explainability (SHAP/LIME)
- Bias and fairness testing
- Full observability stack (Azure Monitor / Prometheus)
- Active learning loop (flag low-confidence predictions for retraining)

---

## 9. KEY TALKING POINTS

Use these talking points for a 10-minute project presentation:

**30 seconds — Problem Statement**
"Water quality data is fragmented. Models are trained in isolation, handed off manually, deployed silently. We built a pipeline that makes this automated, reproducible, and traceable."

**60 seconds — Architecture**
"Git push → Blob Storage → Remote MLflow → sklearn Pipeline → Model Registry → ACR → App Service → REST API. Six Azure services, one pipeline."

**30 seconds — Demo Flow**
"Push a config change. Watch GitHub Actions validate, train, register, build, deploy. See the run in MLflow UI. Hit the API."

**60 seconds — Key Decision 1: Skip MLflow Artifact Store**
"PermissionError with remote MLflow. Instead of fighting it, we upload artifacts to Blob directly. Simpler, no dependency, full naming control."

**60 seconds — Key Decision 2: Zero Entra ID**
"No service principals needed. Storage keys, ACR passwords, publish profiles. Any dev with a student subscription can replicate this."

**30 seconds — Key Decision 3: Config-driven Pipeline**
"Change classifier, imputer, preprocessor in YAML. Grid search across any combination. No code changes needed for experimentation."

**30 seconds — Cost**
"~$31/month for the entire stack. Free CI minutes. Approx $0.01 per training run."

**60 seconds — Lessons Learned**
"1. appCommandLine override crashes containers. 2. Ephemeral storage loses MLflow data. 3. ACR admin credentials need enabling. 4. WASBS import error - keep artifacts simple."

**30 seconds — Future Roadmap**
"Data drift monitoring, test coverage, A/B testing, managed identity migration, automated rollback, feature store."

---

## 10. DEMO SCRIPT

### Demo Scenario 1: Successful pipeline run
1. Push a config change (e.g., change classifier to RandomForest)
2. Show GitHub Actions running in real-time
3. Point to MLflow UI showing the new run
4. Point to `/model-info` showing the new model version in Production
5. POST a sample to `/predict` and show the result

### Demo Scenario 2: Model promotes over previous
1. Two sequential pushes with improving test F1
2. Show Production stage moved from v2 to v3 in MLflow UI
3. Show `/model-info` reporting the promoted version

### Demo Scenario 3: Degraded model doesn't promote
1. Push a config that produces lower test F1
2. Show Production stage remains at previous version
3. Show `/model-info` still reporting the better version

### Demo Scenario 4: Failure recovery
1. Intentionally break a config (e.g., invalid classifier name)
2. Show validate job failing fast (<30s)
3. Pipeline stops before training — no wasted compute

---

## 11. COMMON PITFALLS AND RECOVERY

| Pitfall | Symptom | Recovery |
|---|---|---|
| MLflow server down | CI training fails at mlflow.log_param | 1. Check `https://mlflow-server-app.azurewebsites.net` 2. Restart App Service 3. Check logs |
| Storage key rotated | Blob upload fails | Update `AZURE_STORAGE_KEY` in GitHub Secrets |
| ACR password rotated | Docker push fails | Update `ACR_PASSWORD` in GitHub Secrets |
| Publish profile expired | Deploy fails | Download new publish profile from Azure Portal → Update secret |
| Container crash loop | HTTP 503 | Check App Service Log Stream → Diagnose root cause |
| Model version out of sync | app.py reports stale model | Delete model.joblib from Blob and restart API service |
| SQLite database corruption | MLflow returns 500 | SSH → delete mlflow.db → restart. MLflow recreates it fresh. |
| Out of disk space on B1 | Container restart | Check logs for disk pressure → Scale up or clean old runs |

---

## 12. ARCHITECTURE DECISIONS RECORD

| Decision | Options Considered | Chosen | Rationale |
|---|---|---|---|
| Artifact storage | MLflow artifacts vs Blob | Blob (direct upload) | PermissionError avoidance, simpler, full naming control |
| Model registry | MLflow vs custom | MLflow | Already using MLflow for tracking; registry is free |
| CI/CD platform | GitHub Actions vs Jenkins vs GitLab | GitHub Actions | Zero infra, tight GitHub integration, generous free tier |
| Cloud provider | Azure vs AWS vs GCP | Azure | Academic subscription, no budget constraints |
| Inference server | FastAPI vs Flask vs BentoML | FastAPI | Async, auto-docs, fast, sklearn compatible |
| Frontend framework | Next.js vs plain HTML | Next.js | SSR, modern, good DX |
| Container platform | App Service vs AKS vs Container Apps | App Service | Simplest deployment, publish-profile auth |
| Config format | YAML vs JSON vs TOML | YAML | Readable, supports comments, widely used in ML |
| Auth model | Entra ID vs static keys | Static keys | Zero Entra ID permissions available |
| Dataset versioning | Timestamp vs hash vs semantic | Timestamp | Simple, sortable, human-readable |

---

## 13. TECHNICAL SPECS

| Aspect | Detail |
|---|---|
| Runtime | Python 3.11, Node 20 (frontend) |
| ML Framework | scikit-learn 1.6+, LightGBM |
| API Framework | FastAPI 0.115+ |
| Frontend | Next.js 15 (static export) |
| MLOps Frameworks | MLflow 2.12+ |
| CI/CD | GitHub Actions (Ubuntu 22.04 runner) |
| Container Registry | Azure Container Registry (Basic) |
| Compute | Azure App Service B1 (Linux) |
| Storage | Azure Blob Storage (LRS Hot) |
| Model Serialization | joblib (sklearn Pipeline) |
| Experiment Storage | SQLite (persisted via Azure Files) |
| Infrastructure Language | Dockerfile, GitHub Actions YAML |

---

## 14. CODE STRUCTURE

```
ML_water_potability/
├── .github/workflows/
│   ├── mlops-key-auth.yml     # Main CI/CD pipeline
│   ├── frontend-deploy.yml    # Frontend deployment
│   └── mlflow-build.yml       # Manual MLflow image rebuild
├── config/
│   └── train.yaml             # Training configuration
├── dataset/
│   └── water_potability.csv   # Local copy of dataset
├── docker/
│   └── mlflow.Dockerfile      # MLflow server Dockerfile
├── docs/
│   ├── SUMMARY.md             # Project summary with challenges
│   └── PRESENTATION_PACKAGE.md # This file
├── water-potability-frontend/ # Next.js frontend app
├── .dockerignore
├── .env.example
├── AGENTS.md                  # Agent configuration
├── Dockerfile                 # API Dockerfile
├── README.md                  # Project README
├── app.py                     # FastAPI inference server
├── config.py                  # TrainConfig + env loading
├── configurator.py            # Pipeline builder from config
├── dataset_manager.py         # Blob dataset operations
├── requirements.txt           # Python dependencies
└── train.py                   # Training script
```

---

## 15. SCRIPT TO DEPLOY FROM SCRATCH

```bash
# 1. Clone
git clone https://github.com/Zihniii/ML_water_potability
cd ML_water_potability

# 2. Create Azure resources
az group create --name mlops-rg --location southeastasia
az storage account create --name waterpotabilitystorage --resource-group mlops-rg
az storage container create --name datasets --account-name waterpotabilitystorage
az storage container create --name models --account-name waterpotabilitystorage
az container registry create --name waterpotabilityacr --resource-group mlops-rg --sku Basic --admin-user-enabled true
az webapp create --name water-potability-api --resource-group mlops-rg --plan mlops-plan --runtime "PYTHON:3.11"
az webapp create --name mlflow-server-app --resource-group mlops-rg --plan mlops-plan --runtime "PYTHON:3.11"

# 3. Set GitHub secrets
gh secret set AZURE_STORAGE_KEY --repo your-org/ML_water_potability
gh secret set AZURE_ACR_NAME --repo your-org/ML_water_potability --body waterpotabilityacr
gh secret set ACR_USERNAME --repo your-org/ML_water_potability
gh secret set ACR_PASSWORD --repo your-org/ML_water_potability
gh secret set AZURE_API_PUBLISH_PROFILE --repo your-org/ML_water_potability

# 4. Deploy MLflow server first
docker build -f docker/mlflow.Dockerfile -t waterpotabilityacr.azurecr.io/mlflow-server:latest .
docker push waterpotabilityacr.azurecr.io/mlflow-server:latest
az webapp config set --resource-group mlops-rg --name mlflow-server-app \
  --container-image-name waterpotabilityacr.azurecr.io/mlflow-server:latest
az webapp config appsettings set --resource-group mlops-rg --name mlflow-server-app \
  --settings WEBSITES_PORT=5001 WEBSITES_ENABLE_APP_SERVICE_STORAGE=true

# 5. Push code — CI handles the rest
git add .
git commit -m "Initial MLOps pipeline"
git push
```

---

## Appendix A: MLflow UI Access

- URL: `https://mlflow-server-app.azurewebsites.net`
- Features available: Experiment list, run details, params, metrics, model registry
- NOT available: Artifact viewer (no artifact store configured)

## Appendix B: API Endpoints Reference

| Method | Path | Request Body | Response |
|---|---|---|---|
| GET | `/` | — | `{"model_name": ..., "model_version": ...}` |
| GET | `/health` | — | `{"status": "healthy", "model_loaded": true/false}` |
| POST | `/predict` | `{"pH": 7.0, "Hardness": 150.0, ...}` | `{"prediction": 1, "is_potable": true, "confidence": 0.85, ...}` |
| POST | `/predict-with-stats` | Same as /predict | Same + model_version, timestamp |
| GET | `/prediction-stats` | — | `{"total_predictions": ..., "daily_counts": {...}}` |
| GET | `/model-info` | — | Full model metadata + training metrics + API info |

## Appendix C: Directory Reference

- **Azure Portal**: https://portal.azure.com → Resource groups → mlops-rg
- **MLflow UI**: https://mlflow-server-app.azurewebsites.net
- **API**: https://water-potability-api.azurewebsites.net
- **GitHub Repo**: https://github.com/Zihniii/ML_water_potability
- **Frontend**: https://water-potability-frontend.(azurestaticapps.net)

---

*Generated: 2026-06-19 | Pipeline Status: Operational | MLflow: Healthy | Model: v3 in Production*
