# Water Potability — MLOps Pipeline

End-to-end MLOps project with **no Entra ID / service principal**: config-driven training → remote MLflow tracking → model registry promotion → Blob artifact storage → FastAPI serving → CI/CD → Azure App Service.

All auth via storage account keys + ACR admin credentials.

---

## Architecture

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
    │   ├── models/{timestamp}/model.joblib
    │   ├── models/latest/model.joblib
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

- **CI/CD training** → logs to remote MLflow server (persistent, not ephemeral), saves model + metadata to Blob
- **API** → loads model from Blob at startup (no baked-in model, no runtime MLflow dependency)
- **MLflow server** → runs as a Docker container on Azure App Service, persists runs/metrics/registry via SQLite

---

## Quick Start (local)

```bash
git clone https://github.com/Zihniii/ML_water_potability
cd ML_water_potability

# Train locally (logs to remote MLflow if MLFLOW_TRACKING_URI is set)
python train.py --config config/train.yaml

# With grid search
python train.py --config config/train.yaml --hyperparameters config/hyperparameters.yaml

# Start API locally
uvicorn app:app --host 0.0.0.0 --port 5000

# Test
curl http://localhost:5000/health
```

---

## Forks & CI/CD

If you fork this repo and want CI/CD, you need:

### 1. Azure Resources

```bash
az group create --name mlops-rg --location southeastasia

# Storage (Blob)
az storage account create --name <unique-name> --resource-group mlops-rg
az storage container create --account-name <name> --name datasets

# Container Registry
az acr create --resource-group mlops-rg --name <acr-name> --sku Basic --admin-enabled true

# MLflow server App Service
az appservice plan create --name mlops-plan --resource-group mlops-rg --sku B1 --is-linux
az webapp create --resource-group mlops-rg --plan mlops-plan --name <mlflow-app-name> \
  --deployment-container-image-name <acr-name>.azurecr.io/mlflow-server:latest
az webapp config appsettings set --resource-group mlops-rg --name <mlflow-app-name> \
  --settings WEBSITES_PORT=5001 WEBSITES_ENABLE_APP_SERVICE_STORAGE=true

# API App Service
az webapp create --resource-group mlops-rg --plan mlops-plan --name <api-app-name> \
  --deployment-container-image-name <acr-name>.azurecr.io/water-potability-api:latest
```

### 2. Build & push the MLflow server image

```bash
docker build -f docker/mlflow.Dockerfile -t <acr-name>.azurecr.io/mlflow-server:latest .
az acr login --name <acr-name>
docker push <acr-name>.azurecr.io/mlflow-server:latest
```

Then configure the MLflow App Service to use ACR credentials (`DOCKER_REGISTRY_SERVER_URL`, `DOCKER_REGISTRY_SERVER_USERNAME`, `DOCKER_REGISTRY_SERVER_PASSWORD`).

### 3. Add GitHub Secrets

| Secret | Source |
|---|---|
| `AZURE_STORAGE_ACCOUNT` | Your storage account name |
| `AZURE_STORAGE_KEY` | `az storage account keys list --account-name <name> --query "[0].value" -o tsv` |
| `AZURE_ACR_NAME` | Your ACR name |
| `ACR_USERNAME` | Your ACR name |
| `ACR_PASSWORD` | `az acr credential show --name <acr> --query "passwords[0].value" -o tsv` |
| `AZURE_API_PUBLISH_PROFILE` | Download from `water-potability-api` → Deployment Center → Publish Profile |

### 4. Push

```bash
git push origin main
```

CI runs `mlops-key-auth.yml`:
1. Validate code & config
2. Upload dataset to Blob (timestamped + latest)
3. Train model → logs to remote MLflow → registers model version
4. Upload model + metadata to Blob
5. Build & push API Docker image to ACR (SHA tag)
6. Deploy API to App Service

### 5. Verify

```bash
curl https://<api-app-name>.azurewebsites.net/health
curl https://<api-app-name>.azurewebsites.net/model-info
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict with probability, confidence, model version |
| `POST` | `/predict-with-stats` | Predict + cumulative stats |
| `GET` | `/prediction-stats` | Daily counts, avg confidence, model usage |
| `GET` | `/model-info` | Model metadata (version, classifier, metrics, training timestamp, API uptime) |

```bash
curl -X POST https://<api-app-name>.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"ph": 7.0, "Hardness": 200.0, "Solids": 20000.0, "Chloramines": 7.5, "Sulfate": 330.0, "Conductivity": 400.0, "Organic_carbon": 14.0, "Trihalomethanes": 70.0, "Turbidity": 3.5}'
```

---

## Frontend — Azure Static Web Apps

The frontend auto-deploys to Azure Static Web Apps on push to `main` (via `.github/workflows/frontend.yml`).

### Setup

```bash
az staticwebapp create \
  --name water-potability-frontend \
  --resource-group mlops-rg \
  --source https://github.com/<your-org>/<your-repo> \
  --location southeastasia \
  --branch main \
  --app-location water-potability-frontend \
  --output-location out
```

Copy the deployment token from SWA portal (Settings → Deployment Token) and add as GitHub secret `AZURE_SWA_DEPLOY_TOKEN`. Add your backend API URL as GitHub variable `BACKEND_API_URL`.

---

## Dataset

Source: [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets/tree/master)
~3.2K samples, 9 features → binary classification (Potability).

---

## Remaining Risks

- **No unit tests** — only manual test scripts exist.
- **In-memory stats** — prediction stats reset on container restart.
- **MLflow SQLite** — SQLite may not scale under heavy concurrent writes.

---

## Docs

| File | Contents |
|---|---|
| [SUMMARY.md](docs/SUMMARY.md) | Full summary: challenges, fixes, MLOps principles |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, component flow |
| [MLFLOW.md](docs/MLFLOW.md) | MLflow setup, local vs cloud, model stages |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Local compose, production CI/CD, rollback |
| [DATASET_MANAGEMENT.md](docs/DATASET_MANAGEMENT.md) | Blob storage layout, DatasetManager API |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | All endpoints, request/response, error codes |
