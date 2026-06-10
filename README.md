# Water Potability — MLOps Pipeline

End-to-end MLOps project: config-driven training → dataset versioning → cloud MLflow tracking → model registry promotion → FastAPI serving → CI/CD → Azure Container Apps.

---

## Quick Start (local only, no Azure needed)

```bash
git clone https://github.com/Zihniii/ML_water_potability
cd ML_water_potability

cp .env.example .env
docker compose up -d                     # FastAPI (5000) + MLflow (5001)
docker compose --profile train run train # Train + register model
curl http://localhost:5000/health
```

**Prerequisites:** Docker, Python 3.11+

---

## Forks & CI/CD

If you fork this repo and want CI/CD, you need:

### 1. Azure Resources

```bash
az group create --name mlops-rg --location southeastasia

az storage account create --name <unique-name> --resource-group mlops-rg
az storage container create --account-name <name> --name datasets

az ml workspace create --resource-group mlops-rg --name water-potability-mlw \
  --storage-account <name>

az acr create --resource-group mlops-rg --name <acr-name> --sku Basic --admin-enabled true

az provider register -n Microsoft.App
az containerapp env create --resource-group mlops-rg --name water-potability-env
az containerapp create --resource-group mlops-rg --name water-potability-api \
  --environment water-potability-env --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
  --target-port 5000 --ingress external
```

### 2. Add GitHub Secrets (key auth — no AAD needed)

| Secret | Source |
|---|---|
| `AZURE_STORAGE_ACCOUNT` | Your storage account name |
| `AZURE_STORAGE_KEY` | `az storage account keys list --account-name <name> --query "[0].value" -o tsv` |
| `AZURE_ACR_NAME` | Your ACR name |
| `ACR_USERNAME` | Your ACR name |
| `ACR_PASSWORD` | `az acr credential show --name <acr> --query "passwords[0].value" -o tsv` |
| `MLFLOW_TRACKING_URI` | `az ml workspace show -g mlops-rg -n water-potability-mlw --query mlflow_tracking_uri -o tsv` |

### 3. Push — CI trains model & pushes Docker image

```bash
git push origin main
```

CI runs `mlops-key-auth.yml`: validate → upload dataset → train → push image to ACR.

### 4. Deploy to Azure Container Apps (from your machine)

Since your Azure account lacks AAD permissions, deployment is a single command locally:

```powershell
.\scripts\deploy.ps1 -AcrName <your-acr-name>
```

This reads the latest image tag from git, deploys to Container Apps, and runs a health check.

---

## Local Training

```bash
# Default pipeline
python train.py --config config/train.yaml

# With grid search
python train.py --config config/train.yaml --hyperparameters config/hyperparameters.yaml

# Different classifier (edit YAML or use env)
classifier_type: light_gbm  # in config/train.yaml
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict with probability, confidence, model version |
| `POST` | `/predict-with-stats` | Predict + cumulative stats |
| `GET` | `/prediction-stats` | Daily counts, avg confidence, model usage |
| `GET` | `/model-info` | Loaded model metadata |

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ph": 7.0, "Hardness": 200.0, "Solids": 20000.0, "Chloramines": 7.5, "Sulfate": 330.0, "Conductivity": 400.0, "Organic_carbon": 14.0, "Trihalomethanes": 70.0, "Turbidity": 3.5}'
```

## Frontend — Azure Static Web Apps

The frontend auto-deploys to Azure Static Web Apps on push to `main` (via `.github/workflows/frontend.yml`).

### Setup

```bash
# 1. Create the SWA resource (Azure portal or CLI)
az staticwebapp create \
  --name water-potability-frontend \
  --resource-group mlops-rg \
  --source https://github.com/<your-org>/<your-repo> \
  --location southeastasia \
  --branch main \
  --app-location water-potability-frontend \
  --output-location out

# 2. Copy the deployment token from the SWA portal
#    (Settings → Deployment Token) and add it as a GitHub secret:
#    Name: AZURE_SWA_DEPLOY_TOKEN

# 3. Add your backend API URL as a GitHub variable:
#    Name: BACKEND_API_URL
#    Value: https://water-potability-api.azurecontainerapps.io
#    (Or the URL of your deployed container app)

# 4. Push to main — the workflow deploys automatically
git push origin main
```

**Local dev:** `NEXT_PUBLIC_API_URL` defaults to `http://localhost:5000` — matches `docker compose up -d`.

## Docs

| File | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, component flow |
| [MLFLOW.md](docs/MLFLOW.md) | MLflow setup, local vs cloud, model stages |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Local compose, production CI/CD, rollback |
| [DATASET_MANAGEMENT.md](docs/DATASET_MANAGEMENT.md) | Blob storage layout, DatasetManager API |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | All endpoints, request/response, error codes |

## Dataset

Source: [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets/tree/master)  
~3.2K samples, 9 features → binary classification (Potability).

## Legacy

`modeling/`, `config/train_config.yaml`, `config.py`, `get_model_for_production.py`, `test_post_request.py`, and `kubernetes_deployment/` are preserved for reference.
