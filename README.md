# Water Potability — MLOps Pipeline

End-to-end MLOps project: config-driven training → MLflow tracking (Azure ML) → model registry promotion → FastAPI serving → CI/CD → Azure App Service.

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Local Dev   │────▶│  Azure ML Studio │◀────│  CI/CD (GH)   │
│  train.py    │     │  MLflow Tracking │     │  train.py     │
│  (az login)  │     │  + Model Registry│     │  (local MLf)  │
└──────────────┘     └────────┬─────────┘     └───────┬───────┘
                              │                        │
                              │                        ▼
                              │                 ┌───────────────┐
                              │                 │  Docker Image │
                              │                 │  (model baked)│
                              │                 └───────┬───────┘
                              │                         │
                              ▼                         ▼
                        ┌────────────────────────────────────┐
                        │        Azure App Service           │
                        │  FastAPI (loads model.joblib)      │
                        └────────────────────────────────────┘
```

- **Local training** → logs directly to Azure ML Studio via `az login` + `.env`
- **CI/CD training** → logs to local MLflow, saves model as `outputs/model.joblib` → baked into Docker image
- **API** → loads model from baked file (no runtime MLflow dependency)
- **Azure ML Studio** → view all local runs; CI runs logged locally

---

## Quick Start (local, tracks to Azure ML)

```bash
# Prerequisites: Python 3.11+, Azure CLI (az login)

git clone https://github.com/Zihniii/ML_water_potability
cd ML_water_potability

# 1. Authenticate with Azure
az login

# 2. Copy environment config
cp .env.example .env
# Edit .env → set MLFLOW_TRACKING_URI to your Azure ML workspace URI

# 3. Train — logs to Azure ML Studio automatically
python train.py --config config/train.yaml

# 4. Start the API locally
uvicorn app:app --host 0.0.0.0 --port 5000

# 5. Test
curl http://localhost:5000/health
```

---

## Forks & CI/CD

If you fork this repo and want CI/CD, you need:

### 1. Azure Resources

```bash
az group create --name mlops-rg --location southeastasia

az storage account create --name <unique-name> --resource-group mlops-rg
az storage container create --account-name <name> --name datasets

az acr create --resource-group mlops-rg --name <acr-name> --sku Basic --admin-enabled true
```

### 2. Add GitHub Secrets

| Secret | Source |
|---|---|
| `AZURE_STORAGE_ACCOUNT` | Your storage account name |
| `AZURE_STORAGE_KEY` | `az storage account keys list --account-name <name> --query "[0].value" -o tsv` |
| `AZURE_ACR_NAME` | Your ACR name |
| `ACR_USERNAME` | Your ACR name |
| `ACR_PASSWORD` | `az acr credential show --name <acr> --query "passwords[0].value" -o tsv` |
| `AZURE_API_PUBLISH_PROFILE` | Download from App Service → Deployment Center |

### 3. Push — CI trains model & pushes Docker image

```bash
git push origin main
```

CI runs `mlops-key-auth.yml`:
1. Validate code & config
2. Upload dataset to blob storage
3. Start local MLflow server
4. Train model → saves `outputs/model.joblib`
5. Build & push Docker image to ACR (model baked in)
6. Deploy API to App Service

### 4. Create API App Service (one-time)

Create an App Service in Azure Portal:
- Publish: **Container**
- Region: *same as your resource group*
- OS: **Linux**
- Plan: **Basic B1**
- Image: `water-potability-api` (from ACR)
- Port: **5000**
- Name: `water-potability-api`

Download its **Publish Profile** (Deployment Center → Publish Profile) and save as `AZURE_API_PUBLISH_PROFILE` secret.

### 5. Next push — fully automated

```
push → validate → upload dataset → train → build image → push to ACR → deploy API
```

Access the API at `https://water-potability-api.azurewebsites.net/health`

---

## 🔴 Known Limitation — CI Azure ML auth

CI currently trains with a **local** MLflow server (no Azure login). Runs do not appear in Azure ML Studio.

To enable CI → Azure ML tracking, create a service principal and set `AZURE_CREDENTIALS` secret:

```bash
az ad sp create-for-rbac --name "mlops-sp" \
  --role contributor \
  --scopes /subscriptions/<subscription-id>

# Save the JSON output as GitHub secret AZURE_CREDENTIALS
```

Requires **Entra ID permissions** (Directory admin or Application developer role).

---

## Local Training (with Azure ML tracking)

```bash
# Ensure you're logged into Azure
az login

# Set the tracking URI (or use .env)
$env:MLFLOW_TRACKING_URI="azureml://<your-workspace-uri>"  # PowerShell

# Run training
python train.py --config config/train.yaml

# With grid search
python train.py --config config/train.yaml --hyperparameters config/hyperparameters.yaml
```

Runs appear in Azure ML Studio automatically via `azureml-mlflow` + `DefaultAzureCredential`.

---

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

---

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
#    Value: https://water-potability-api.azurewebsites.net

# 4. Push to main — the workflow deploys automatically
git push origin main
```

**Local dev:** `NEXT_PUBLIC_API_URL` defaults to `http://localhost:5000`

---

## Docs

| File | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, component flow |
| [MLFLOW.md](docs/MLFLOW.md) | MLflow setup, local vs cloud, model stages |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Local compose, production CI/CD, rollback |
| [DATASET_MANAGEMENT.md](docs/DATASET_MANAGEMENT.md) | Blob storage layout, DatasetManager API |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | All endpoints, request/response, error codes |

---

## Dataset

Source: [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets/tree/master)  
~3.2K samples, 9 features → binary classification (Potability).

---

## Implementation Report

### Changes made

| File | Change |
|---|---|
| `train.py` | Added `load_dotenv()` to load `.env`; dump model to `outputs/model.joblib` after training |
| `app.py` | Added fallback to load `outputs/model.joblib` if MLflow registry fails |
| `Dockerfile` | Added `COPY ./outputs/ ./outputs/` to bake model into image |
| `.dockerignore` | Removed `outputs/` so model file is included in Docker build |
| `requirements.txt` | Added `python-dotenv` for `.env` loading |
| `.github/workflows/mlops-key-auth.yml` | Removed Azure login step; always use local MLflow; deploy without cloud MLflow URI |
| `outputs/.gitkeep` | Placeholder to keep `outputs/` in git |

### Local development

1. `az login` — authenticate with Azure
2. `cp .env.example .env` — set `MLFLOW_TRACKING_URI` to Azure ML workspace
3. `python train.py --config config/train.yaml` — trains and logs to Azure ML Studio

### CI/CD

1. Pushes to `main` trigger the workflow
2. Local MLflow server starts for training
3. Model is trained, saved to `outputs/model.joblib`
4. Docker image is built with model baked in
5. Image pushed to ACR, deployed to App Service
6. API loads model from local file at startup

### Azure ML Studio validation

**Run:** `c8f77108-ecf0-4f74-9724-264c175a157c`  
**Experiment:** `water-potability-mlops`  
**Model:** `water_potability_model` → version 4 → Production  
**Metrics logged:** test_accuracy, test_f1_score, test_precision, test_recall, train_accuracy, train_f1_score  
**Artifacts:** model, model_metadata.json  
**View:** https://southeastasia.api.azureml.ms/mlflow/v2.0/subscriptions/20039353-4ec4-44c9-8b25-bcebfe540d05/resourceGroups/mlops-rg/providers/Microsoft.MachineLearningServices/workspaces/water-potability-mlw/#/experiments/9b732cf4-d3ec-4465-9284-daa6a04dcb5b/runs/c8f77108-ecf0-4f74-9724-264c175a157c

### Remaining risks

- **CI Azure auth:** `AZURE_CREDENTIALS` secret is empty — CI trains with local MLflow. To track CI runs in Azure ML Studio, a service principal must be created and set as `AZURE_CREDENTIALS`.
- **No unit tests:** Only manual test scripts exist.
- **In-memory stats:** Prediction stats reset on container restart.

---

## Legacy

`modeling/`, `config/train_config.yaml`, `config.py`, `get_model_for_production.py`, `test_post_request.py`, and `kubernetes_deployment/` are preserved for reference.
