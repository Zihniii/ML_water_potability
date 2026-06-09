# Water Potability — MLOps Pipeline

End-to-end MLOps project: train → version → register → deploy with GitHub Actions, Azure ML MLflow, and Azure Container Apps.

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Dataset     │────>│  GitHub Actions   │────>│  Azure Blob     │
│  dataset/     │     │  (CI/CD trigger)  │     │  datasets/      │
└──────────────┘     └──────┬───────────┘     └─────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  train.py        │
                    │  (Pipeline)      │
                    │ Imputer → RF     │
                    └──────┬───────────┘
                           │
                    ┌──────▼───────────┐     ┌─────────────────┐
                    │  Azure ML        │────>│  Model Registry  │
                    │  MLflow Tracking │     │  water_potability│
                    └──────────────────┘     │  _model (Prod)   │
                                             └────────┬────────┘
                                                      │
                                             ┌────────▼────────┐
                                             │  Azure Container│
                                             │  Apps (FastAPI) │
                                             │  /predict       │
                                             │  /predict-with- │
                                             │  stats          │
                                             │  /model-info    │
                                             └─────────────────┘
```

## Pipeline Flow

1. **Push to `main`** — triggers GitHub Actions when `dataset/`, `train.py`, `app.py`, `requirements.txt` or `Dockerfile` change.
2. **Generate dataset version** — timestamp (`YYYY-MM-DD_HHMMSS`) or manual override via `workflow_dispatch`.
3. **Azure login** — via `AZURE_CREDENTIALS` secret.
4. **Upload dataset to Blob** — versioned path (`<version>/water_potability.csv`) + `latest_water_potability.csv` alias.
5. **Setup Python 3.11** and install dependencies.
6. **Fetch MLflow URI** — dynamically from Azure ML Workspace via `az ml workspace show`.
7. **Train (`train.py`)** — `SimpleImputer(median)` → `RandomForestClassifier(200, balanced)`, logs params/metrics/model to MLflow, registers model as `water_potability_model`.
8. **ACR build & push** — `water-potability-api:${{ github.sha }}`.
9. **Deploy to Container Apps** — update image with env vars.

## Required GitHub Secrets

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | Azure service principal JSON (for `azure/login`) |
| `AZURE_RESOURCE_GROUP` | Azure resource group name |
| `AZURE_STORAGE_ACCOUNT` | Blob storage account name |
| `AZURE_ACR_NAME` | Azure Container Registry name (no `.azurecr.io`) |
| `AZURE_CONTAINER_APP_NAME` | Container App name |
| `AZURE_CONTAINER_APP_ENV` | Container App Environment name |
| `AZURE_ML_WORKSPACE` | Azure ML Workspace name |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `POST` | `/predict` | Predict potability for one sample |
| `POST` | `/predict-with-stats` | Predict + return model metadata + cumulative stats |
| `GET` | `/prediction-stats` | Cumulative stats since container start |
| `GET` | `/model-info` | Loaded model metadata |

### Example request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.5,
    "Sulfate": 330.0,
    "Conductivity": 400.0,
    "Organic_carbon": 14.0,
    "Trihalomethanes": 70.0,
    "Turbidity": 3.5
  }'
```

### Example response

```json
{"prediction": 1, "label": "Potable"}
```

## Local Docker

```bash
docker build -t water-potability-api .
docker run -p 5000:5000 -t water-potability-api
```

Requires `MLFLOW_TRACKING_URI` env var at runtime (the model loads from the registry).

---

## Legacy

Original training (`modeling/`), config (`config.py`), production model script (`get_model_for_production.py`), and Kubernetes deployment are preserved for reference.

## Dataset

Source: [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets/tree/master)  
~3.2K samples, binary classification for water potability.
