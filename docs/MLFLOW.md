# MLflow — MLOps Tracking & Registry

## Architecture

```
                    ┌──────────────────────┐
                    │   Azure ML Studio    │
                    │  (MLflow Tracking +  │
                    │   Model Registry)    │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Local Dev   │    │  CI/CD (GH)  │    │ App Service  │
   │ train.py    │    │  train.py    │    │   FastAPI    │
   │ (az login)  │    │ (local MLf)  │    │ model.joblib│
   │ Azure ML URI│    │ baked model  │    │   (baked)   │
   └─────────────┘    └──────────────┘    └──────────────┘
```

- **Local development**: trains logs directly to Azure ML Studio via `DefaultAzureCredential` (uses your `az login` session). Runs, metrics, parameters, and artifacts appear in Azure ML Studio.
- **CI/CD**: trains with a local MLflow server (ephemeral). The trained model is saved as `outputs/model.joblib` and baked into the Docker image.
- **API**: tries to load from MLflow registry first; falls back to local `model.joblib` if registry fails.

## How This Project Uses MLflow

### Local Development (tracks to Azure ML)

```bash
# Authenticate
az login

# Set tracking URI to Azure ML workspace
$env:MLFLOW_TRACKING_URI="azureml://<workspace-uri>"

# Train — logs appear in Azure ML Studio
python train.py --config config/train.yaml
```

### CI/CD Pipeline (local MLflow)

The CI workflow starts a local MLflow server (`http://127.0.0.1:5001`), trains the model, logs to local MLflow, then saves the model as `outputs/model.joblib`. The Docker image is built with the model baked in. No cloud MLflow auth needed.

### Serving (`app.py`)

```python
# 1. Try MLflow registry (works in local dev with Azure ML URI)
if MLFLOW_TRACKING_URI:
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE_OR_ALIAS])
    pipeline = mlflow.sklearn.load_model(model_uri)

# 2. Fallback: load baked model.joblib (works in CI/deployed API)
if pipeline is None:
    pipeline = joblib.load("outputs/model.joblib")
```

## Local vs Cloud (Development vs CI/CD)

| Aspect | Local Development | CI/CD Pipeline |
|---|---|---|
| MLflow backend | Azure ML Workspace | Local server (`http://127.0.0.1:5001`) |
| Auth method | `az login` (user session) | None (local server) |
| Run visibility | Azure ML Studio | Local only (ephemeral) |
| Artifact storage | Azure Blob (workspace-linked) | `mlruns/` (ephemeral) |
| Model delivery | Not used in CI | Baked into Docker image |
| Suitability | Experimentation & tracking | Automated deployment |

## Model Stages & Promotion

Models move through stages:
1. **None** — freshly registered, unassigned
2. **Staging** — validated, ready for pre-production
3. **Production** — actively serving via API
4. **Archived** — superseded by newer version

The training pipeline (`train.py`) includes automatic promotion logic:
- After training, compares the current `test_f1_score` against the existing Production model
- Promotes to Production if the metric is >= current production model
- Archives the previous Production version

Promotion works in both local and CI modes (both access the MLflow registry).

## Key Commands

```bash
# 1. Local development with Azure ML tracking
az login
$env:MLFLOW_TRACKING_URI="azureml://<workspace-uri>"
python train.py --config config/train.yaml

# 2. Fetch tracking URI from Azure ML
az ml workspace show \
  --resource-group <rg> \
  --name <workspace> \
  --query mlflow_tracking_uri -o tsv

# 3. Transition model stage (if needed)
mlflow models --version <version> stage Production

# 4. View local MLflow experiments
mlflow experiments list
```

## Common Issues

1. **Model not found at startup**: If using MLflow registry, ensure a model version exists in the Production stage. If using the baked model, ensure `outputs/model.joblib` exists in the image.

2. **`az login` expired**: Local dev requires an active Azure session. Run `az login` again.

3. **Azure ML extension not installed**:
   ```bash
   az extension add --name ml -y
   ```

4. **CI failing on Azure auth**: The CI pipeline does NOT authenticate to Azure ML. It uses a local MLflow server and bakes the model into the Docker image. To enable CI→Azure ML tracking, set up a service principal as `AZURE_CREDENTIALS`.

5. **Registered model name mismatch**: `train.py` registers as `water_potability_model`, `app.py` loads the same name. Both use env `MODEL_NAME` for override.
