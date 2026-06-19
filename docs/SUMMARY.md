# MLOps Summary — Water Potability

## What We Built

End-to-end MLOps pipeline with **no Entra ID / service principal** — all auth via storage account keys + ACR admin credentials.

### Components

| Layer | Tech | What it does |
|---|---|---|
| Source control | GitHub | Trigger CI on push to `main` |
| CI/CD | GitHub Actions (`mlops-key-auth.yml`) | Validate → train → upload artifacts → build image → deploy |
| Experiment tracking | **Remote MLflow server** on Azure App Service | Persists runs, metrics, params across CI runs (no ephemeral local server) |
| Model registry | MLflow Model Registry inside the same server | Versioned model entries, stage promotion (Production / Archived) |
| Artifact storage | **Azure Blob Storage** (`waterpotabilitystorage`) | Dataset + model.joblib + metadata, versioned by timestamp |
| Container registry | **Azure Container Registry** (waterpotabilityacr) | Immutable SHA-tagged images |
| API serving | **FastAPI** on Azure App Service (`water-potability-api`) | Predict, health, stats, model-info |
| Frontend | Next.js on Azure Static Web Apps | UI for predictions |

### Pipeline flow

```
push → validate → upload dataset to Blob → train model
  → log params/metrics to remote MLflow → register model version
  → save model.joblib + model_metadata.json → upload to Blob
  → build Docker image → push to ACR → deploy to App Service
```

## Repeated Challenge: MLflow Server App Configuration

### The challenge: keeping the remote MLflow container alive and healthy

The MLflow server runs as a custom Docker container on Azure App Service. Several misconfigurations caused it to crash or become unreachable.

### Problem 1: `appCommandLine` overrides Dockerfile CMD

Azure App Service has an `appCommandLine` setting that, when non-empty, replaces the Dockerfile's `CMD`.

```
az webapp config set --acr-use-identity false    # sets appCommandLine to " "
```

A space character as `appCommandLine` causes the container to try executing `" "` as the startup command instead of `CMD mlflow server ...`. The container crashes immediately → App Service returns 503.

**Fix:** Clear `appCommandLine` via the Azure REST API:

```powershell
Invoke-RestMethod -Method Put -Uri ".../config/web?api-version=2023-01-01" \
  -Body '{"properties":{"appCommandLine":""}}'
```

`az webapp config set --startup-file ""` does not accept an empty string.

### Problem 2: `WEBSITES_PORT` must match the container's listening port

The MLflow Dockerfile starts the server on port **5001** (`--port 5001`). If `WEBSITES_PORT` is unset or set to the default 5000/80, Azure's load balancer cannot route traffic to the container → timeout / 503.

**Fix:** Always set `WEBSITES_PORT=5001` in App Service application settings.

### Problem 3: SQLite database lost on restart

Without `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`, the App Service's `/home/` directory is ephemeral — SQLite data disappears on every restart.

**Fix:** Set `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true` so MLflow's `sqlite:////home/mlflow/mlflow.db` persists.

### Problem 4: `wasbs://` artifact root crashes the container

Attempting to use Azure Blob as the artifact store by setting `--default-artifact-root wasbs://mlflow-artifacts@...` required `azure-storage-blob` in the Dockerfile and the `AZURE_STORAGE_ACCOUNT` / `AZURE_STORAGE_KEY` environment variables present at MLflow server startup. MLflow would crash during initialization when trying to connect to WASBS — possibly due to missing or misconfigured credentials at that point in the startup sequence.

**Fix:** Revert to local artifact root (`/home/mlflow/artifacts`) in the original Dockerfile. Don't use MLflow's artifact store at all — CI handles Blob upload directly.

### Problem 5: Image tag changes via Azure CLI are unreliable

Commands like `az webapp config set --generic-configurations` and `az webapp config container set` frequently timed out or produced `BadRequest` errors. The `az rest` PATCH endpoint was also fragile with JSON body parsing in PowerShell.

**Fix:** Use `Invoke-RestMethod` with a direct bearer token from `az account get-access-token` to PATCH the `config/web` resource:

```powershell
$token = az account get-access-token --resource https://management.azure.com --query accessToken -o tsv
$body = '{"properties":{"linuxFxVersion":"DOCKER|waterpotabilityacr.azurecr.io/mlflow-server:latest"}}'
Invoke-RestMethod -Uri ".../config/web?api-version=2023-01-01" -Method Put -Headers @{"Authorization"="Bearer $token"} -Body $body
```

### Problem 6: ACR credentials scrambled during configuration changes

Running `az webapp config container set` or `az webapp config set` can corrupt the stored ACR password, causing the container to fail pulling the image on restart.

**Fix:** Re-retrieve the ACR password and re-apply it:

```bash
az acr credential show --name waterpotabilityacr --query "passwords[0].value" -o tsv
```

Then set it via App Service → Configuration → Application settings → `DOCKER_REGISTRY_SERVER_PASSWORD`.

### Lesson learned

**Never touch App Service settings that are working.** Every change (image tag, appCommandLine, container config) risks breaking the running server. The MLflow Docker image is stable; rebuild and deploy it only when the Dockerfile itself changes.

## Repeated Challenge: PermissionError on Artifact Logging

### The problem

`mlflow.log_artifact()` and `mlflow.sklearn.log_model()` failed with:

```
PermissionError: [Errno 13] Permission denied: '/home/mlflow'
```

**Why:** The remote MLflow server's `--default-artifact-root` is `/home/mlflow/artifacts` — a local path inside the App Service container. When CI runs `mlflow.log_artifact()`, it asks the remote server (via REST) to write the artifact. The App Service container runs as a non-root user with no write permission to `/home/mlflow/`.

Attempted fixes that didn't work:
- Adding `azure-storage-blob` to the Dockerfile and using `wasbs://` artifact root (crashed the container at startup)
- Trying to change permissions in the Dockerfile
- Local `az storage blob upload` from CI → can't reach the container's filesystem

### The fix: bypass MLflow artifacts entirely

**Decision:** Skip MLflow's built-in artifact store. CI handles model persistence directly via `az storage blob upload` — same method already used for dataset uploads.

**Changes:**
1. Removed `mlflow.log_artifact()` and `mlflow.sklearn.log_model()` from `train.py`
2. CI uploads `model.joblib` and `model_metadata.json` to Blob manually (versioned + latest)
3. API loads model from Blob at startup, not from MLflow
4. Model version registered in MLflow registry with a `blob://` source URI (metadata only)

**Result:** MLflow tracking (params, metrics, run metadata, model registry) still works via REST API. Artifacts live in Blob, uploaded directly by CI. No PermissionError.

## Why This Is End-to-End MLOps

### 1. Data versioning
- Dataset uploaded to Blob at `{timestamp}/water_potability.csv` + `latest_water_potability.csv`
- Every CI run creates a versioned snapshot

### 2. Experiment tracking
- All runs logged to persistent remote MLflow server
- Params (classifier, imputer, test_size, etc.) and metrics (accuracy, F1, precision, recall) recorded
- Feature importance logged for tree-based models

### 3. Model registry
- Each CI run registers a new model version via `mlflow.create_model_version()`
- `_promote_if_better()` compares current run's metric against Production and auto-promotes
- Previous Production versions are archived

### 4. Artifact versioning
- Model + metadata stored at `models/{timestamp}/model.joblib` (immutable) and `models/latest/` (overwritten)
- Enables rollback: point API to any previous version

### 5. CI/CD automation
- Single `git push` triggers: validate → data upload → train → register → save → build → deploy
- No manual steps between commit and production deployment

### 6. Reproducibility
- Training config is committed (YAML): classifier type, hyperparams, test split, etc.
- Grid search results are logged with best params
- Metrics include train AND test scores (overfitting detection)

### 7. Model serving
- FastAPI with endpoints: predict, predict-with-stats, health, model-info, prediction-stats
- Model loaded from Blob at startup (no MLflow dependency at runtime)
- Per-container prediction stats (daily counts, confidence, version distribution)

### 8. Monitoring
- `/model-info` returns: model name/version, classifier type, imputer, feature count, training metrics (accuracy, F1, precision, recall), test size, training timestamp, API uptime, prediction count
- `/prediction-stats` returns: cumulative counts, potable/not-potable breakdown, average confidence, daily breakdown

## Architecture

```
GitHub Push
    │
    ▼
GitHub Actions
    ├── validate (syntax + YAML)
    ├── upload dataset (timestamped + latest)
    ├── train.py ────────────────────────────────► Remote MLflow (App Service)
    │   ├── load dataset from Blob                     ├── log params
    │   ├── build pipeline (config-driven)             ├── log metrics
    │   ├── train + evaluate                           ├── register model version
    │   └── save model.joblib + metadata               └── promote to Production
    ├── upload artifacts to Blob ◄── key auth
    │   ├── models/{timestamp}/model.joblib
    │   ├── models/latest/model.joblib
    │   ├── models/{timestamp}/model_metadata.json
    │   └── models/latest/model_metadata.json
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

## Key Architecture Decisions

| Decision | Rationale |
|---|---|
| Remote MLflow server (not ephemeral local) | Ephemeral local server resets runs between CI → loses history, resets version numbers. Remote server persists everything. |
| Blob artifacts (not MLflow artifacts) | MLflow's artifact store requires write access to the server's filesystem or a properly configured WASBS backend. Direct Blob upload by CI is simpler and doesn't depend on MLflow's artifact plumbing. |
| Storage account key auth (not managed identity) | No Entra ID permissions available. Account key works for Blob uploads from CI. |
| Immutable SHA image tags (not `latest`) | Every CI build gets a unique tag. Rollback is trivial. Production is never accidentally broken by an untested push. |
| Model version in MLflow registry (not just Blob) | Gives a sequential version number, enables stage promotion logic, and keeps the registry as the source of truth for which version is in Production. |
