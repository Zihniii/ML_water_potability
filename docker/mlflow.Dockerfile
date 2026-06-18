FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_HOME=/home/mlflow
# Default artifact root — overridable via App Service env var MLFLOW_ARTIFACT_ROOT
ENV MLFLOW_ARTIFACT_ROOT=wasbs://mlflow-artifacts@waterpotabilitystorage.blob.core.windows.net/

# azure-storage-blob required for wasbs:// artifact URIs
RUN pip install --no-cache-dir mlflow==2.12.2 azure-storage-blob

RUN mkdir -p /home/mlflow/artifacts

EXPOSE 5001

CMD mlflow server \
      --host 0.0.0.0 --port 5001 \
      --backend-store-uri sqlite:///${MLFLOW_HOME}/mlflow.db \
      --default-artifact-root ${MLFLOW_ARTIFACT_ROOT}