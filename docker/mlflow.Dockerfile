FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.12.2 && \
    mkdir -p /home/mlflow

EXPOSE 5001

ENV MLFLOW_HOME=/home/mlflow

CMD mlflow server \
      --host 0.0.0.0 --port 5001 \
      --backend-store-uri sqlite:///${MLFLOW_HOME}/mlflow.db \
      --default-artifact-root ${MLFLOW_HOME}/artifacts
