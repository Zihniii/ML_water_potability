FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_HOME=/home/mlflow

RUN pip install --no-cache-dir mlflow==2.12.2

RUN mkdir -p /home/mlflow/artifacts

EXPOSE 5001

CMD ["mlflow","server","--host","0.0.0.0","--port","5001","--backend-store-uri","sqlite:////home/mlflow/mlflow.db","--default-artifact-root","/home/mlflow/artifacts"]