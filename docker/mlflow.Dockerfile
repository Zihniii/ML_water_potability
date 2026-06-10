FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.12.2

EXPOSE 5001

VOLUME ["/mlflow"]

CMD ["mlflow", "server", \
      "--host", "0.0.0.0", "--port", "5001", \
      "--backend-store-uri", "sqlite:///mlflow/mlflow.db", \
      "--default-artifact-root", "/mlflow/artifacts"]
