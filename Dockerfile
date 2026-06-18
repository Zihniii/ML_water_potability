FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -y && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./*.py .
COPY ./dataset/ ./dataset/
COPY ./outputs/ ./outputs/

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
