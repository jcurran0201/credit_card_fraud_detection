FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY Deployment_code.py .
COPY fraud_rf_model.joblib .
ENV FRAUD_TRAIN_CSV=fraudTrain.csv \
    FRAUD_TEST_CSV=fraudTest.csv \
    FRAUD_MODEL_PATH=fraud_rf_model.joblib \
    PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn Deployment_code:app --host 0.0.0.0 --port ${PORT} --workers 1"]
