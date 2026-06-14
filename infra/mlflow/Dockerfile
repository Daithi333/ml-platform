FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libpq-dev \
        && rm -rf /var/lib/apt/lists/*

COPY requirements-mlflow.txt /tmp/requirements-mlflow.txt

RUN pip install --no-cache-dir -r /tmp/requirements-mlflow.txt

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
