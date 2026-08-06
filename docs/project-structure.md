# Project Structure

## Philosophy

This is an MLOps **system**, not a model training project. The models are trivial payloads
used to exercise the platform. The value is in the infrastructure: registry, deployment
strategies, monitoring, retraining automation, governance.

The code is not an installable Python package. It is a collection of services, scripts, and
configs that are orchestrated independently (by Airflow, CI/CD, Docker, etc.).

## Architecture

Two functionally separate apps deployed to separate containers:

```
Client
  |
  v
Platform API (:8000)                    Model Server (:8001 per model)
  - Management plane                      - Inference plane
  - Registry browsing                     - Loads one model at startup
  - Inference routing (proxies)           - Serves /predict + /health
  - Health, monitoring                    - Single worker, model in memory
  |                                       - Horizontal scaling per model
  +--- HTTP ---> model-newsgroups:8001
  +--- HTTP ---> model-fraud:8001 (later)
  +--- HTTP ---> model-forecast:8001 (later)
```

In production, each model server becomes its own ECS Service / K8s Deployment.
The platform API becomes the gateway that routes to the correct model service
(or is replaced by path-based routing at the load balancer).

## Layout

```
ml-platform/
  README.md
  pyproject.toml            # uv project, shared deps, dev tooling
  uv.lock
  compose.yml              # Local dev stack with profiles
  Makefile                  # Dev convenience commands

  src/
    serving/                # Platform API (management + routing)
      app.py                # FastAPI app with lifespan
      dependencies.py       # DI (settings, registry client)
      routers/
        health.py
        predict.py          # Proxies to model server containers
        registry.py
      schemas/
        errors.py
        health.py
        predict.py
        registry.py
      services/
        health.py
        predict.py          # HTTP client to model servers
        registry.py

    model_server/           # Model Server (inference, one model per container)
      app.py                # FastAPI app, loads model at startup
      config.py             # MODEL_NAME, MLFLOW_TRACKING_URI from env
      loader.py             # Loads model from MLflow registry
      schemas.py            # Request/response schemas

    models/                 # Model training (config-driven, dataset-agnostic)
      architectures/        # Reusable model architectures (the code)
        text_classifier.py  # TF-IDF + LogReg pipeline builder
      configs/              # Model instance definitions (YAML)
        newsgroups-classifier.yaml
      datasets/             # Pluggable data loaders (sklearn, csv, parquet, s3)
        loaders.py
      schema.py             # Pydantic schema for model config validation
      train.py              # Generic CLI entry point (reads config, dispatches)

    registry/               # Model registry client (wraps MLflow)
      client.py             # RegistryClient class
      factory.py            # Factory: settings -> client

    config.py               # Centralised config (pydantic-settings, prefixed groups)
    exceptions.py           # Domain error types
    logs.py                 # Structured logging setup

  data/                     # Local training data (gitignored)

  infra/
    platform-api/           # Platform API Dockerfile (dev + prod targets)
      Dockerfile
    model-server/           # Model server Dockerfile (dev + prod targets)
      Dockerfile
    airflow/                # Airflow (isolated, own Dockerfile + deps)
      Dockerfile
      entrypoint.sh
      requirements-airflow.txt
      dags/
    mlflow/                 # MLflow tracking server
      Dockerfile
      requirements-mlflow.txt

  ui/                       # React frontend (later)

  tests/
    api/                    # Platform API tests
    integration/            # Integration tests

  docs/                     # Learning notes and reference docs

  .github/
    workflows/
      ci.yml
```

## Key Decisions

1. **Two apps, two containers** -- Platform API (management/routing) and Model Server
   (inference) are separate FastAPI apps with separate Dockerfiles. This mirrors
   production where each model runs in its own container.

2. **Model server loads at startup** -- model is fetched from MLflow once during
   container boot and held in process memory. Single worker. Scaling is horizontal
   (more containers), not vertical (more workers).

3. **Platform API proxies inference** -- the predict endpoint makes an HTTP call to the
   model server. In production, this is replaced by load balancer routing or service mesh.

4. **Config-driven training** -- model configs are YAML files. Same architecture code
   serves different datasets. Add a new model by writing a YAML, not Python.

5. **One compose service per model** -- each model server is a separate compose service
   with the same image but different `MODEL_NAME` env var. Adding a model = adding a
   service definition.

6. **Isolated infrastructure** -- Airflow, MLflow, and model servers each have their own
   Dockerfiles and dependency sets. No cross-contamination.

7. **Config uses explicit prefixes** -- `MLFLOW__TRACKING_URI`, `DATA__ROOT`. Each
   settings group has its own env prefix to avoid collisions.

8. **Factory + DI pattern** -- clients are classes with settings injected via constructor.
   Factories wire them. FastAPI DI injects via `app.state` + lifespan.

9. **Docker profiles** -- `make up` (platform API), `make up-models` (+ MLflow + model
   servers), `make up-airflow` (+ Airflow), `make up-all` (everything).

10. **React UI as a separate app** -- lives in `ui/`, communicates with platform API
    via REST. API-first: OpenAPI spec enables typed client generation.

## Dependencies

### Core (pyproject.toml — shared by both apps)
- mlflow -- registry client, experiment tracking
- pydantic-settings -- environment config
- structlog -- structured logging
- fastapi + uvicorn -- both apps use FastAPI
- httpx -- platform API proxies to model servers
- scikit-learn, pandas, numpy -- model training + inference
- pyyaml -- model config loading

### Isolated (own Dockerfiles)
- Airflow -- `infra/airflow/requirements-airflow.txt`
- MLflow server -- `infra/mlflow/requirements-mlflow.txt`

### Dev tooling (dependency-groups in pyproject.toml)
- pytest, pytest-asyncio, pytest-cov
- ruff, mypy
- pre-commit

### Added later (by phase)
- boto3, sagemaker -- phase 7 (AWS deployment)
- confluent-kafka -- phase 9 (streaming)
- evidently or alibi-detect -- phase 5 (drift detection)
- prometheus-client -- phase 5 (metrics)
