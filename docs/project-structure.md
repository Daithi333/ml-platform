# Project Structure

## Philosophy

This is an MLOps **system**, not a model training project. The models are trivial payloads
used to exercise the platform. The value is in the infrastructure: registry, deployment
strategies, monitoring, retraining automation, governance.

The code is not an installable Python package. It is a collection of services, scripts, and
configs that are orchestrated independently (by Airflow, CI/CD, Docker, etc.).

## Layout

```
ml-platform/
  README.md
  STRUCTURE.md
  Dockerfile                # Multi-stage: base (prod deps), dev (+ test deps), final (prod)
  pyproject.toml            # uv project, shared deps, dev tooling
  uv.lock
  compose.yml              # Local dev stack with profiles
  Makefile                  # Dev convenience commands

  src/
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
      client.py             # RegistryClient class (load, list, cache)
      factory.py            # Factory function wiring settings -> client

    serving/                # FastAPI inference API
      app.py                # App factory with lifespan
      dependencies.py       # FastAPI DI (settings, registry)
      routers/
        health.py
        predict.py
        registry.py
      schemas/
        errors.py
        health.py
        predict.py
      services/             # Business logic (no HTTP concerns)
        health.py
        predict.py
        registry.py
      middleware/           # Traffic splitting, canary logic (later)

    features/               # Feature store abstraction (later)
    monitoring/             # Drift detection, metrics (later)
    deployment/             # Deployment strategies (later)

    config.py               # Centralised config (pydantic-settings, prefixed groups)
    exceptions.py           # Domain error types
    logs.py                 # Structured logging setup

  data/                     # Local training data (gitignored)

  infra/
    airflow/                # Airflow (isolated, own Dockerfile + deps)
      Dockerfile
      entrypoint.sh
      requirements-airflow.txt
      dags/
    mlflow/                 # MLflow server (own Dockerfile + deps)
      Dockerfile
      requirements-mlflow.txt
    cdk/ or terraform/      # AWS IaC (later phases)

  ui/                       # React frontend (later)
    src/
      features/
        predict/
        registry/
        monitoring/
      components/
      hooks/
      services/
      models/
    package.json
    vite.config.ts
    tsconfig.json

  tests/
    api/                    # API / router tests
    integration/            # Integration tests (with real services)

  docs/                     # Learning notes and reference docs

  .github/
    workflows/
      ci.yml                # Lint + test pipeline
```

## Key Decisions

1. **No installable package** -- `src/` is a flat collection of concerns, not a Python
   package with `__init__.py` files. Each subdirectory is invoked independently.

2. **Models are trivial** -- scikit-learn classifiers trained in seconds. The system
   handles them; they don't need to be impressive.

3. **Config-driven training** -- model configs are YAML files. Same architecture code
   serves different datasets. Add a new model by writing a YAML, not Python.

4. **Airflow and MLflow are isolated** -- own directories, own Dockerfiles, own deps.
   They don't pollute the app's dependency tree.

5. **Local dev simulates prod** -- FastAPI stands in for SageMaker endpoints, Docker
   volumes for S3, Postgres for RDS. Same patterns, no AWS cost.

6. **Deployment strategies are code** -- canary, blue-green, shadow mode will be
   implemented as reusable modules, not ad-hoc scripts.

7. **Config uses explicit prefixes** -- `MLFLOW__TRACKING_URI`, `DATA__ROOT`. Each
   settings group has its own env prefix to avoid collisions.

8. **Factory + DI pattern** -- clients are classes with settings injected via constructor.
   Factories wire them. FastAPI DI injects them into routes via `app.state` + lifespan.

9. **React UI as a separate app** -- lives in `ui/`, own `package.json`, own container.
   Vite + TypeScript + React. API-first: OpenAPI spec enables typed client generation.

10. **Docker profiles** -- `make up` (API only), `make up-mlflow` (+ MLflow),
    `make up-airflow` (+ Airflow), `make up-all` (everything).

## Dependencies

### Core (pyproject.toml)
- mlflow -- registry, experiment tracking, model versioning
- pydantic-settings -- environment-specific config
- structlog -- structured logging
- fastapi + uvicorn -- API serving
- scikit-learn, pandas, numpy -- model training
- pyyaml -- model config loading
- redis -- online feature store (later)

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
