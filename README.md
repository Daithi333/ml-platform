# ML Platform (Working Title: `ml-platform`)

A production-grade MLOps system demonstrating the full lifecycle of deploying, monitoring, and managing multiple ML models across environments. Built on AWS (Bedrock, SageMaker) with local development alternatives.

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment config
cp .env.example .env

# Start MLflow + model server stack
make up-models

# Train the newsgroups classifier
make train-classifier

# View experiment in MLflow UI
open http://localhost:5001

# Test prediction (platform API proxies to model server)
curl -X POST http://localhost:8000/api/v1/models/newsgroups-classifier/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["NASA launched a new satellite into orbit today"]}'

# Browse model registry
curl http://localhost:8000/api/v1/registry/models

# Run tests
make test              # unit + api + contract (no Docker needed)
make test-integration  # full stack (Docker required)
```

The model server loads its model at startup and fails fast if none is registered. On a fresh
stack, train first, then restart it: `docker compose restart model-newsgroups`.

## Train and Promote (Airflow)

The `train_and_promote` DAG trains a model, evaluates it against the `promotion:` criteria in
its YAML config, and sets the MLflow `@champion` alias if it passes.

```
run_training -> evaluate_run -> gate -+-> promote -+-> notify
                                      +-> reject  -+
```

```bash
make up-all
docker compose exec airflow-webserver airflow dags trigger train_and_promote \
  --conf '{"model_config": "newsgroups-classifier", "triggered_by": "manual"}'
open http://localhost:8080
```

A candidate is promoted only if it clears the absolute thresholds (`min_accuracy`,
`min_f1_macro`) and, when `require_improvement` is set, beats the current champion on
`primary_metric`. Notifications always go to the logs, and also to Slack when
`NOTIFICATIONS__SLACK_WEBHOOK_URL` is set.

New DAGs start paused. Unpause them in the UI before triggering.

### Picking up a new champion

The DAG promotes. It does not deploy. Promotion sets the `@champion` alias in MLflow, and the
model server loads `@champion` once at startup. After a promotion, restart the model server to
serve the new version:

```bash
docker compose restart model-newsgroups
curl http://localhost:8001/health
```

This separation is deliberate and mirrors production, where promotion (registry state) and
deployment (what is running) are distinct steps:

| Environment | How a new champion reaches serving |
|---|---|
| Local | Manual `docker compose restart` of the model server |
| EKS + GitOps | Promotion bumps the pinned version in a config repo; ArgoCD/Flux rolls the pods |
| SageMaker | Registry approval emits an EventBridge event; pipeline calls `UpdateEndpoint` (blue/green with auto-rollback) |

When the AWS phase arrives, deployment becomes a pluggable `Deployer` step after `promote`
(same pattern as notification clients), so the pipeline itself stays platform-agnostic.

### DAGs in production

Locally, DAGs are bind-mounted from `infra/airflow/dags` and Airflow always runs the latest file
on disk. Edits apply within ~30s, even to in-flight runs. Production should instead:

- Load DAGs from a versioned bundle (Git or S3), so each run is pinned to the commit it started
  on and a mid-run deploy cannot change its behaviour
- Ship DAGs through CI: lint, import-check (`airflow dags list-import-errors`), test, then
  publish a tagged version
- Roll back by pointing the bundle at the previous tag, not by editing files in place
- Version the DAG's code dependencies (`src/`) together with the DAG, since tasks import from it

## Current Status

- Two-app architecture: Platform API (management/routing) + Model Server (inference)
- Config-driven model training (YAML configs, pluggable architectures + datasets)
- MLflow experiment tracking and model registry (Postgres-backed, local artifacts)
- Platform API proxies inference to dedicated model server containers
- Model server loads model at startup, holds in memory (production pattern)
- Registry API (list models, inspect versions)
- Automated train/evaluate/promote pipeline in Airflow with pluggable notifications
  (promotion sets `@champion`; the model server picks it up on restart)
- Docker Compose with profiles (platform-api, models, mlflow, airflow)
- CI pipeline (lint + test via GitHub Actions)

### Known gaps

- `make list-models` fails because `--list` conflicts with the required `--config` argument
  group in `train.py`.
- Integration tests (`tests/integration/`) are not yet written.

## Philosophy

This is not a "train a model" project. It's a "run models in production" project. The focus is on the system around the models: deployment strategies, monitoring, retraining, governance, and operational excellence.

## Domain Options

The platform is designed to be domain-agnostic — the MLOps infrastructure works regardless of what models you're running. Start with one domain, evolve to the other when the platform is mature.

### Option A: Content Intelligence (simpler, familiar)

Builds on familiarity from the RAG project. Three model families for a content platform:

| Model | Type | Serving Pattern | Retraining |
|-------|------|-----------------|------------|
| **Content Classifier** | Text classification (topic/quality) | Real-time inference | Weekly on new labelled data |
| **Engagement Predictor** | Regression (predicted reads/likes) | Batch scoring | Daily on engagement metrics |
| **Anomaly Detector** | Unsupervised (spam/bot detection) | Streaming pipeline | Triggered on drift detection |

Good for: getting the platform running quickly with lightweight models and familiar data.

### Option B: E-Commerce (richer, more realistic)

Three model families with genuinely different SLAs, data sources, and failure modes:

| Model | Type | Serving Pattern | Retraining | SLA |
|-------|------|-----------------|------------|-----|
| **Product Recommendations** | Collaborative filtering + embeddings | Real-time (per page load) | Daily on interaction data | <100ms p99 |
| **Fraud Detection** | Classification (transaction risk scoring) | Streaming (per transaction) | Triggered on precision drift | <50ms p99 |
| **Demand Forecasting** | Time series regression (inventory planning) | Batch (nightly) | Weekly on sales data | Next-day delivery |

Good for: demonstrating why you need different deployment strategies, monitoring approaches, and retraining cadences. Each model has genuinely different operational characteristics.

### Recommended Path

Start with Option A (one model, content classifier) to build the platform foundation. Once the infrastructure is solid (registry, CI/CD, monitoring, deployment strategies), migrate to Option B where the operational complexity justifies the platform's existence. The platform itself doesn't change — only the models plugged into it.

## Architecture Layers

### 1. Model Development
- Experiment tracking (MLflow or SageMaker Experiments)
- Reproducible training pipelines (SageMaker Pipelines or Step Functions)
- Hyperparameter tuning (SageMaker HyperBand)
- Local training with same code (Docker-based)

### 2. Feature Store
- Feature definitions with versioning and governance
- Online store (low-latency serving) + Offline store (training)
- AWS: SageMaker Feature Store
- Local: Redis (online) + Parquet files (offline)

### 3. Model Registry
- Model versioning with metadata and lineage
- Approval workflow (staging -> production)
- AWS: SageMaker Model Registry
- Local: MLflow Model Registry

### 4. Deployment Strategies
- **Real-time endpoint** (Content Classifier) — SageMaker endpoint with auto-scaling
- **Batch transform** (Engagement Predictor) — scheduled SageMaker Batch Transform
- **Streaming** (Anomaly Detector) — Kinesis + Lambda or SageMaker endpoint behind Kinesis
- **A/B testing** — traffic splitting between model versions on the same endpoint
- **Shadow mode** — new model receives traffic but responses aren't served to users
- **Blue-green** — instant cutover with rollback capability
- **Canary** — gradual traffic shift with automated rollback on metric degradation

### 5. Monitoring and Observability
- Model performance metrics (accuracy, latency, throughput)
- Data drift detection (input distribution shift)
- Prediction drift (output distribution shift)
- Custom CloudWatch dashboards per model
- Alerting on SLA breaches
- AWS: SageMaker Model Monitor + CloudWatch
- Local: Prometheus + Grafana

### 6. Automated Retraining
- Drift-triggered retraining (anomaly detector)
- Scheduled retraining (engagement predictor — daily)
- Performance-triggered retraining (classifier accuracy drops below threshold)
- Automated evaluation gate before promotion

### 7. CI/CD
- Infrastructure as Code (Terraform or CDK)
- Model CI: train -> evaluate -> register
- Model CD: promote -> deploy -> validate -> monitor
- Environment-specific configs (dev/staging/prod)
- Rollback automation

### 8. Governance
- Model cards (documentation per model version)
- Data lineage (which data trained which model)
- Access control (who can promote to production)
- Audit trail (all deployments logged)

## Tech Stack

| Layer | AWS | Local Dev |
|-------|-----|-----------|
| Training | SageMaker Training Jobs | Docker + local GPU/CPU |
| Feature Store | SageMaker Feature Store | Redis + Parquet |
| Model Registry | SageMaker Model Registry | MLflow |
| Serving (RT) | SageMaker Endpoints | FastAPI + Docker |
| Serving (Batch) | SageMaker Batch Transform | Local script |
| Serving (Stream) | Kinesis + Lambda | Kafka + local consumer |
| Monitoring | SageMaker Model Monitor + CloudWatch | Prometheus + Grafana |
| Orchestration | Step Functions / SageMaker Pipelines | Airflow |
| IaC | CDK or Terraform | Docker Compose |
| CI/CD | CodePipeline or GitHub Actions | GitHub Actions |

## Cost Management Strategy

- **Development**: Entirely local (Docker, MLflow, FastAPI mock endpoints)
- **Validation**: Short-lived AWS resources, torn down after each session
- **Persistent**: Only S3 buckets and ECR repos (pennies/month)
- **Expensive resources** (endpoints, training jobs): Created on-demand, destroyed after validation
- **Budget alerts**: AWS Budgets set at $20/month hard cap
- **Spot instances**: For training jobs (60-90% cheaper)
- **Serverless where possible**: Lambda for inference on low-traffic models

## Suggested Implementation Order

1. **Local foundation** — Docker Compose, MLflow, FastAPI serving, one model (classifier)
2. **Feature store** — Redis online + Parquet offline, feature versioning
3. **Model registry** — MLflow with approval workflow
4. **CI/CD** — GitHub Actions: train -> evaluate -> register
5. **Monitoring** — Prometheus + Grafana locally, drift detection
6. **Second model** — Batch scoring (engagement predictor), different deployment pattern
7. **AWS deployment** — CDK/Terraform, SageMaker endpoint for classifier
8. **A/B testing** — Traffic splitting between model versions
9. **Third model** — Streaming (anomaly detector), Kinesis/Kafka
10. **Retraining automation** — Drift triggers, scheduled retraining
11. **Shadow mode + canary** — Advanced deployment strategies
12. **Governance** — Model cards, lineage, audit

## Key Principles

- Every AWS resource has a local equivalent for development
- Infrastructure is code (no ClickOps)
- Models are versioned artefacts, not code
- Deployment is separate from training
- Monitoring is not optional
- Rollback is always possible
- Cost is a first-class constraint
