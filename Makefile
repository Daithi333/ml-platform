SHELL := /bin/bash

.PHONY: help up up-mlflow up-models up-airflow up-all down down-volumes build ps logs test test-unit test-api test-contract test-integration test-all lint format format-check typecheck pre-commit-install pre-commit-run train-classifier train list-models

help:
	@echo "Infrastructure:"
	@echo "  make up               - start platform API"
	@echo "  make up-mlflow        - start platform API + MLflow"
	@echo "  make up-models        - start platform API + MLflow + model servers"
	@echo "  make up-airflow       - start platform API + Airflow"
	@echo "  make up-all           - start everything"
	@echo "  make down             - stop stack"
	@echo "  make down-volumes     - stop stack and remove volumes"
	@echo "  make build            - rebuild images"
	@echo "  make ps               - show containers"
	@echo "  make logs             - tail logs"
	@echo ""
	@echo "ML:"
	@echo "  make train-classifier - train the newsgroups classifier"
	@echo "  make train MODEL=x    - train any model by config name"
	@echo "  make list-models      - list available model configs"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - run unit + api + contract tests (no Docker needed)"
	@echo "  make test-unit        - run unit tests only"
	@echo "  make test-api         - run api tests only"
	@echo "  make test-contract    - run contract tests only"
	@echo "  make test-integration - run integration tests (Docker required)"
	@echo "  make test-all         - run all tests inside Docker"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - run ruff linter"
	@echo "  make format           - format code with ruff"
	@echo "  make format-check     - check code formatting"
	@echo "  make typecheck        - run type checker"
	@echo "  make pre-commit-install - install pre-commit hooks"
	@echo "  make pre-commit-run   - run pre-commit on all files"
	@echo ""

up:
	docker compose up -d

up-mlflow:
	docker compose --profile mlflow up -d

up-models:
	docker compose --profile mlflow --profile models up -d

up-airflow:
	docker compose --profile airflow up -d

up-all:
	docker compose --profile mlflow --profile models --profile airflow up -d

down:
	docker compose --profile mlflow --profile models --profile airflow down

down-volumes:
	docker compose --profile mlflow --profile models --profile airflow down -v

build:
	docker compose build --no-cache

ps:
	docker compose ps -a

logs:
	docker compose logs -f --tail=200

test:
	uv run pytest tests/unit tests/api tests/contract -v

test-unit:
	uv run pytest tests/unit -v

test-api:
	uv run pytest tests/api -v

test-contract:
	uv run pytest tests/contract -v

test-integration:
	docker compose exec platform-api pytest tests/integration -v

test-all:
	docker compose exec platform-api pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src

pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

train-classifier:
	uv run python -m src.models.train --config newsgroups-classifier

train:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make train MODEL=<config-name>"; exit 1; fi
	uv run python -m src.models.train --config $(MODEL)

list-models:
	uv run python -m src.models.train --list
