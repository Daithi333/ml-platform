SHELL := /bin/bash

.PHONY: help up up-mlflow up-airflow up-all down down-volumes build ps logs test test-api test-local lint format format-check typecheck pre-commit-install pre-commit-run train-classifier train list-models

help:
	@echo "Infrastructure:"
	@echo "  make up               - start core (API only)"
	@echo "  make up-mlflow        - start core + MLflow stack"
	@echo "  make up-airflow       - start core + Airflow stack"
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
	@echo "  make test             - run all tests (Docker)"
	@echo "  make test-api         - run api tests (Docker)"
	@echo "  make test-local       - run all tests (local, no Docker)"
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

up-airflow:
	docker compose --profile airflow up -d

up-all:
	docker compose --profile mlflow --profile airflow up -d

down:
	docker compose --profile mlflow --profile airflow down

down-volumes:
	docker compose --profile mlflow --profile airflow down -v

build:
	docker compose build --no-cache

ps:
	docker compose ps -a

logs:
	docker compose logs -f --tail=200

test:
	docker compose exec api pytest tests/ -v

test-api:
	docker compose exec api pytest tests/api -v

test-local:
	uv run pytest tests/ -v

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
