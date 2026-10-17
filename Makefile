# OCP video pipeline. Run `make help`.
BACKEND := backend
CLI     := cd $(BACKEND) && uv run python -m app.cli
PORT    ?= 8000
Q       ?=
K       ?= 8
DIR     ?= data/poc20

.DEFAULT_GOAL := help
.PHONY: help install dev db-up db-down db-reset db-init db-psql ingest-plan ingest ingest-retry \
        poc20 load embed stats search lint tf-init tf-plan tf-apply

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---- setup / run ----
install: ## Install backend dependencies
	cd $(BACKEND) && uv sync

dev: db-up ## Start Postgres and the API with reload (http://localhost:$(PORT)/docs)
	cd $(BACKEND) && uv run uvicorn main:app --reload --port $(PORT)

# ---- database (local pgvector on :5433) ----
db-up: ## Start local Postgres + pgvector and apply the schema
	cd $(BACKEND) && docker compose up -d --wait
	$(CLI) db-init

db-down: ## Stop Postgres (data is kept)
	cd $(BACKEND) && docker compose down

db-reset: ## Delete all Postgres data and recreate the schema
	cd $(BACKEND) && docker compose down -v
	$(MAKE) db-up

db-init: ## Apply sql/001_init.sql (idempotent)
	$(CLI) db-init

db-psql: ## Open psql on the local database
	cd $(BACKEND) && docker compose exec db psql -U ocp -d ocp

# ---- ingest: YouTube -> S3 ----
ingest-plan: ## Count videos in the spreadsheet within MAX_AGE_DAYS
	$(CLI) plan

ingest: ## Download pending videos with yt-dlp and upload to S3 (8 workers)
	$(CLI) ingest --workers 8

ingest-retry: ## Retry failed downloads at low concurrency
	$(CLI) ingest --workers 2 --retry-failed

# ---- analysis: S3 -> Pegasus -> Kimi K3 -> records ----
poc20: ## Analyse the 20 hand-picked 800 VDC videos (~1h, ~$20; Pegasus limit is 3 concurrent)
	cd $(BACKEND) && uv run python -u scripts/poc20.py 20

# ---- load + query ----
load: ## Load analysis records into Postgres and S3 (DIR=data/poc20)
	$(CLI) load $(DIR)

embed: ## Embed chunks that have no embedding yet (Bedrock Titan v2)
	$(CLI) embed

stats: ## Row counts in Postgres
	$(CLI) stats

search: ## Hybrid search, e.g. make search Q="solid state transformer" K=5
	@test -n "$(Q)" || { echo 'usage: make search Q="your query" [K=8]'; exit 1; }
	$(CLI) search "$(Q)" -k $(K)

# ---- quality ----
lint: ## ruff check
	cd $(BACKEND) && uv run ruff check .

# ---- AWS (terraform uses your `aws login` session via exported credentials) ----
tf-init: ## terraform init
	cd terraform && terraform init -input=false

tf-plan: ## terraform plan
	cd terraform && eval "$$(aws configure export-credentials --format env)" && terraform plan -input=false

tf-apply: ## terraform apply (asks for confirmation)
	cd terraform && eval "$$(aws configure export-credentials --format env)" && terraform apply
