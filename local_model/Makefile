# ── Qwen3.5-9B · vLLM deployment
# Usage: make <target>
#
# Required tools: docker, docker compose, openssl, curl

COMPOSE         := docker-compose
CONTAINER_VLLM  := qwen35-vllm
CONTAINER_NGINX := qwen35-proxy
API_URL         := http://localhost:8080
API_KEYS_FILE   := api_keys.conf

.DEFAULT_GOAL := help

# ── Colours 
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m
YELLOW := \033[33m

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "  $(BOLD)Qwen3.5-9B · vLLM Server$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ \
		{ printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ── Build & run 
.PHONY: build
build: ## Build the Docker image (no cache)
	$(COMPOSE) build --no-cache

.PHONY: up
up: ## Start all services (build if needed)
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  $(GREEN)✓ Services started.$(RESET)"
	@echo "  API will be ready at $(API_URL)/v1 once the model loads (~2–5 min)."
	@echo "  Run $(CYAN)make logs-vllm$(RESET) to watch progress."

.PHONY: down
down: ## Stop and remove containers (keeps volumes)
	$(COMPOSE) down

.PHONY: restart
restart: ## Restart all services
	$(COMPOSE) restart

.PHONY: restart-nginx
restart-nginx: ## Reload Nginx only (e.g. after editing api_keys.conf)
	$(COMPOSE) restart $(CONTAINER_NGINX)
	@echo "  $(GREEN)✓ Nginx reloaded.$(RESET)"

# ── Logs ───
.PHONY: logs
logs: ## Tail logs from all services
	$(COMPOSE) logs -f

.PHONY: logs-vllm
logs-vllm: ## Tail vLLM logs (model loading progress)
	$(COMPOSE) logs -f $(CONTAINER_VLLM)

.PHONY: logs-nginx
logs-nginx: ## Tail Nginx access / error logs
	$(COMPOSE) logs -f $(CONTAINER_NGINX)

# ── Status ─
.PHONY: status
status: ## Show running containers and GPU usage
	@echo "$(BOLD)── Containers ─────────────────────────────────────$(RESET)"
	@$(COMPOSE) ps
	@echo ""
	@echo "$(BOLD)── GPU Usage ───────────────────────────────────────$(RESET)"
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
		--format=csv,noheader,nounits \
		| awk -F', ' '{printf "  GPU %s %-22s  MEM %5s/%5s MiB  UTIL %3s%%  TEMP %s°C\n", \
			$$1, $$2, $$3, $$4, $$5, $$6}'

.PHONY: health
health: ## Check API health endpoint
	@curl -sf $(API_URL)/health \
		&& echo "  $(GREEN)✓ API is healthy$(RESET)" \
		|| echo "  $(YELLOW)✗ API not yet ready (model may still be loading)$(RESET)"

# ── API key management ────────────────────────────────────────────────────────
.PHONY: add-key
add-key: ## Add a new API key  →  make add-key USER=alice
ifndef USER
	$(error Usage: make add-key USER=<username>)
endif
	@KEY=$$(openssl rand -hex 32); \
	echo '"Bearer sk-$(USER)-'"$$KEY"'"   1;' >> $(API_KEYS_FILE); \
	echo ""; \
	echo "  $(GREEN)✓ Key added for $(USER)$(RESET)"; \
	echo ""; \
	echo "  Share this with $(USER):"; \
	echo "  $(BOLD)  API URL : $(API_URL)/v1$(RESET)"; \
	echo "  $(BOLD)  API Key : sk-$(USER)-$$KEY$(RESET)"; \
	echo ""; \
	echo "  Run $(CYAN)make restart-nginx$(RESET) to apply."

.PHONY: list-keys
list-keys: ## List all users in api_keys.conf
	@echo "$(BOLD)── API Keys ($(API_KEYS_FILE)) ──────────────────────$(RESET)"
	@grep -o 'sk-[^"]*' $(API_KEYS_FILE) | awk '{print "  " $$0}' \
		|| echo "  (no keys found)"

.PHONY: remove-key
remove-key: ## Remove a user's key  →  make remove-key USER=alice
ifndef USER
	$(error Usage: make remove-key USER=<username>)
endif
	@if grep -q 'sk-$(USER)-' $(API_KEYS_FILE); then \
		sed -i '/sk-$(USER)-/d' $(API_KEYS_FILE); \
		echo "  $(GREEN)✓ Key for $(USER) removed.$(RESET)"; \
		echo "  Run $(CYAN)make restart-nginx$(RESET) to apply."; \
	else \
		echo "  $(YELLOW)No key found for user: $(USER)$(RESET)"; \
	fi

# ── Quick test ─
.PHONY: test
test: ## Send a test prompt (requires API_KEY=<key>)
ifndef API_KEY
	$(error Usage: make test API_KEY=<your-key>)
endif
	@curl -s $(API_URL)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $(API_KEY)" \
		-d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Reply with one sentence only: what are you?"}],"max_tokens":128,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print('\n  ' + d['choices'][0]['message']['content'].strip() + '\n')" \
		2>/dev/null || echo "  $(YELLOW)Could not parse response — is the API ready?$(RESET)"

# ── Cleanup 
.PHONY: clean
clean: ## Stop containers and remove images
	$(COMPOSE) down --rmi local

.PHONY: clean-all
clean-all: ## Stop containers, remove images AND the model cache volume
	@echo "$(YELLOW)This will delete the downloaded model (~18 GB). Continue? [y/N]$(RESET)" && \
		read ans && [ "$$ans" = "y" ] && \
		$(COMPOSE) down --rmi local -v && \
		echo "  $(GREEN)✓ Everything removed.$(RESET)" || \
		echo "  Aborted."
