COMPOSE := docker-compose
API_URL := http://localhost:8088
KEYS    := api_keys.conf
MODEL   := qwen2.5-14b-instruct

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Qwen2.5-14B-Instruct vLLM Server (CUDA 12.1 custom image)"
	@echo ""
	@echo "  build           Build the custom CUDA 12.1 image"
	@echo "  up              Start all services"
	@echo "  down            Stop all services"
	@echo "  logs            Tail all logs"
	@echo "  logs-vllm       Tail vLLM logs only"
	@echo "  logs-nginx      Tail nginx logs only"
	@echo "  status          Containers + GPU usage"
	@echo "  health          Check API health"
	@echo "  add-key USER=x  Add API key for user"
	@echo "  list-keys       List all API keys"
	@echo "  remove-key USER=x  Remove user's key"
	@echo "  test API_KEY=x  Send a test prompt"
	@echo "  clean           Stop + remove local images"

.PHONY: build
build:
	$(COMPOSE) build --no-cache

.PHONY: up
up:
	$(COMPOSE) up -d
	@echo "Services started. Watch progress: make logs-vllm"

.PHONY: down
	down:
	$(COMPOSE) down

.PHONY: logs
logs:
	$(COMPOSE) logs -f

.PHONY: logs-vllm
logs-vllm:
	$(COMPOSE) logs -f qwen25-vllm

.PHONY: logs-nginx
logs-nginx:
	$(COMPOSE) logs -f qwen25-proxy

.PHONY: status
status:
	@$(COMPOSE) ps
	@echo ""
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
		--format=csv,noheader,nounits \
		| awk -F', ' '{printf "GPU %s  %-22s  %5s/%5s MiB  %s%% util\n",$$1,$$2,$$3,$$4,$$5}'

.PHONY: health
health:
	@curl -sf $(API_URL)/health && echo "API is healthy" || echo "API not ready yet"

.PHONY: add-key
add-key:
ifndef USER
	$(error Usage: make add-key USER=alice)
endif
	@KEY=$$(openssl rand -hex 32); \
	echo '"Bearer sk-$(USER)-'"$$KEY"'"   1;' >> $(KEYS); \
	echo "Key for $(USER): sk-$(USER)-$$KEY"; \
	echo "Reload nginx: $(COMPOSE) restart qwen25-proxy"

.PHONY: list-keys
list-keys:
	@grep -o 'sk-[^"]*' $(KEYS) || echo "(no keys)"

.PHONY: remove-key
remove-key:
ifndef USER
	$(error Usage: make remove-key USER=alice)
endif
	@sed -i '/sk-$(USER)-/d' $(KEYS) && echo "Removed $(USER). Reload: $(COMPOSE) restart qwen25-proxy"

.PHONY: test
test:
ifndef API_KEY
	$(error Usage: make test API_KEY=sk-...)
endif
	@curl -s $(API_URL)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $(API_KEY)" \
		-d '{"model":"$(MODEL)","messages":[{"role":"user","content":"What are you?"}],"max_tokens":64}' \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())"

.PHONY: clean
clean:
	$(COMPOSE) down --rmi local
