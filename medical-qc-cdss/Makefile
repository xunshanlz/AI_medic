SHELL := /bin/sh
PY ?= python
COMPOSE ?= docker compose

.PHONY: help install test lint run selfcheck demo up up-full down eval build clean

help:
	@echo "install     安装依赖（pip install -e '.[dev]'）"
	@echo "test        运行全部单测（含需要 fastapi 的集成测试）"
	@echo "selfcheck   仅运行零依赖算法层自检（无网络/无 GPU 可用）"
	@echo "run         本地启动服务（mock 模型，SQLite，内存向量库）"
	@echo "up          启动 api + redis（默认 mock 模型）"
	@echo "up-full     启动 api + redis + mysql + milvus + nginx"
	@echo "down        停止并清理容器"
	@echo "eval        用金标准集跑离线评测（需真实模型）"
	@echo "build       构建镜像"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

selfcheck:
	$(PY) scripts/selfcheck.py

run:
	QC_LLM_PROVIDER=mock QC_VECTOR_STORE=memory QC_DATABASE_URL=sqlite+aiosqlite:///./data/qc_cdss.db \
	$(PY) -m uvicorn qc_cdss.api.app:app --host 0.0.0.0 --port 8000 --reload

demo:
	$(PY) scripts/smoke.py --base-url http://127.0.0.1:8000

up:
	$(COMPOSE) up -d --build api redis

up-full:
	$(COMPOSE) --profile mysql --profile milvus --profile gateway up -d --build

down:
	$(COMPOSE) --profile mysql --profile milvus --profile gateway --profile gpu down

eval:
	$(PY) scripts/eval.py --out reports

build:
	$(COMPOSE) build api

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache reports/*.json reports/*.md
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
