FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tzdata \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖描述，利用 Docker layer 缓存；mysql extra 提供 asyncmy 驱动
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[mysql,migration]"

COPY data ./data
COPY scripts ./scripts

# 非 root 运行
RUN useradd -m -u 10001 qc && mkdir -p /app/data && chown -R qc:qc /app
USER qc

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "qc_cdss.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
