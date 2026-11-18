# QC-CDSS · 病历内涵质控与临床决策支持平台

面向三甲医院私有化场景的大模型应用：**规则前置 + vLLM 推理 + 确定性证据溯源 + ICD 编码推荐 + 指南证据驱动的 CDSS**。

> 演示数据仅用于工程验证，不构成诊疗建议。所有指南文件均为公开要点整理，不可用于真实临床决策。

## 解决什么问题

| 痛点 | 现状 | 本项目的做法 |
| --- | --- | --- |
| 病历内涵质量靠人工抽查 | 质控科按 5% 抽检，漏检率高、反馈滞后 | 规则 + LLM 双通道全量质控，问题带原文定位 |
| 大模型结论不可追溯 | 模型给出"诊断依据不足"，医生无法核对 | 引文逐字校验，返回字符 offset，三态（已核验/待核验/无效） |
| 私有化算力有限 | 单卡 32B，长病历并发即超时 | 长短请求隔离、分片并行、分级超时、结果缓存、降级不阻断 |
| CDSS 建议无依据 | 通用问答容易"编指南" | 双路召回 + RRF + 证据绑定，证据不足直接拒答 |
| 编码靠人工 | ICD 编码耗时长、口径不一 | 候选召回 + LLM 重排 + 编码白名单校验 |

## 架构

```mermaid
flowchart LR
    subgraph Client[医生工作站 / 质控平台]
        UI[EMR 前端]
    end

    subgraph Gateway[Nginx]
        NG[反向代理<br/>SSE 不缓冲]
    end

    subgraph API[FastAPI 服务]
        QC[质控接口<br/>同步 / SSE]
        CDSS[CDSS 接口]
        OPS[探针 / 指标]
    end

    subgraph Pipeline[LangGraph 质控流水线]
        P1[prepare<br/>分段 + 等长脱敏]
        P2[rules<br/>9 条确定性规则]
        P3[llm_qc<br/>6 检查项 × 分片并发]
        P4[icd<br/>召回 + 重排 + 校验]
        P5[finalize<br/>去重 / 评分 / 摘要]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph Infra[基础设施]
        VLLM[vLLM<br/>OpenAI 兼容]
        MILVUS[(Milvus / Chroma)]
        REDIS[(Redis)]
        MYSQL[(MySQL / SQLite)]
    end

    UI --> NG --> QC
    QC --> Pipeline
    CDSS --> Pipeline
    P3 --> VLLM
    CDSS --> MILVUS
    Pipeline --> REDIS
    Pipeline --> MYSQL
```

## 核心能力

- **三层结构化输出治理**：受约束解码 → 纠错重试 → 字段级抢救 → 降级标记，模型输出异常不阻断主流程。
- **确定性证据溯源**：归一化 + 偏移映射 + 精确/近似双通道匹配，阈值 88/70，返回原文 `start/end` 供高亮。
- **等长 PII 脱敏**：送模前把姓名/身份证/手机号替换为等长占位符，脱敏文本与原文偏移一一对应。
- **并发与降级**：短请求预留槽位、长病历分片并行、分支级超时、Redis 结果缓存、主备模型自动切换。
- **可审计**：审计日志哈希链、模型/提示词版本随结果落库、`/metrics` 暴露质量与性能指标。
- **可评测**：金标准集 + 召回率/证据通过率/ICD Top-k/时延分位数，一条命令产出报告。

## 快速开始

### 方式一：Docker（推荐）

```bash
cp .env.example .env          # 默认 mock 模型，无需 GPU
docker compose up -d --build api redis

curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/readyz
```

跑一次全链路冒烟（质控 / SSE / CDSS / 分诊）：

```bash
python -m pip install httpx
python scripts/smoke.py --base-url http://127.0.0.1:8000
```

### 方式二：本地 Python

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 零依赖自检（无需任何三方包）
python scripts/selfcheck.py

# 启动（mock 模型 + SQLite + 内存向量库）
QC_LLM_PROVIDER=mock uvicorn qc_cdss.api.app:app --port 8000
```

### 方式三：私有化 vLLM + Milvus

```bash
# 1) 起推理与向量库（需要 NVIDIA GPU + nvidia-container-toolkit）
docker compose --profile gpu --profile milvus up -d vllm milvus etcd minio

# 2) 切配置
# .env
QC_LLM_PROVIDER=vllm
QC_LLM_BASE_URL=http://vllm:8000/v1
QC_LLM_MODEL=Qwen2.5-32B-Instruct-AWQ
QC_LLM_JSON_MODE=guided
QC_VECTOR_STORE=milvus
QC_EMBEDDING_PROVIDER=openai
QC_EMBEDDING_BASE_URL=http://vllm:8000/v1
QC_EMBEDDING_MODEL=bge-m3
QC_DATABASE_URL=mysql+asyncmy://qc:qc_pass@mysql:3306/qc_cdss?charset=utf8mb4

# 3) 起服务并建索引
docker compose --profile mysql --profile gateway up -d api mysql nginx
docker compose exec api python scripts/build_index.py --store milvus
```

## 接口示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/qc/check \
  -H 'Content-Type: application/json' \
  -d @data/emr/emr_inpatient_defective.json | jq
```

响应（节选）：

```json
{
  "job_id": "9f3c...",
  "status": "succeeded",
  "score": 45.0,
  "findings": [
    {
      "code": "QC-R006",
      "title": "危急值/高危症状预警",
      "severity": "high",
      "section": "体格检查",
      "detail": "血压达危急值水平",
      "evidence": {
        "start": 212, "end": 226, "quote": "血压 190/110",
        "state": "verified", "score": 100.0
      }
    }
  ],
  "icd": [{ "code": "J18.9", "name": "肺炎，未特指", "confidence": 0.82 }],
  "degraded": false,
  "model": "mock-qwen",
  "prompt_version": "2026.11.1"
}
```

流式（SSE）：

```bash
curl -N -X POST http://127.0.0.1:8000/v1/qc/check/stream \
  -H 'Content-Type: application/json' \
  -d @data/emr/emr_inpatient_defective.json
```

事件序列：`job.started → stage.completed(prepare/rules) → finding* → stage.completed(llm_qc) → icd* → summary → result`。

## 目录结构

```
medical-qc-cdss/
├── src/qc_cdss/
│   ├── domain/            # 零依赖业务核心：归一化、证据校验、规则、评分
│   ├── llm/               # OpenAI 兼容网关、提示词、结构化治理、Mock
│   ├── rag/               # 切分、嵌入、向量库、BM25、RRF
│   ├── pipeline/          # LangGraph 流水线、并发调度、ICD、聚合
│   ├── api/               # FastAPI 路由、SSE、鉴权、容器
│   ├── db/                # SQLAlchemy 模型与仓储
│   ├── config.py          # 环境变量配置
│   ├── schemas.py         # 对外契约
│   ├── privacy.py         # 等长脱敏
│   ├── audit.py           # 审计哈希链
│   └── metrics.py         # Prometheus 指标
├── scripts/               # selfcheck / build_index / eval / smoke
├── data/                  # 指南、ICD 目录、样例病历、金标准集
├── deploy/                # Nginx
├── docs/                  # 开发文档（主文档 + API/DB/评测/部署）
├── tests/                 # 67 个单测 + 接口集成测试
├── docker-compose.yml
└── pyproject.toml
```

## 测试与评测

```bash
python -m unittest discover -s tests -t .   # 无需三方依赖，67 项
python scripts/selfcheck.py                 # 12 项核心算法自检
pytest -q                                   # 含 FastAPI 集成测试（容器内）
python scripts/eval.py --out reports        # 金标准集评测（真实模型才有意义）
```

当前基线：**67 项单测通过、12 项自检通过**；接口集成测试需安装运行依赖。

## 文档

| 文档 | 内容 |
| --- | --- |
| [项目开发文档](docs/00-项目开发文档.md) | 背景、架构、技术选型、核心设计、里程碑、风险、面试口径 |
| [API 接口规范](docs/01-API接口规范.md) | 端点、字段、状态码、SSE 协议 |
| [数据库设计](docs/02-数据库设计.md) | ER、表结构、索引、审计链、保留策略 |
| [评测方案](docs/03-评测方案与实施.md) | 指标定义、金标准集、评测流程、门槛 |
| [部署与运维](docs/04-部署与运维手册.md) | Compose/K8s、GPU 调优、监控、备份、排障 |

## 已知边界

- `data/` 中的指南、ICD 目录、病历均为**演示数据**，上线需替换为医院授权知识库与国临版编码表。
- `QC_LLM_PROVIDER=mock` 只用于链路验证，**不产生任何效果指标**。
- 评分仅用于病历质量趋势分析，不用于医生个人考核（产品红线，见主文档）。
