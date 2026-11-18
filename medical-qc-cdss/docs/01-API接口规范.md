# API 接口规范

| 项 | 值 |
| --- | --- |
| 版本 | v1 |
| Base URL | `http(s)://<host>:8000` |
| 内容类型 | `application/json; charset=utf-8`（SSE 为 `text/event-stream`） |
| 鉴权 | `Authorization: Bearer <JWT>`（`QC_AUTH_ENABLED=true` 时必填） |
| 追踪 | 请求可带 `X-Trace-Id`；响应头返回同名值，便于日志串联 |
| 时区 | 所有时间为 ISO 8601，带时区偏移 |

## 1. 通用约定

### 1.1 错误响应

```json
{
  "code": "VALIDATION_ERROR",
  "message": "请求参数校验失败",
  "detail": [
    { "loc": ["body", "record"], "msg": "Value error, record 与 raw_text 至少提供一个", "type": "value_error" }
  ],
  "trace_id": "8f2c1d..."
}
```

| HTTP | code | 说明 |
| --- | --- | --- |
| 400 | `BAD_REQUEST` | 业务前置条件不满足（CDSS/ICD 接口病历为空等） |
| 401 | `UNAUTHORIZED` | Token 缺失/签名错误/过期 |
| 404 | `NOT_FOUND` | 任务或问题不存在 |
| 422 | `VALIDATION_ERROR` | Schema 校验失败 |
| 503 | `NOT_READY` | 服务未就绪、ICD 目录未加载 |
| 500 | `INTERNAL_ERROR` | 未捕获异常 |

### 1.2 偏移语义

`evidence.start` / `evidence.end` 为 **左闭右开**，指向"标题：正文"拼接后的整篇文本：

```python
text = "主诉：发热3天\n现病史：3天前受凉后出现发热……"
text[evidence.start:evidence.end] == evidence.quote
```

- 由于送模前做了**等长脱敏**，同一偏移在原始病历与脱敏文本中均成立；
- `evidence.section` 给出所属段落，便于前端定位；
- `pending` / `invalid` 的结论 `start/end` 可能为 0，前端应优先判断 `state`。

## 2. 质控接口

### 2.1 POST `/v1/qc/check`

同步质控，阻塞至完成。适合医生端保存病历后的即时校验。

请求体 `QcRequest`：

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `hospital_id` | string | 否 | `demo-hospital` | 租户标识 |
| `patient_id` | string | 是 | — | 脱敏患者 ID（不要传姓名/身份证） |
| `encounter_id` | string | 是 | — | 就诊号 |
| `encounter_type` | enum | 否 | `inpatient` | `inpatient` / `outpatient` / `emergency` |
| `sex` | enum | 否 | `unknown` | `male` / `female` / `unknown` |
| `age` | int | 否 | null | 0–150 |
| `record` | object | 二选一 | null | 结构化段落，键支持中文标题或英文别名 |
| `raw_text` | string | 二选一 | null | 纯文本病历（≤ 20 万字符） |
| `admit_time` | datetime | 否 | null | 入院时间，用于时限规则 |
| `record_time` | datetime | 否 | null | 记录时间，用于时限规则 |
| `options` | object | 否 | 见下 | 质控选项 |

`options`：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `checklist` | string[] | null | 只跑指定检查项编码 |
| `enable_icd` | bool | true | 是否推荐 ICD |
| `enable_cdss` | bool | true | 预留 |
| `max_findings` | int | 50 | 结果截断 |
| `strict_evidence` | bool | true | 未核验结论是否降级为待复核 |

`record` 支持的键（别名不区分中英文）：

| 中文标题 | 英文别名 |
| --- | --- |
| 主诉 | `chief_complaint` |
| 现病史 | `present_illness` / `history_of_present_illness` |
| 既往史 | `past_history` |
| 个人史 / 婚育史 / 月经史 / 家族史 | `personal_history` / `marital_history` / `menstrual_history` / `family_history` |
| 体格检查 / 专科检查 | `physical_exam` / `specialist_exam` |
| 辅助检查 | `auxiliary_exam` / `lab_results` |
| 初步诊断 | `diagnosis` / `preliminary_diagnosis` |
| 鉴别诊断 | `differential_diagnosis` |
| 诊疗计划 | `treatment_plan` |
| 病程记录 / 手术记录 / 医嘱 / 医师签名 | `progress_note` / `operation_record` / `orders` / `physician_signature` |

响应 `QcResult`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `job_id` | string | 任务 ID |
| `trace_id` | string | 链路 ID |
| `status` | enum | `succeeded` / `degraded` / `failed` |
| `score` | float | 0–100，病历质量分 |
| `findings` | array | 问题列表，按严重级别排序 |
| `icd` | array | Top-3 编码建议 |
| `summary` | string | 中文摘要 |
| `degraded` | bool | 是否发生降级 |
| `degraded_reasons` | string[] | 降级原因（最多 10 条） |
| `model` | string | 实际使用的模型 |
| `prompt_version` | string | 提示词版本 |
| `timings` | array | `{stage, ms}` 阶段耗时 |
| `latency_ms` | float | 端到端耗时 |
| `ttft_ms` | float/null | 流式首 token 时延（同步接口为 null） |
| `created_at` | datetime | 生成时间 |

`Finding`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `finding_id` | string/null | 落库后的主键（同步响应可能为空） |
| `code` | string | 规则/检查项编码 |
| `title` | string | 问题标题 |
| `category` | enum | `completeness` / `consistency` / `logic` / `timeliness` / `safety` / `dosage` / `coding` |
| `severity` | enum | `high` / `medium` / `low` |
| `section` | string | 所属段落 |
| `detail` | string | 问题说明 |
| `suggestion` | string | 修改建议 |
| `evidence` | object/null | 证据片段 |
| `confidence` | float | 0–1 |
| `source` | enum | `rule` / `llm` |
| `needs_review` | bool | 是否需人工复核 |
| `review_status` | enum | `pending` / `accepted` / `rejected` |

`evidence`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `start` / `end` | int | 原文区间（左闭右开） |
| `quote` | string | 原文片段 |
| `state` | enum | `verified` / `pending` / `invalid` |
| `score` | float | 0–100 匹配得分 |
| `strategy` | string | `exact` / `exact_compact` / `fuzzy` / `fuzzy_compact` / `none` |
| `section` | string/null | 所属段落 |

示例：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/qc/check \
  -H 'Content-Type: application/json' \
  -H 'X-Trace-Id: demo-001' \
  -d @data/emr/emr_inpatient_defective.json
```

```json
{
  "job_id": "b2f1c0a93e5d4f6a8b7c9d0e1f2a3b4c",
  "trace_id": "demo-001",
  "status": "succeeded",
  "score": 45.0,
  "findings": [
    {
      "finding_id": null,
      "code": "QC-R006",
      "title": "危急值/高危症状预警",
      "category": "safety",
      "severity": "high",
      "section": "体格检查",
      "detail": "血压达危急值水平",
      "suggestion": "立即评估生命体征，按危急值流程处置并记录",
      "evidence": {
        "start": 212, "end": 226, "quote": "血压 190/110",
        "state": "verified", "score": 100.0, "strategy": "exact"
      },
      "confidence": 1.0,
      "source": "rule",
      "needs_review": false,
      "review_status": "pending"
    }
  ],
  "icd": [
    { "code": "J18.9", "name": "肺炎，未特指", "confidence": 0.82, "rationale": "...", "verified": true }
  ],
  "summary": "病历质量得分 45.0 分，共发现 7 项问题（高危 4 项、中危 3 项）。建议优先处理高危项……",
  "degraded": false,
  "model": "mock-qwen",
  "prompt_version": "2026.11.1",
  "timings": [{ "stage": "prepare", "ms": 1.2 }, { "stage": "rules", "ms": 3.8 }],
  "latency_ms": 218.4,
  "ttft_ms": null,
  "created_at": "2026-11-18T11:20:31+08:00"
}
```

### 2.2 POST `/v1/qc/check/stream`

SSE 流式质控。请求体同 2.1。

响应头：

```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

事件：

| event | data | 说明 |
| --- | --- | --- |
| `job.started` | `{job_id, trace_id}` | 任务开始 |
| `stage.started` | `{stage, ...}` | 阶段开始（`llm_qc` 含 items/tasks） |
| `stage.completed` | `{stage, ...}` | 阶段完成（prepare/rules/llm_qc/icd/finalize） |
| `finding` | `Finding` | 单条问题，到达即可渲染 |
| `icd` | `IcdSuggestion` | 单条编码建议 |
| `summary` | `{score, counts, total, needs_review, degraded, summary}` | 评分摘要 |
| `result` | `QcResult` | 完整结果（与同步一致） |
| `error` | `{message, trace_id}` | 可恢复错误 |
| 注释帧 | `: ping` | 心跳，默认 15s |

示例：

```bash
curl -N -X POST http://127.0.0.1:8000/v1/qc/check/stream \
  -H 'Content-Type: application/json' \
  -d @data/emr/emr_inpatient_defective.json
```

```
event: job.started
data: {"job_id":"...","trace_id":"..."}

event: stage.completed
data: {"stage":"rules","findings":7}

event: finding
data: {"code":"QC-R006","severity":"high",...}

event: result
data: {"job_id":"...","score":45.0,...}
```

前端建议：

- 用 `EventSource` 不便于带 POST body，推荐 `fetch` + `ReadableStream` 自行解析；
- 断线重连由客户端决定；本接口不保证事件重放，重连应重发完整请求；
- 收到 `result` 后可关闭连接。

### 2.3 GET `/v1/qc/jobs/{job_id}`

返回 `JobDetail`：

```json
{
  "result": { "...QcResult..." },
  "findings": [ { "...Finding...", "finding_id": "12" } ],
  "stats": { "score": 45.0, "findings": 7, "degraded": false }
}
```

### 2.4 POST `/v1/qc/findings/{finding_id}/review`

请求：

```json
{ "status": "accepted", "reviewer": "dr.zhang", "comment": "确认存在" }
```

响应：

```json
{ "ok": true, "finding_id": 12, "status": "accepted" }
```

### 2.5 GET `/v1/qc/stats`

参数：`hospital_id`（可选）。

```json
{
  "total_jobs": 128,
  "total_findings": 642,
  "severity": { "high": 180, "medium": 300, "low": 162 },
  "review": { "pending": 400, "accepted": 180, "rejected": 62 },
  "adoption_rate": 0.7438
}
```

## 3. CDSS 接口

### 3.1 POST `/v1/cdss/advice`

请求 `CdssRequest`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `hospital_id` | string | 否 | 租户 |
| `patient_id` / `encounter_id` | string | 否 | 上下文标识 |
| `question` | string | 是 | 临床问题，2–2000 字 |
| `diagnoses` | string[] | 否 | 已知诊断，提升召回 |
| `record_excerpt` | string | 否 | 病历摘要，≤ 2 万字 |
| `sex` / `age` | — | 否 | 人群信息 |
| `top_k` | int | 否 | 检索片段数，默认 5 |

响应 `CdssResponse`：

```json
{
  "request_id": "0d1c...",
  "advice": [
    {
      "advice_type": "exam",
      "content": "建议完善 CURB-65 评分与血氧饱和度评估，判断是否需要住院。",
      "rationale": "指南建议按病情严重程度分层决定治疗场所。",
      "evidence": [
        {
          "doc_id": "community_acquired_pneumonia",
          "title": "社区获得性肺炎诊疗指南（演示版）",
          "chunk_id": "community_acquired_pneumonia#0003",
          "quote": "CURB-65 评分：意识障碍、尿素氮 > 7 mmol/L……",
          "score": 96.0,
          "state": "verified",
          "start": 12,
          "end": 58
        }
      ],
      "confidence": 0.78
    }
  ],
  "disclaimer": "本建议由 AI 基于知识库生成，仅供临床参考，最终决策由医师作出。",
  "degraded": false,
  "latency_ms": 1830.2
}
```

说明：

- `evidence.start/end` 是 **chunk 内**偏移，配合 `chunk_id` 使用；
- `advice_type`：`exam` / `medication` / `follow_up` / `warning` / `education`；
- 无检索结果或模型判定证据不足时，`advice=[]`、`degraded=true`，前端应展示"暂无足够依据"。

### 3.2 POST `/v1/cdss/icd/suggest`

请求 `IcdRequest`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `diagnoses` | string[] | 否 | 诊断文本 |
| `record` / `raw_text` | — | 二选一 | 病历内容 |
| `top_k` | int | 否 | 默认 3，最大 10 |

响应：`IcdSuggestion[]`，按置信度降序。

### 3.3 POST `/v1/cdss/triage`

请求：

```json
{ "symptoms": "突发胸痛伴大汗、呼吸困难", "age": 58, "sex": "male" }
```

响应：

```json
{
  "level": "emergency",
  "matched": ["胸痛", "呼吸困难"],
  "message": "存在急危重症风险信号，请立即前往急诊或拨打 120，勿在线上等待。"
}
```

`level`：`emergency` / `urgent` / `normal`。纯规则，无模型调用。

## 4. 运维接口

| 方法 | 路径 | 响应 |
| --- | --- | --- |
| GET | `/healthz` | `{"status":"ok"}` |
| GET | `/readyz` | `{"status":"ready","knowledge_chunks":N,"icd_entries":M,"startup_errors":[]}` |
| GET | `/v1/meta` | 模型/提示词/存储版本 |
| GET | `/metrics` | Prometheus 文本格式 |

## 5. 版本与兼容

- 路径版本 `/v1`，破坏性变更升 `/v2`；
- 新增字段默认向后兼容；删除/改语义必须升版本；
- `model`、`prompt_version`、`checklist_version` 随响应返回，便于客户端做兼容判断；
- SDK 建议：以 OpenAPI（`/openapi.json`）生成，不要手写。
