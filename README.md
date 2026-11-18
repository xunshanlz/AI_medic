# AI 病历质控与 CDSS 智能决策应用

> 按《项目文档》第 10 章的实施顺序，分解成可一步步验收的教学版工程。
> 每个阶段都有可运行产物、可复现命令和一组「学会了吗」自测题。

## 一、当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 01 | 服务骨架：配置、日志、应用工厂、健康探针、单测 | ✅ 完成 |
| 02 | 模型接入层：provider 适配器 + OpenAI 兼容客户端 + 非流式接口 | ⬜ 待开始 |
| 03 | SSE 流式：业务事件流、心跳、流内错误 | ⬜ |
| 04 | 结构化契约：Pydantic 模型 + 三层容错解析 | ⬜ |
| 05 | 确定性溯源校验：精确 + rapidfuzz + 三态 + Offsets | ⬜ |
| 06 | 长病历分片与 Map-Reduce 聚合 | ⬜ |
| 07 | 工程横切：request_id、限流、熔断、分级超时、JSON 日志 | ⬜ |
| 08 | 存储与审计：SQLite(开发) → MySQL、审计只追加 | ⬜ |
| 09 | 评测门禁：金标集、断言脚本、指标口径 | ⬜ |
| 10 | 交付编排：Docker Compose + Nginx（可选） | ⬜ |

## 二、环境

- Python：`.venv`（本项目已建，3.12.10）
- 包管理：pip（已装 fastapi / uvicorn / pydantic-settings / pytest / httpx）
- 推理层：开发期用云端 OpenAI 兼容接口（`https://api.deepseek.com/v1`）；
  院内交付时换成 vLLM，业务代码不变。
- 未安装：Docker、Git、MySQL、Redis（阶段 08 / 10 再处理）

## 三、跑起来

```powershell
# 1) 安装依赖（首次）
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 启动开发服务
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# 3) 验证
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
# 浏览器打开 http://127.0.0.1:8000/docs 看自动生成的接口文档

# 4) 跑测试
.\.venv\Scripts\python.exe -m pytest
```

## 四、目录结构

```
app/
  main.py           应用工厂 create_app() + uvicorn 入口
  core/
    config.py       配置层（pydantic-settings，CDSS_ 前缀）
    logging.py      日志初始化（阶段 07 升级为结构化 JSON）
  api/
    health.py       /healthz 存活探针、/readyz 就绪探针
tests/
  test_health.py    阶段 01 验收测试
docs/
  学习进度.md       每阶段的学习记录与自测题
```

## 五、一条重要的口径

本项目是**医院私有化辅助工具**：结论必须可追溯、可审核、可拦截。
所以后面每个阶段都会围绕同一句话加一块能力：

> 模型负责「理解语义」，代码负责「校验与留痕」。
> 模型说不清的，代码不允许它混过去。

开发期调用云端接口只允许使用**虚构/脱敏**的病历样例，禁止使用真实患者数据。
