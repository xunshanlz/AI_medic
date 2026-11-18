"""AI 病历质控与 CDSS 智能决策应用 —— 应用包。

分层约定（与项目文档 4.1 节一致）：
    app/api        接入层：只做协议、参数校验、路由，不放业务判定
    app/core       横切能力：配置、日志、错误
    app/llm        推理层适配：provider 路由与 OpenAI 兼容客户端
    app/contracts  结构化契约：Pydantic 模型
    app/verify     确定性溯源校验层
    app/orchestrator  长病历分片与 Map-Reduce 调度
    app/storage    数据访问与审计
"""

__version__ = "0.1.0"
