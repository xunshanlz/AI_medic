"""日志初始化。

阶段 01 用「人可读 + 时间戳」的简单格式，方便新手看。
阶段 07 会升级成结构化 JSON 日志（文档 5.7 / 8.5 要求），并加 request_id 串联。
"""

import logging


def configure_logging(level: str = "INFO") -> None:
    """把根 logger 的 handler 重设一遍，避免重复输出。

    force=True 的作用：uvicorn 启动时会先配置一次日志，
    这里强制覆盖，保证我们自己 app 的日志格式统一。
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
