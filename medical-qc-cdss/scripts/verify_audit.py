#!/usr/bin/env python
"""校验审计日志哈希链是否连续（等保审计要求）。

用法::

    python scripts/verify_audit.py
    python scripts/verify_audit.py --database-url "mysql+asyncmy://qc:***@mysql:3306/qc_cdss"

说明：本脚本校验 ``prev_hash`` 与上一条 ``entry_hash`` 的链接关系。
更强的防篡改需要把每日链尾哈希锚定到外部 WORM 存储（见部署文档）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from sqlalchemy import select

from qc_cdss.config import get_settings
from qc_cdss.db.models import AuditLog
from qc_cdss.db.session import build_engine, build_session_factory


async def run(database_url: str | None, limit: int) -> int:
    settings = get_settings()
    if database_url:
        settings.database_url = database_url

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        async with session_factory() as session:
            rows = list(
                await session.scalars(
                    select(AuditLog).order_by(AuditLog.id.asc()).limit(limit)
                )
            )
    finally:
        await engine.dispose()

    if not rows:
        print("[ok] 审计表为空（首次部署属正常）")
        return 0

    previous = "GENESIS"
    for row in rows:
        if row.prev_hash != previous:
            print(
                f"[FAIL] 哈希链断点 id={row.id}："
                f"期望 prev={previous[:16]}…，实际 {row.prev_hash[:16]}…"
            )
            return 1
        if len(row.entry_hash) != 64:
            print(f"[FAIL] id={row.id} entry_hash 长度异常：{len(row.entry_hash)}")
            return 1
        previous = row.entry_hash

    print(f"[ok] 共校验 {len(rows)} 条，哈希链连续；链尾 entry_hash={rows[-1].entry_hash[:16]}…")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="校验审计哈希链")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()
    return asyncio.run(run(args.database_url, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
