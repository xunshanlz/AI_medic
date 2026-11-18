#!/usr/bin/env python
"""构建指南知识库索引（memory / milvus / chroma）。

用法::

    python scripts/build_index.py                 # 按 .env 配置
    python scripts/build_index.py --store milvus  # 写入 Milvus
    python scripts/build_index.py --query "社区获得性肺炎抗菌药物选择"

说明：memory 模式下索引只存在于当前进程，服务启动时会自动重建，因此本脚本
主要用于验证切分质量与调试检索；milvus/chroma 模式才会真正持久化。
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

from qc_cdss.config import get_settings
from qc_cdss.rag.embedding import build_embedder
from qc_cdss.rag.retriever import Retriever, chunk_markdown, load_documents
from qc_cdss.rag.store import build_vector_store


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.store:
        settings.vector_store = args.store
    if args.embedding:
        settings.embedding_provider = args.embedding
    if args.data_dir:
        settings.data_dir = Path(args.data_dir)

    documents = load_documents(settings.guidelines_dir)
    if not documents:
        print(f"[ERROR] 未找到知识库文件：{settings.guidelines_dir}")
        return 1

    total_chunks = 0
    for doc_id, title, text in documents:
        chunks = chunk_markdown(doc_id, title, text)
        total_chunks += len(chunks)
        print(f"  {doc_id:<40} {len(chunks):>3} chunks  《{title}》")

    embedder = build_embedder(settings)
    store = build_vector_store(settings, embedder.dim)
    retriever = await Retriever.build(
        settings.guidelines_dir, embedder, store, rrf_k=settings.rrf_k
    )
    print(
        f"\n[done] docs={len(documents)} chunks={total_chunks} "
        f"dense_dim={embedder.dim} store={settings.vector_store}"
    )

    if args.query:
        print(f"\n[query] {args.query}")
        for index, hit in enumerate(
            await retriever.retrieve(args.query, top_k=args.top_k), start=1
        ):
            text = hit.chunk.text.replace("\n", " ")[:90]
            print(
                f"  {index}. score={hit.score:.5f} dense#{hit.dense_rank} lex#{hit.lexical_rank} "
                f"{hit.chunk.doc_id} :: {text}"
            )

    await retriever.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建指南知识库索引")
    parser.add_argument("--store", choices=["memory", "milvus", "chroma"], default=None)
    parser.add_argument("--embedding", choices=["hash", "openai"], default=None)
    parser.add_argument("--data-dir", default=None, help="数据目录，默认 ./data")
    parser.add_argument("--query", default=None, help="构建后执行一次检索验证")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
