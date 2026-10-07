#!/usr/bin/env python3
"""Embed chunks and index them into Qdrant (dense + sparse), incrementally.

Only new or changed chunks are embedded; points of removed chunks are deleted.
The embedding model is loaded only if there is something to embed.

Examples:
    python scripts/index.py                          # every chunked document
    python scripts/index.py unece-r107-rev9
    python scripts/index.py --recreate               # drop and rebuild the collection

Qdrant location: --qdrant-url, else $QDRANT_URL, else http://localhost:6333.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from regrag.chunking import load_chunks
from regrag.chunking.storage import CHUNKS_MANIFEST_FILE
from regrag.embedding import EmbedderConfig, LazyEmbedder
from regrag.indexing import ChunkIndex, IndexConfig
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR

logger = logging.getLogger("index")


def _load_bge_m3(config: EmbedderConfig):
    from regrag.embedding.bge_m3 import BGEM3Embedder  # torch import only when needed

    return BGEM3Embedder(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("doc_ids", nargs="*", help="documents to index (default: all chunked)")
    parser.add_argument("--root", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    parser.add_argument("--collection", default=IndexConfig().collection)
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP call is noise

    doc_ids = args.doc_ids or sorted(
        p.name for p in args.root.iterdir() if (p / CHUNKS_MANIFEST_FILE).exists()
    )
    if not doc_ids:
        logger.error("No chunked documents in %s; run scripts/chunk.py first", args.root)
        return 1

    emb_config = EmbedderConfig()
    embedder = LazyEmbedder(emb_config, lambda: _load_bge_m3(emb_config))
    client = QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    index = ChunkIndex(client, embedder, IndexConfig(collection=args.collection))

    if args.recreate and client.collection_exists(index.cfg.collection):
        client.delete_collection(index.cfg.collection)
        logger.warning("Dropped collection %s", index.cfg.collection)

    failed = []
    for doc_id in doc_ids:
        try:
            report = index.index_document(doc_id, load_chunks(args.root / doc_id))
            logger.info("%s", report.model_dump())
        except Exception:
            logger.exception("%s: failed", doc_id)
            failed.append(doc_id)

    logger.info("Collection %s: %d points total", index.cfg.collection, index.count())
    if failed:
        logger.error("%d of %d documents failed: %s", len(failed), len(doc_ids), failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
