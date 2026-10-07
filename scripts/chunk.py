#!/usr/bin/env python3
"""Chunk processed documents: data/processed/<doc_id>/blocks.jsonl -> chunks.jsonl.

Skips documents whose chunks are up to date (same blocks, chunker version, config).

Examples:
    python scripts/chunk.py                      # every document in data/processed
    python scripts/chunk.py unece-r107-rev9
    python scripts/chunk.py unece-r107-rev9 --force
"""

import argparse
import logging
import sys
from pathlib import Path

from regrag.chunking import (
    Chunker,
    ChunkerConfig,
    HFTokenCounter,
    chunks_staleness_reason,
    save_chunks,
)
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR, MANIFEST_FILE, load_document

logger = logging.getLogger("chunk")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("doc_ids", nargs="*", help="documents to chunk (default: all)")
    parser.add_argument("--root", type=Path, default=DEFAULT_PROCESSED_DIR, help="processed root")
    parser.add_argument("--force", action="store_true", help="re-chunk even if up to date")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    doc_ids = args.doc_ids or sorted(
        p.name for p in args.root.iterdir() if (p / MANIFEST_FILE).exists()
    )
    if not doc_ids:
        logger.error("No processed documents in %s; run scripts/ingest.py first", args.root)
        return 1

    config = ChunkerConfig()
    chunker: Chunker | None = None  # the tokenizer is loaded only if there is work to do

    failed = []
    for doc_id in doc_ids:
        doc_dir = args.root / doc_id
        try:
            reason = "--force" if args.force else chunks_staleness_reason(doc_dir, config)
            if reason is None:
                logger.info("%s: up to date, skipping", doc_id)
                continue
            logger.info("%s: chunking (%s)", doc_id, reason)
            chunker = chunker or Chunker(config, HFTokenCounter(config.tokenizer))
            chunks, report = chunker.chunk_document(load_document(doc_dir))
            save_chunks(doc_dir, doc_id, chunks, config, report)
            logger.info("%s: %s", doc_id, report.model_dump())
        except Exception:
            logger.exception("%s: failed", doc_id)
            failed.append(doc_id)

    if failed:
        logger.error("%d of %d documents failed: %s", len(failed), len(doc_ids), failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
