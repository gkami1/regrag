#!/usr/bin/env python3
"""Parse PDFs into data/processed/<doc_id>/ (manifest + blocks.jsonl).

Skips documents whose processed output is already up to date (same PDF bytes,
parser version and config), so re-running over a whole folder is cheap.

Examples:
    python scripts/ingest.py data/raw/R107r9e.pdf --doc-id unece-r107-rev9
    python scripts/ingest.py data/raw/*.pdf
    python scripts/ingest.py data/raw/R107r9e.pdf --force
"""

import argparse
import logging
import sys
from pathlib import Path

from regrag.ingestion.pdf_parser import ParserConfig, default_doc_id, parse_pdf
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR, save_document, staleness_reason

logger = logging.getLogger("ingest")


def ingest_one(pdf: Path, doc_id: str, out_root: Path, config: ParserConfig, force: bool) -> None:
    doc_dir = out_root / doc_id
    reason = "--force" if force else staleness_reason(doc_dir, pdf, config)
    if reason is None:
        logger.info("%s: up to date, skipping (%s)", doc_id, doc_dir)
        return
    logger.info("%s: parsing %s (%s)", doc_id, pdf, reason)
    doc = parse_pdf(pdf, config, doc_id=doc_id)
    save_document(doc, pdf, config, out_root)
    s = doc.structure
    logger.info(
        "%s: %d blocks, %d scopes, %d numbered, %d backward jumps",
        doc_id,
        len(doc.blocks),
        len(s.scopes),
        s.numbered_blocks,
        len(s.backward_jumps),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to ingest")
    parser.add_argument("--doc-id", help="document id (only with a single PDF)")
    parser.add_argument("--out", type=Path, default=DEFAULT_PROCESSED_DIR, help="output root")
    parser.add_argument("--force", action="store_true", help="re-parse even if up to date")
    args = parser.parse_args(argv)

    if args.doc_id and len(args.pdfs) > 1:
        parser.error("--doc-id can only be used with a single PDF")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = ParserConfig()

    failed = []
    for pdf in args.pdfs:
        doc_id = args.doc_id or default_doc_id(pdf)
        try:
            ingest_one(pdf, doc_id, args.out, config, args.force)
        except Exception:
            # One bad PDF must not stop a batch; report it and carry on.
            logger.exception("%s: failed", pdf)
            failed.append(pdf)

    if failed:
        logger.error("%d of %d documents failed: %s", len(failed), len(args.pdfs), failed)
        return 1  # non-zero exit code so scripts/CI notice
    return 0


if __name__ == "__main__":
    sys.exit(main())
