"""High-level document loader.

Resolution order:
1. If file already exists in data/raw/ — use it.
2. Otherwise try to download it.
3. If download fails — raise a clear error telling the user to drop the PDF manually.
"""

import logging
from pathlib import Path

from regrag.ingestion.downloader import DOCUMENT_URLS, download_document

logger = logging.getLogger(__name__)


def resolve_document(
    doc_key: str,
    raw_dir: Path = Path("data/raw"),
) -> Path:
    """Return path to a locally available PDF, downloading if needed."""
    if doc_key not in DOCUMENT_URLS:
        raise ValueError(f"Unknown document: {doc_key}. Available: {list(DOCUMENT_URLS)}")

    expected = raw_dir / DOCUMENT_URLS[doc_key]["filename"]

    if expected.exists():
        logger.info("Using existing file: %s", expected)
        return expected

    try:
        return download_document(doc_key, output_dir=raw_dir)
    except Exception as e:
        raise RuntimeError(
            f"Could not obtain {doc_key!r} automatically: {e}\n"
            f"→ Скачай PDF вручную и положи в: {expected}"
        ) from e