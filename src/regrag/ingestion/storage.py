"""Persist parsed documents to data/processed/ and load them back.

Layout, one directory per document:

    data/processed/<doc_id>/
        blocks.jsonl     one ParsedBlock per line, ALL roles (furniture is tagged, not dropped)
        document.json    manifest: versions, source fingerprint, config, pages, reports

This is the contract between the parsing stage and everything downstream
(chunking, indexing), so it is versioned twice:

- SCHEMA_VERSION: the *shape* of the files. Bump on any incompatible field change.
  Loading a file with another schema version fails with a clear error.
- PARSER_VERSION (pdf_parser.py): the *logic* that produced the content.

Caching rule: a document is re-parsed only when something that determines its
output changed: the PDF bytes (sha256), the parser version, the parser config,
or the schema. See `staleness_reason`.

Crash safety: every file is written to a temp file and atomically renamed into
place, and the manifest is removed first and written last. So "manifest exists"
implies "blocks.jsonl is complete", and the block count in the manifest is
checked on load as a second line of defence.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from regrag.ingestion.models import (
    DocId,
    LayoutReport,
    Model,
    PageInfo,
    ParsedBlock,
    ParsedDocument,
    StructureReport,
    StyleReport,
)
from regrag.ingestion.pdf_parser import PARSER_VERSION, ParserConfig
from regrag.io_utils import file_sha256, write_lines_atomic

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_FILE = "document.json"
BLOCKS_FILE = "blocks.jsonl"
DEFAULT_PROCESSED_DIR = Path("data/processed")


class StorageError(Exception):
    """A processed document is missing, incomplete or invalid."""


class SchemaVersionError(StorageError):
    """A processed document was written with an incompatible schema version."""


class SourceInfo(Model):
    path: str
    sha256: str
    size_bytes: int


class Manifest(Model):
    schema_version: int
    parser_version: str
    doc_id: DocId
    created_at: datetime
    source: SourceInfo
    config: ParserConfig
    block_count: int
    pages: list[PageInfo]
    layout: LayoutReport | None = None
    styles: StyleReport | None = None
    structure: StructureReport | None = None


def save_document(
    doc: ParsedDocument,
    source_pdf: Path,
    config: ParserConfig,
    out_root: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Write `doc` to <out_root>/<doc_id>/ and return that directory."""
    out_dir = out_root / doc.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_FILE

    # Invalidate first: if we crash below, there is no manifest describing
    # half-written blocks.
    manifest_path.unlink(missing_ok=True)

    # Pydantic writes non-ASCII as-is (no \u escapes), so Cyrillic GOST text
    # stays readable in the file.
    write_lines_atomic(out_dir / BLOCKS_FILE, [b.model_dump_json() for b in doc.blocks])

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        parser_version=PARSER_VERSION,
        doc_id=doc.doc_id,
        created_at=datetime.now(UTC),
        source=SourceInfo(
            path=source_pdf.as_posix(),
            sha256=file_sha256(source_pdf),
            size_bytes=source_pdf.stat().st_size,
        ),
        config=config,
        block_count=len(doc.blocks),
        pages=doc.pages,
        layout=doc.layout,
        styles=doc.styles,
        structure=doc.structure,
    )
    write_lines_atomic(manifest_path, [manifest.model_dump_json(indent=2)])
    logger.info("Saved %s: %d blocks -> %s", doc.doc_id, len(doc.blocks), out_dir)
    return out_dir


def load_manifest(doc_dir: Path) -> Manifest:
    path = doc_dir / MANIFEST_FILE
    if not path.exists():
        raise StorageError(f"No manifest at {path} (never parsed, or an interrupted write)")
    raw = path.read_text(encoding="utf-8")
    # Check the version before full validation: an old file would also fail
    # validation, but "schema 1 != 2" is a far more useful message than a list
    # of missing fields.
    try:
        version = json.loads(raw).get("schema_version")
    except json.JSONDecodeError as e:
        raise StorageError(f"{path}: not valid JSON: {e}") from e
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path}: schema version {version!r}, this code reads {SCHEMA_VERSION}. "
            "Re-run ingestion with --force to rebuild it."
        )
    try:
        return Manifest.model_validate_json(raw)
    except ValidationError as e:
        raise StorageError(f"{path}: invalid manifest:\n{e}") from e


def load_document(doc_dir: Path) -> ParsedDocument:
    """Load and validate a processed document. Raises StorageError on any problem."""
    manifest = load_manifest(doc_dir)
    path = doc_dir / BLOCKS_FILE
    if not path.exists():
        raise StorageError(f"Missing {path}")

    blocks: list[ParsedBlock] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                blocks.append(ParsedBlock.model_validate_json(line))
            except ValidationError as e:
                # Point at the exact line, so the file can be inspected directly.
                raise StorageError(f"{path}:{lineno}: invalid block:\n{e}") from e

    if len(blocks) != manifest.block_count:
        raise StorageError(
            f"{path}: {len(blocks)} blocks, manifest says {manifest.block_count} "
            "(truncated or mismatched file)"
        )
    return ParsedDocument(
        doc_id=manifest.doc_id,
        source_path=manifest.source.path,
        pages=manifest.pages,
        blocks=blocks,
        layout=manifest.layout,
        styles=manifest.styles,
        structure=manifest.structure,
    )


def staleness_reason(doc_dir: Path, source_pdf: Path, config: ParserConfig) -> str | None:
    """Why the processed document must be rebuilt, or None if it is up to date."""
    try:
        manifest = load_manifest(doc_dir)
    except StorageError as e:
        return str(e).splitlines()[0]
    if manifest.parser_version != PARSER_VERSION:
        return f"parser version {manifest.parser_version} -> {PARSER_VERSION}"
    # Compare JSON forms: compiled regexes and floats compare reliably that way.
    if manifest.config.model_dump(mode="json") != config.model_dump(mode="json"):
        return "parser config changed"
    if manifest.source.sha256 != file_sha256(source_pdf):
        return "source PDF changed (sha256 differs)"
    return None
