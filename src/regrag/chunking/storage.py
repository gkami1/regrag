"""Persist chunks next to the parsed document they were built from.

    data/processed/<doc_id>/
        blocks.jsonl, document.json    (ingestion stage)
        chunks.jsonl                   one Chunk per line
        chunks.json                    manifest: versions, input fingerprint, config, report

Same contract as ingestion: schema version checked on load, atomic writes,
manifest written last. The cache key is the sha256 of blocks.jsonl, i.e. the
chunker's actual input: if re-ingestion produces identical blocks, existing
chunks stay valid; if anything in the blocks changed, they are rebuilt.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from regrag.chunking.models import Chunk, ChunkerConfig, ChunkingReport
from regrag.ingestion.models import DocId, Model
from regrag.ingestion.storage import BLOCKS_FILE, SchemaVersionError, StorageError
from regrag.io_utils import file_sha256, write_lines_atomic

logger = logging.getLogger(__name__)

CHUNK_SCHEMA_VERSION = 1
# Version of the chunking *logic*. Bump when a change may alter the chunks
# produced from the same blocks and config.
CHUNKER_VERSION = "0.1.0"
CHUNKS_FILE = "chunks.jsonl"
CHUNKS_MANIFEST_FILE = "chunks.json"


class ChunksManifest(Model):
    schema_version: int
    chunker_version: str
    doc_id: DocId
    created_at: datetime
    blocks_sha256: str  # fingerprint of the input
    config: ChunkerConfig
    chunk_count: int
    report: ChunkingReport


def save_chunks(
    doc_dir: Path,
    doc_id: str,
    chunks: list[Chunk],
    config: ChunkerConfig,
    report: ChunkingReport,
) -> None:
    manifest_path = doc_dir / CHUNKS_MANIFEST_FILE
    manifest_path.unlink(missing_ok=True)  # invalidate first, write last
    write_lines_atomic(doc_dir / CHUNKS_FILE, (c.model_dump_json() for c in chunks))
    manifest = ChunksManifest(
        schema_version=CHUNK_SCHEMA_VERSION,
        chunker_version=CHUNKER_VERSION,
        doc_id=doc_id,
        created_at=datetime.now(UTC),
        blocks_sha256=file_sha256(doc_dir / BLOCKS_FILE),
        config=config,
        chunk_count=len(chunks),
        report=report,
    )
    write_lines_atomic(manifest_path, [manifest.model_dump_json(indent=2)])
    logger.info("Saved %d chunks -> %s", len(chunks), doc_dir / CHUNKS_FILE)


def load_chunks_manifest(doc_dir: Path) -> ChunksManifest:
    path = doc_dir / CHUNKS_MANIFEST_FILE
    if not path.exists():
        raise StorageError(f"No chunks manifest at {path} (never chunked, or an interrupted write)")
    raw = path.read_text(encoding="utf-8")
    try:
        version = json.loads(raw).get("schema_version")
    except json.JSONDecodeError as e:
        raise StorageError(f"{path}: not valid JSON: {e}") from e
    if version != CHUNK_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path}: schema version {version!r}, this code reads {CHUNK_SCHEMA_VERSION}. "
            "Re-run chunking with --force to rebuild it."
        )
    try:
        return ChunksManifest.model_validate_json(raw)
    except ValidationError as e:
        raise StorageError(f"{path}: invalid manifest:\n{e}") from e


def load_chunks(doc_dir: Path) -> list[Chunk]:
    manifest = load_chunks_manifest(doc_dir)
    path = doc_dir / CHUNKS_FILE
    if not path.exists():
        raise StorageError(f"Missing {path}")
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                chunks.append(Chunk.model_validate_json(line))
            except ValidationError as e:
                raise StorageError(f"{path}:{lineno}: invalid chunk:\n{e}") from e
    if len(chunks) != manifest.chunk_count:
        raise StorageError(
            f"{path}: {len(chunks)} chunks, manifest says {manifest.chunk_count} "
            "(truncated or mismatched file)"
        )
    return chunks


def chunks_staleness_reason(doc_dir: Path, config: ChunkerConfig) -> str | None:
    """Why the chunks must be rebuilt, or None if they are up to date."""
    try:
        manifest = load_chunks_manifest(doc_dir)
    except StorageError as e:
        return str(e).splitlines()[0]
    if manifest.chunker_version != CHUNKER_VERSION:
        return f"chunker version {manifest.chunker_version} -> {CHUNKER_VERSION}"
    if manifest.config.model_dump(mode="json") != config.model_dump(mode="json"):
        return "chunker config changed"
    if manifest.blocks_sha256 != file_sha256(doc_dir / BLOCKS_FILE):
        return "blocks changed (re-ingested with different output)"
    return None
