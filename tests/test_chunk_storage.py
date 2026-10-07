"""Tests for chunk persistence and chunk cache invalidation."""

import pytest

from regrag.chunking import ChunkerConfig, chunks_staleness_reason, load_chunks, save_chunks
from regrag.chunking import storage as chunk_storage
from regrag.chunking.models import Chunk, ChunkingReport
from regrag.ingestion.storage import BLOCKS_FILE, StorageError

CONFIG = ChunkerConfig()
REPORT = ChunkingReport(
    chunk_count=1,
    input_blocks=2,
    excluded_blocks=0,
    split_blocks=0,
    below_min=0,
    tokens_min=20,
    tokens_median=20,
    tokens_p90=20,
    tokens_max=20,
)


@pytest.fixture
def doc_dir(tmp_path):
    (tmp_path / BLOCKS_FILE).write_text('{"fake": "blocks"}\n', encoding="utf-8")
    return tmp_path


def make_chunk() -> Chunk:
    return Chunk(
        id="d:c00000",
        doc_id="d",
        index=0,
        embed_text="d | Annex 8: Доступность > 3.6 Wheelchair\n\nТекст",
        header="d | Annex 8: Доступность > 3.6 Wheelchair",
        text="Текст",
        token_count=20,
        scope="Annex 8",
        section_path=["Annex 8: Доступность", "3.6 Wheelchair"],
        sections=["3.6"],
        page_start=109,
        page_end=110,
        block_ids=["d:00001", "d:00002"],
    )


def test_round_trip(doc_dir):
    save_chunks(doc_dir, "d", [make_chunk()], CONFIG, REPORT)
    assert load_chunks(doc_dir) == [make_chunk()]
    assert chunks_staleness_reason(doc_dir, CONFIG) is None


def test_changed_blocks_invalidate_chunks(doc_dir):
    save_chunks(doc_dir, "d", [make_chunk()], CONFIG, REPORT)
    (doc_dir / BLOCKS_FILE).write_text('{"fake": "re-ingested"}\n', encoding="utf-8")
    assert "blocks changed" in chunks_staleness_reason(doc_dir, CONFIG)


def test_changed_config_invalidates_chunks(doc_dir):
    save_chunks(doc_dir, "d", [make_chunk()], CONFIG, REPORT)
    assert chunks_staleness_reason(doc_dir, ChunkerConfig(target_tokens=800, max_tokens=1000)) == (
        "chunker config changed"
    )


def test_new_chunker_version_invalidates_chunks(doc_dir, monkeypatch):
    save_chunks(doc_dir, "d", [make_chunk()], CONFIG, REPORT)
    monkeypatch.setattr(chunk_storage, "CHUNKER_VERSION", "9.9.9")
    assert "chunker version" in chunks_staleness_reason(doc_dir, CONFIG)


def test_truncated_chunks_file_is_detected(doc_dir):
    save_chunks(doc_dir, "d", [make_chunk(), make_chunk()], CONFIG, REPORT)
    path = doc_dir / chunk_storage.CHUNKS_FILE
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(StorageError, match="1 chunks, manifest says 2"):
        load_chunks(doc_dir)
