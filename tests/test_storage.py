"""Tests for saving/loading processed documents and cache invalidation.

Uses a small synthetic document and a fake "PDF" (just bytes): storage never
opens the PDF, it only fingerprints it.
"""

import json

import pytest
from pydantic import ValidationError

from regrag.ingestion import storage
from regrag.ingestion.layout import LayoutConfig
from regrag.ingestion.models import PageInfo, ParsedBlock, ParsedDocument, StructureReport
from regrag.ingestion.pdf_parser import ParserConfig
from regrag.ingestion.storage import (
    BLOCKS_FILE,
    MANIFEST_FILE,
    SchemaVersionError,
    StorageError,
    load_document,
    save_document,
    staleness_reason,
)

CONFIG = ParserConfig()


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.7 fake content")
    return path


@pytest.fixture
def doc() -> ParsedDocument:
    blocks = [
        ParsedBlock(
            id="test-doc:00000",
            text="E/ECE/324/Rev.3",
            page_number=1,
            bbox=(70.0, 34.3, 520.0, 54.7),
            role="header",
        ),
        ParsedBlock(
            id="test-doc:00001",
            text="7.6.1. Number of exits",
            page_number=1,
            bbox=(70.0, 80.0, 520.0, 92.5),
            font_size=10.0,
            block_type="heading",
            section_number="7.6.1",
            scope="Annex 3",
            breadcrumb=["Annex 3: Requirements", "7.6 Exits", "7.6.1 Number of exits"],
        ),
        ParsedBlock(
            id="test-doc:00002",
            text="Приложение А (обязательное)",  # Cyrillic must survive and stay readable
            page_number=2,
            bbox=(70.0, 80.0, 520.0, 92.5),
            heading_level=1,
        ),
    ]
    return ParsedDocument(
        doc_id="test-doc",
        source_path="data/raw/doc.pdf",
        pages=[
            PageInfo(number=1, width=595, height=842),
            PageInfo(number=2, width=842, height=595),
        ],
        blocks=blocks,
        structure=StructureReport(numbered_blocks=1),
    )


def test_round_trip_is_lossless(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path / "processed")
    loaded = load_document(out)
    assert loaded.blocks == doc.blocks
    assert loaded.pages == doc.pages
    assert loaded.structure == doc.structure
    assert loaded.blocks[1].bbox == (70.0, 80.0, 520.0, 92.5)  # tuple restored from JSON list


def test_files_are_one_block_per_line_and_human_readable(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    lines = (out / BLOCKS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "Приложение" in lines[2]  # not Пр...
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == storage.SCHEMA_VERSION
    assert manifest["config"]["profile"]["name"] == "unece"
    assert not list(out.glob("*.tmp"))  # temp files were renamed away


def test_fresh_output_is_up_to_date(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    assert staleness_reason(out, pdf, CONFIG) is None


def test_changed_pdf_is_stale(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    pdf.write_bytes(b"%PDF-1.7 a new revision")
    assert "sha256" in staleness_reason(out, pdf, CONFIG)


def test_changed_config_is_stale(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    other = ParserConfig(layout=LayoutConfig(min_repeat_ratio=0.5))
    assert staleness_reason(out, pdf, other) == "parser config changed"


def test_new_parser_version_is_stale(tmp_path, pdf, doc, monkeypatch):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    monkeypatch.setattr(storage, "PARSER_VERSION", "99.0.0")
    assert "parser version" in staleness_reason(out, pdf, CONFIG)


def test_missing_output_is_stale(tmp_path, pdf):
    assert "No manifest" in staleness_reason(tmp_path / "never-parsed", pdf, CONFIG)


def test_other_schema_version_is_refused(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    path = out / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SchemaVersionError, match="schema version 999"):
        load_document(out)


def test_corrupted_block_is_reported_with_line_number(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    path = out / BLOCKS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"role":"body"', '"role":"fotter"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(StorageError, match=rf"{BLOCKS_FILE}:2: invalid block"):
        load_document(out)


def test_truncated_blocks_file_is_detected(tmp_path, pdf, doc):
    out = save_document(doc, pdf, CONFIG, tmp_path)
    path = out / BLOCKS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    with pytest.raises(StorageError, match="2 blocks, manifest says 3"):
        load_document(out)


def test_unsafe_doc_id_is_rejected():
    with pytest.raises(ValidationError):
        ParsedDocument(doc_id="../../etc", source_path="x.pdf", pages=[])
