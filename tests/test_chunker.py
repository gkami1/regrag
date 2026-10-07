"""Unit tests for structure-aware chunking.

Uses WordCounter (1 token = 1 word) and small budgets, so sizes can be checked
by hand and no tokenizer download is needed.
"""

import pytest
from pydantic import ValidationError

from regrag.chunking import Chunker, ChunkerConfig, WordCounter
from regrag.ingestion.models import PageInfo, ParsedBlock, ParsedDocument

CONFIG = ChunkerConfig(target_tokens=40, max_tokens=60, min_tokens=8, max_scope_title_words=3)
REQ = "Annex 3: Requirements"
EXITS = (REQ, "7 Requirements", "7.6 Exits")


def words(n: int, word: str = "w") -> str:
    return " ".join([word] * n)


class DocBuilder:
    def __init__(self) -> None:
        self.blocks: list[ParsedBlock] = []

    def add(self, text: str, path: tuple[str, ...], scope: str = "Annex 3", page: int = 1):
        number = text.split(". ", 1)[0] if text[:1].isdigit() else None
        self.blocks.append(
            ParsedBlock(
                id=f"d:{len(self.blocks):05d}",
                text=text,
                page_number=page,
                bbox=(0, 0, 1, 1),
                scope=scope,
                section_number=number,
                breadcrumb=list(path),
            )
        )
        return self

    def build(self) -> ParsedDocument:
        return ParsedDocument(
            doc_id="d",
            source_path="d.pdf",
            pages=[PageInfo(number=1, width=1, height=1)],
            blocks=self.blocks,
        )


def chunk(doc: ParsedDocument, config: ChunkerConfig = CONFIG):
    return Chunker(config, WordCounter()).chunk_document(doc)


def big_section() -> ParsedDocument:
    """7.6 Exits with three subsections of ~25 words each: too big for one chunk."""
    b = DocBuilder().add("7.6. Exits", EXITS)
    for i in (1, 2, 3):
        sub = (*EXITS, f"7.6.{i} Sub {i}")
        b.add(f"7.6.{i}. Sub {i}", sub)
        b.add(f"7.6.{i}.1. {words(22)}", (*sub, f"7.6.{i}.1 w w w"))
    return b.build()


def test_small_section_is_one_chunk_with_common_header():
    doc = (
        DocBuilder()
        .add("7.6. Exits", EXITS)
        .add("7.6.1. Number of exits", (*EXITS, "7.6.1 Number of exits"))
        .add(
            "7.6.1.1. The minimum number of doors shall be two.", (*EXITS, "7.6.1 Number of exits")
        )
        .build()
    )
    chunks, _ = chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].header == "d | Annex 3: Requirements > 7 Requirements > 7.6 Exits"
    assert chunks[0].sections == ["7.6", "7.6.1", "7.6.1.1"]
    assert chunks[0].embed_text.startswith(chunks[0].header + "\n\n")


def test_every_block_appears_exactly_once_in_order():
    doc = big_section()
    chunks, report = chunk(doc)
    assert [i for c in chunks for i in c.block_ids] == [b.id for b in doc.blocks]
    assert report.chunk_count == len(chunks) > 1


def test_no_chunk_exceeds_max_tokens_header_included():
    chunks, report = chunk(big_section())
    assert all(c.token_count <= CONFIG.max_tokens for c in chunks)
    assert all(c.token_count == len(c.embed_text.split()) for c in chunks)
    assert report.tokens_max <= CONFIG.max_tokens


def test_tiny_heading_is_merged_not_left_alone():
    chunks, _ = chunk(big_section())
    # "7.6. Exits" (2 words) must not be a chunk of its own.
    assert all(c.text != "7.6. Exits" for c in chunks)
    assert chunks[0].text.startswith("7.6. Exits\n7.6.1. Sub 1")


def test_chunks_never_cross_scopes():
    b = DocBuilder()
    b.add("1. Scope", ("Regulation", "1 Scope"), scope="Regulation")
    b.add("1.1. Applies to buses.", ("Regulation", "1 Scope"), scope="Regulation")
    b.add("1. General", ("Annex 8: Accessibility", "1 General"), scope="Annex 8")
    chunks, _ = chunk(b.build())
    assert [c.scope for c in chunks] == ["Regulation", "Annex 8"]


def test_oversized_block_is_split_into_sentences():
    long_text = "7.7.1. " + " ".join(f"Sentence {i} {words(15)}." for i in range(6))
    doc = DocBuilder().add(long_text, (*EXITS, "7.7.1 w w w")).build()
    chunks, report = chunk(doc)
    assert report.split_blocks == 1
    assert len(chunks) > 1
    assert all(c.token_count <= CONFIG.max_tokens for c in chunks)
    assert all(c.block_ids == ["d:00000"] for c in chunks)  # all from the one block
    assert " ".join(c.text for c in chunks).split() == long_text.split()  # nothing lost


def test_front_matter_is_excluded():
    b = DocBuilder()
    b.add("UNITED NATIONS", ("Front matter",), scope="Front matter")
    b.add("1. Scope", ("Regulation", "1 Scope"), scope="Regulation")
    chunks, report = chunk(b.build())
    assert report.excluded_blocks == 1
    assert [c.text for c in chunks] == ["1. Scope"]


def test_long_scope_title_is_truncated_in_header():
    path = ("Annex 8: Accommodation and accessibility for passengers with reduced mobility",)
    chunks, _ = chunk(DocBuilder().add("Some text here.", path, scope="Annex 8").build())
    assert chunks[0].header == "d | Annex 8: Accommodation and accessibility…"
    assert chunks[0].section_path == list(path)  # metadata keeps the full title


def test_late_scope_title_is_used_for_the_whole_scope():
    # The parser completes a scope title incrementally; early blocks carry the short head.
    b = DocBuilder()
    b.add("Annex 3", ("Annex 3",))
    b.add("Requirements", ("Annex 3: Requirements",))
    b.add("7. Requirements", (REQ, "7 Requirements"))
    chunks, _ = chunk(b.build())
    assert len(chunks) == 1
    assert chunks[0].section_path == [REQ]


def test_invalid_budgets_are_rejected():
    with pytest.raises(ValidationError):
        ChunkerConfig(target_tokens=600, max_tokens=512)
    with pytest.raises(ValidationError):
        ChunkerConfig(min_tokens=400, target_tokens=350)
