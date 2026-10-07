"""The golden evaluation set: schema, loading, corpus checks, dev/test split, leakage.

Format and authoring rules: evals/README.md. Gold labels point at document
*sections* (scope + section number), never at chunk ids, so the set stays valid
when chunking changes. That is what makes "chunk size 350 vs 800" measurable.
"""

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError, model_validator

from regrag.ingestion.models import Model, ParsedBlock
from regrag.ingestion.storage import StorageError
from regrag.io_utils import write_lines_atomic

QuestionType = Literal[
    "numeric",
    "definition",
    "applicability",
    "comparison",
    "cross_reference",
    "vocabulary_mismatch",
    "unanswerable",
]
Status = Literal["draft", "verified", "rejected"]
Split = Literal["dev", "test"]

TEST_SHARE = 0.3


class GoldRef(Model):
    doc_id: str
    scope: str
    section: str | None = None  # "7.6.1.1"; None for unnumbered content
    page: int | None = None  # required when section is None

    @model_validator(mode="after")
    def _section_or_page(self) -> Self:
        if self.section is None and self.page is None:
            raise ValueError("a gold reference needs a section or, for unnumbered content, a page")
        return self

    def label(self) -> str:
        where = self.section if self.section is not None else f"p.{self.page}"
        return f"{self.scope} / {where}"


class GoldItem(Model):
    id: str
    question: str
    lang: Literal["en", "ru"]
    type: QuestionType
    gold: list[GoldRef]
    answer: str | None
    source: str
    status: Status = "draft"
    notes: str = ""

    @model_validator(mode="after")
    def _unanswerable_has_no_gold(self) -> Self:
        if (self.type == "unanswerable") != (not self.gold):
            raise ValueError("unanswerable items must have gold [], all others at least one gold")
        if self.type == "unanswerable" and self.answer is not None:
            raise ValueError("unanswerable items must have answer null")
        return self

    @property
    def split(self) -> Split:
        """Deterministic from the id, so no author can choose which split a question lands in."""
        bucket = int(hashlib.sha256(self.id.encode()).hexdigest(), 16) % 100
        return "test" if bucket < TEST_SHARE * 100 else "dev"


def load_golden(path: Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = GoldItem.model_validate_json(line)
            except ValidationError as e:
                raise StorageError(f"{path}:{lineno}: invalid golden item:\n{e}") from e
            if item.id in seen:
                raise StorageError(f"{path}:{lineno}: duplicate id {item.id!r}")
            seen.add(item.id)
            items.append(item)
    return items


def save_golden(path: Path, items: Iterable[GoldItem]) -> None:
    """Atomic rewrite; keeps non-ASCII (Russian) readable for review in diffs."""
    write_lines_atomic(
        path, (json.dumps(i.model_dump(mode="json"), ensure_ascii=False) for i in items)
    )


# --- checks against the corpus ---------------------------------------------------


def _covers(section: str | None, gold_section: str) -> bool:
    return section is not None and (
        section == gold_section or section.startswith(gold_section + ".")
    )


def gold_text(ref: GoldRef, blocks: list[ParsedBlock]) -> str:
    """Body text of the gold section and its subsections (or of the page, if unnumbered)."""
    parts = []
    inside = False  # are we currently within the gold section?
    for b in blocks:
        if b.role != "body":
            continue
        if b.scope != ref.scope:
            inside = False
            continue
        if ref.section is None:
            if b.page_number == ref.page:
                parts.append(b.text)
            continue
        # A numbered block opens or closes the section; unnumbered blocks (list
        # items, table rows) belong to the most recent numbered one.
        if b.section_number is not None:
            inside = _covers(b.section_number, ref.section)
        if inside:
            parts.append(b.text)
    return "\n".join(parts)


def check_against_corpus(
    items: list[GoldItem], blocks_by_doc: dict[str, list[ParsedBlock]]
) -> list[str]:
    """Problems that schema validation cannot see: gold pointing at nothing."""
    problems = []
    for item in items:
        for ref in item.gold:
            blocks = blocks_by_doc.get(ref.doc_id)
            if blocks is None:
                problems.append(f"{item.id}: unknown doc_id {ref.doc_id!r}")
            elif not gold_text(ref, blocks):
                problems.append(f"{item.id}: gold {ref.label()} matches no text in {ref.doc_id}")
    return problems


# --- leakage -----------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def longest_shared_run(question: str, source: str) -> int:
    """Longest run of consecutive words the question shares with the source text.

    High values mean the question was written in the document's own words, which
    makes retrieval look easier than it is for real users.
    """
    q = _WORD_RE.findall(question.lower())
    s = _WORD_RE.findall(source.lower())
    best = 0
    prev = [0] * (len(s) + 1)
    for qw in q:
        cur = [0] * (len(s) + 1)
        for j, sw in enumerate(s, start=1):
            if qw == sw:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best
