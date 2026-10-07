"""Chunk data model and chunker configuration."""

from typing import Self

from pydantic import ConfigDict, Field, model_validator

from regrag.ingestion.models import DocId, Model


class ChunkerConfig(Model):
    """Every setting that affects chunker output; recorded in the chunks manifest.

    Budgets are in tokens of the *embedding model's* tokenizer, because that is
    where the limit applies (and characters per token differ a lot between
    English and Russian). They are hyperparameters: the evaluation stage decides
    the final values, not intuition.
    """

    model_config = ConfigDict(frozen=True)

    tokenizer: str = "BAAI/bge-m3"
    target_tokens: int = Field(350, gt=0)  # pack siblings up to this size...
    max_tokens: int = Field(512, gt=0)  # ...never exceed this (header included)
    min_tokens: int = Field(64, ge=0)  # chunks below this get merged with a neighbour if possible
    max_scope_title_words: int = Field(8, ge=0)  # keeps the header compact
    exclude_scopes: tuple[str, ...] = ("Front matter",)

    @model_validator(mode="after")
    def _check_budgets(self) -> Self:
        if not self.min_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("budgets must satisfy min_tokens < target_tokens <= max_tokens")
        return self


class Chunk(Model):
    id: str  # "<doc_id>:c<index>", e.g. "unece-r107-rev9:c00042"
    doc_id: DocId
    index: int = Field(ge=0)
    # What gets embedded: compact breadcrumb header + body. The header gives the
    # vector its context ("Exits" for a paragraph that only says "doors").
    embed_text: str
    header: str
    text: str  # body only: what the LLM is shown and what citations quote
    token_count: int  # tokens of embed_text, special tokens included
    scope: str  # "Annex 3"
    # Common breadcrumb of everything in the chunk: scope head, then sections.
    # Parent of the chunk for small-to-big expansion at generation time.
    section_path: list[str]
    sections: list[str] = Field(default_factory=list)  # numbers covered, in order: ["7.6.1.1", ...]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    block_ids: list[str]  # source blocks, in order (a split block appears in several chunks)


class ChunkingReport(Model):
    chunk_count: int
    input_blocks: int  # body blocks considered
    excluded_blocks: int  # body blocks in excluded scopes (front matter)
    split_blocks: int  # blocks too large for one chunk, split at sentence level
    below_min: int  # chunks that stayed under min_tokens (nothing to merge with)
    tokens_min: int
    tokens_median: int
    tokens_p90: int
    tokens_max: int
