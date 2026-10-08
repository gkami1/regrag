"""Generation data types: the context given to the LLM and the answer it produces."""

from typing import Literal

from pydantic import ConfigDict, Field

from regrag.ingestion.models import Model


class GenerationConfig(Model):
    model_config = ConfigDict(frozen=True)

    context_k: int = Field(10, gt=0)  # reranked hits given to the LLM (R@10 >> R@5 in eval)
    max_expansions: int = Field(3, ge=0)  # extra chunks pulled in by following references


class ContextChunk(Model):
    """One unit of context, labelled C1, C2, ... in the prompt."""

    label: str
    chunk_id: str
    doc_id: str
    scope: str
    sections: list[str]
    page_start: int
    page_end: int
    header: str
    text: str
    index: int  # position in the document, for ordering
    origin: Literal["retrieved", "reference"]
    rerank_score: float | None = None
    via: str | None = None  # for references: the citing text, e.g. "Annex 4, Figure 6"

    def source(self) -> str:
        secs = f"{self.sections[0]}–{self.sections[-1]}" if self.sections else "unnumbered"
        pages = f"p.{self.page_start}" + (
            f"–{self.page_end}" if self.page_end != self.page_start else ""
        )
        return f"{self.doc_id} · {self.scope} · {secs} · {pages}"


class Citation(Model):
    chunk_label: str
    quote: str
    verified: bool  # quote found verbatim (whitespace-normalised) in that chunk
    chunk_id: str | None = None  # None if the label does not exist in the context
    scope: str | None = None
    sections: list[str] = Field(default_factory=list)
    page_start: int | None = None


class Usage(Model):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Answer(Model):
    question: str
    covered: bool  # the model's judgement: does the context answer the question?
    text: str
    citations: list[Citation]
    context: list[ContextChunk]
    model: str
    prompt_version: str
    usage: Usage = Field(default_factory=Usage)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None  # set when the model output could not be parsed

    @property
    def grounded(self) -> bool:
        """A covered answer must be backed by at least one verified quote."""
        return not self.covered or any(c.verified for c in self.citations)
