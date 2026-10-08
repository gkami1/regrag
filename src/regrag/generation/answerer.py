"""LLM backends that turn (question, context) into a cited Answer.

The pipeline depends only on the `Answerer` protocol. `VLLMAnswerer` talks to any
OpenAI-compatible server (vLLM speaks that protocol); a Claude backend can sit
next to it for comparison on the same golden set.

Grounding without native citations: the output is constrained to a JSON schema
(chunk ids limited to the labels in the context), and then every quote is checked
to occur verbatim in the chunk it claims to come from. A quote the model made up
or paraphrased fails that check and is reported as unverified.
"""

import json
import logging
import re
import time
import unicodedata
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from regrag.generation.models import Answer, Citation, ContextChunk, Usage
from regrag.generation.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    output_schema,
    render_user_message,
)
from regrag.ingestion.models import Model

logger = logging.getLogger(__name__)


class Answerer(Protocol):
    @property
    def model_id(self) -> str: ...

    def answer(self, question: str, context: list[ContextChunk]) -> Answer: ...


class VLLMConfig(Model):
    model_config = ConfigDict(frozen=True)

    model: str  # the name the server serves, e.g. "Qwen/Qwen3.8-27B-FP8"
    # 0 = deterministic: reproducible evaluation runs, and no creativity wanted here.
    temperature: float = Field(0.0, ge=0)
    max_tokens: int = Field(2048, gt=0)
    # Qwen3-family chat templates switch "thinking" on/off per request. Off by
    # default: lower latency; whether thinking improves answers is an experiment.
    enable_thinking: bool = False


# --- parsing and verification ---------------------------------------------------


class _RawCitation(BaseModel):
    chunk: str
    quote: str


class _RawOutput(BaseModel):
    answer: str
    citations: list[_RawCitation] = []
    covered: bool


_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})


def normalize_for_match(text: str) -> str:
    """Make a verbatim check robust to formatting, not to wording.

    Unicode normal form, typographic quotes/dashes and whitespace runs differ
    between PDF extraction and model output without changing the words.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS)
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_citations(raw: list[_RawCitation], context: list[ContextChunk]) -> list[Citation]:
    by_label = {c.label: c for c in context}
    out = []
    for rc in raw:
        chunk = by_label.get(rc.chunk)
        quote = normalize_for_match(rc.quote)
        verified = bool(chunk and quote and quote in normalize_for_match(chunk.text))
        out.append(
            Citation(
                chunk_label=rc.chunk,
                quote=rc.quote,
                verified=verified,
                chunk_id=chunk.chunk_id if chunk else None,
                scope=chunk.scope if chunk else None,
                sections=chunk.sections if chunk else [],
                page_start=chunk.page_start if chunk else None,
            )
        )
    return out


# --- vLLM / OpenAI-compatible backend ------------------------------------------------


class VLLMAnswerer:
    """`client` is an `openai.OpenAI` instance pointed at the vLLM server (or a fake in tests)."""

    def __init__(self, client: Any, config: VLLMConfig) -> None:
        self.client = client
        self.cfg = config

    @property
    def model_id(self) -> str:
        return f"vllm:{self.cfg.model}"

    def answer(self, question: str, context: list[ContextChunk]) -> Answer:
        labels = [c.label for c in context]
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_message(question, context)},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            # Constrained decoding: the server only lets the model emit tokens that
            # keep the output valid against this schema.
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "cited_answer", "schema": output_schema(labels)},
            },
            # vLLM-specific request fields travel in extra_body.
            extra_body={"chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking}},
        )
        elapsed = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        usage = Usage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )
        base = dict(
            question=question,
            context=context,
            model=self.model_id,
            prompt_version=PROMPT_VERSION,
            usage=usage,
            timings_ms={"llm_ms": round(elapsed, 1)},
            finish_reason=choice.finish_reason,
        )
        content = choice.message.content or ""
        try:
            raw = _RawOutput.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            # Most often finish_reason == "length": the JSON was cut off by max_tokens.
            logger.warning("unparseable model output (%s): %r", choice.finish_reason, content[:200])
            return Answer(
                covered=False, text=content, citations=[], error=f"{type(e).__name__}", **base
            )
        return Answer(
            covered=raw.covered,
            text=raw.answer,
            citations=verify_citations(raw.citations, context),
            **base,
        )
