"""Tests for reference parsing, context assembly, prompt schema and the vLLM answerer.

The answerer runs against a fake OpenAI-compatible client: no server, no cost,
and the exact request it sends can be inspected.
"""

import json
from types import SimpleNamespace

from fakes import make_chunk

from regrag.generation import (
    ChunkStore,
    GenerationConfig,
    VLLMAnswerer,
    VLLMConfig,
    build_context,
    find_references,
)
from regrag.generation.prompt import output_schema, render_user_message
from regrag.retrieval.models import Hit

# --- references ----------------------------------------------------------------------


def refs(text: str, scope: str = "Annex 3"):
    return [(r.scope, r.section, r.figure, r.cross_scope) for r in find_references(text, scope)]


def test_scoped_paragraph_and_figure_references():
    found = refs("see paragraph 3.3.3. of Annex 5 and Annex 4, Figure 6.")
    assert ("Annex 5", "3.3.3", None, True) in found
    assert ("Annex 4", None, "Figure 6", True) in found


def test_this_annex_and_this_regulation_resolve_to_scopes():
    assert refs("paragraph 7.7.5.1. of this annex") == [("Annex 3", "7.7.5.1", None, False)]
    assert refs("paragraph 2.18. of this Regulation") == [("Regulation", "2.18", None, True)]


def test_bare_paragraph_does_not_steal_a_scoped_reference():
    # Without the (?![\d.]) guard, "paragraph 3.3" would also match here.
    assert refs("paragraph 3.3.3. of Annex 5") == [("Annex 5", "3.3.3", None, True)]


def test_bare_annex_reference_and_annex_13_is_not_annex_1():
    found = refs("contrast according to Annex 5; tested per Annex 13 - Part 1")
    assert ("Annex 5", None, None, True) in found
    assert not any(scope == "Annex 1" for scope, *_ in found)


def test_cross_scope_references_come_first():
    found = refs("paragraph 7.6.1. applies; see also Annex 4, Figure 6")
    assert found[0][3] is True and found[-1][3] is False


# --- context assembly -----------------------------------------------------------------


def chunk(i, text, scope="Annex 3", sections=None, doc_id="doc"):
    c = make_chunk(i, text, doc_id=doc_id, scope=scope)
    return c.model_copy(update={"sections": sections or [f"7.6.{i}"]})


STORE = ChunkStore(
    [
        chunk(0, "7.6.1. Doors shall be two, see Annex 4, Figure 6."),
        chunk(1, "7.6.2. Positioning of exits."),
        chunk(2, "7.6.3. Width according to Annex 5."),
        chunk(3, "Figure 28 Wheelchair space", scope="Annex 4", sections=[]),
        chunk(4, "Figure 6 Gangway gauge 450 mm", scope="Annex 4", sections=[]),
        chunk(5, "1. Contrast C = |r1 - r2| / (r1 + r2)", scope="Annex 5", sections=["1"]),
    ]
)


def hit(c, score: float) -> Hit:
    return Hit(
        chunk_id=c.id,
        doc_id=c.doc_id,
        scope=c.scope,
        section_path=c.section_path,
        sections=c.sections,
        page_start=c.page_start,
        page_end=c.page_end,
        header=c.header,
        text=c.text,
        rerank_score=score,
    )


def test_context_is_in_document_order_with_labels_and_expansions():
    hits = [hit(STORE.by_id["doc:c00002"], 3.0), hit(STORE.by_id["doc:c00000"], 5.0)]
    ctx = build_context(hits, STORE, GenerationConfig(context_k=10, max_expansions=3))
    assert [c.label for c in ctx] == [f"C{i}" for i in range(1, len(ctx) + 1)]
    texts = [c.text for c in ctx]
    assert texts.index("7.6.1. Doors shall be two, see Annex 4, Figure 6.") < texts.index(
        "7.6.3. Width according to Annex 5."
    )
    added = {c.text for c in ctx if c.origin == "reference"}
    # Figure 6 (not Figure 28: word boundary) and the small Annex 5 were followed.
    assert added == {"Figure 6 Gangway gauge 450 mm", "1. Contrast C = |r1 - r2| / (r1 + r2)"}


def test_expansion_budget_and_no_duplicates():
    hits = [hit(STORE.by_id["doc:c00000"], 5.0), hit(STORE.by_id["doc:c00004"], 1.0)]
    ctx = build_context(hits, STORE, GenerationConfig(max_expansions=3))
    assert len({c.chunk_id for c in ctx}) == len(ctx)  # Figure 6 retrieved AND referenced: once
    none = build_context(hits, STORE, GenerationConfig(max_expansions=0))
    assert all(c.origin == "retrieved" for c in none)


def test_large_annex_is_not_pulled_in_whole():
    big = ChunkStore(
        [chunk(i, f"8.{i}. text", scope="Annex 8", sections=[f"8.{i}"]) for i in range(5)]
        + [chunk(9, "see Annex 8", scope="Annex 3", sections=["7.1"])]
    )
    ctx = build_context([hit(big.by_id["doc:c00009"], 1.0)], big)
    assert [c.scope for c in ctx] == ["Annex 3"]


# --- prompt -----------------------------------------------------------------------------


def test_schema_restricts_citations_to_existing_labels():
    schema = output_schema(["C1", "C2"])
    assert schema["properties"]["citations"]["items"]["properties"]["chunk"]["enum"] == ["C1", "C2"]
    assert list(schema["properties"]) == ["answer", "citations", "covered"]  # judgement last


def test_user_message_contains_every_chunk_and_the_question():
    ctx = build_context([hit(STORE.by_id["doc:c00001"], 1.0)], STORE)
    msg = render_user_message("How many doors?", ctx)
    assert '<chunk id="C1" source="doc · Annex 3 · 7.6.1–7.6.1 · p.47">' in msg
    assert "Question: How many doors?" in msg
    assert msg.index("</chunk>") < msg.index("Question:")  # excerpts first, question last


# --- answerer ---------------------------------------------------------------------------


class FakeOpenAI:
    """Mimics openai.OpenAI().chat.completions.create and records the request."""

    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.requests: list[dict] = []
        self._content, self._finish = content, finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content), finish_reason=self._finish
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=80),
        )


CTX = build_context(
    [hit(STORE.by_id["doc:c00000"], 5.0)], STORE, GenerationConfig(max_expansions=0)
)


def answer_with(output: dict | str, **kw):
    content = output if isinstance(output, str) else json.dumps(output)
    client = FakeOpenAI(content, **kw)
    answer = VLLMAnswerer(client, VLLMConfig(model="qwen-test")).answer("How many doors?", CTX)
    return answer, client.requests[0]


def test_request_is_deterministic_constrained_and_thinking_off():
    _, req = answer_with({"answer": "Two [C1].", "citations": [], "covered": True})
    assert req["temperature"] == 0.0
    assert req["response_format"]["json_schema"]["schema"]["properties"]["citations"]
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert [m["role"] for m in req["messages"]] == ["system", "user"]


def test_verbatim_quotes_are_verified_and_invented_ones_are_not():
    answer, _ = answer_with(
        {
            "answer": "Two doors [C1].",
            "citations": [
                {
                    "chunk": "C1",
                    "quote": "Doors  shall be\ntwo",
                },  # whitespace differs: still verbatim
                {"chunk": "C1", "quote": "Doors shall be three"},  # not in the text
            ],
            "covered": True,
        }
    )
    assert [c.verified for c in answer.citations] == [True, False]
    assert answer.citations[0].chunk_id == "doc:c00000" and answer.grounded
    assert answer.usage.prompt_tokens == 1200


def test_covered_answer_without_verified_quote_is_not_grounded():
    answer, _ = answer_with({"answer": "Two.", "citations": [], "covered": True})
    assert not answer.grounded


def test_not_covered_answer_is_grounded_by_definition():
    answer, _ = answer_with({"answer": "Not covered.", "citations": [], "covered": False})
    assert answer.grounded and not answer.covered


def test_truncated_output_is_reported_not_raised():
    answer, _ = answer_with('{"answer": "Two doors because', finish_reason="length")
    assert answer.error == "JSONDecodeError" and answer.finish_reason == "length"
    assert not answer.covered
