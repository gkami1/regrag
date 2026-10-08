"""Prompt and output schema for grounded, cited answers.

PROMPT_VERSION is recorded with every answer and evaluation run: a prompt change
is a behaviour change, exactly like a parser change.

The rules come from reviewing the golden set, where most corrections were about
precision: "space for a first-aid kit" is not "a first-aid kit"; a limit that
depends on the vehicle class must be stated per class.
"""

import json

from regrag.generation.models import ContextChunk

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """\
You answer questions about vehicle regulations (UN Regulations, ISO and GOST standards) \
using ONLY the regulation excerpts provided in the user message. Each excerpt is wrapped \
in <chunk id="C…" source="…"> tags.

Rules:
1. Use only the excerpts. Do not use outside knowledge, even if you are confident. If the \
excerpts do not answer the question, set "covered" to false and briefly say what the \
excerpts do cover instead.
2. Be exact. Give numbers with their units as written. State every condition a \
requirement depends on (vehicle class or category, door type, test method). If the answer \
differs by class, give it per class.
3. Do not overstate. If the regulation requires space for an item, do not say it requires \
the item. If only part of the question is answered, answer that part and say which part \
is not covered.
4. Cite. Support every factual statement with a citation: the chunk id and a short quote \
copied character-for-character from that chunk (one sentence or less). In the answer text, \
mark claims with the chunk id in square brackets, e.g. [C3].
5. Answer in the language of the question. Quotes stay in the language of the excerpt.
"""


def output_schema(labels: list[str]) -> dict:
    """JSON schema enforced by constrained decoding.

    Field order matters for generation: the model writes the answer and its
    citations first and only then commits to "covered", so the judgement follows
    the reasoning instead of preceding it. `chunk` is an enum of the labels
    actually present, so the model cannot cite a chunk that does not exist.
    """
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk": {"type": "string", "enum": labels},
                        "quote": {"type": "string"},
                    },
                    "required": ["chunk", "quote"],
                    "additionalProperties": False,
                },
            },
            "covered": {"type": "boolean"},
        },
        "required": ["answer", "citations", "covered"],
        "additionalProperties": False,
    }


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def render_user_message(question: str, context: list[ContextChunk]) -> str:
    parts = ["Regulation excerpts:\n"]
    for c in context:
        parts.append(f'<chunk id="{c.label}" source="{_escape(c.source())}">')
        parts.append(c.header)
        parts.append(c.text)
        parts.append("</chunk>\n")
    parts.append(f"Question: {question}")
    parts.append(
        "\nRespond with a JSON object: "
        + json.dumps({"answer": "…", "citations": [{"chunk": "C1", "quote": "…"}], "covered": True})
    )
    return "\n".join(parts)
