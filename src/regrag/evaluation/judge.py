"""LLM-as-judge for answer correctness, with a regulation-specific rubric.

The judge sees the question, the reference (golden) answer, the TEXT of the gold
sections, and the system's answer with its quotes. Seeing the evidence lets it
separate "worded differently from the reference" from "wrong per the regulation".

Two flags target the failure modes observed in real answers:
  overstated         - claims more than the evidence supports ("a trolleybus can
                       drive without wires" when only dual-mode ones can)
  missing_condition  - drops a condition the requirement depends on ("3 doors"
                       without "2 for a double-decker")

A judge is a measuring instrument: its prompt is versioned (a change re-grades
everything), its verdicts are cached (re-running an evaluation does not re-pay or
re-roll a non-deterministic judge), and it must be calibrated against human labels
before its numbers are trusted.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, ValidationError

from regrag.ingestion.models import Model

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "1"

Verdict = Literal["correct", "partial", "incorrect"]

JUDGE_SYSTEM_PROMPT = """\
You are a strict reviewer of answers to questions about vehicle regulations. You \
receive a QUESTION, a REFERENCE ANSWER written by an expert, the EVIDENCE (the exact \
regulation text that answers the question), and a SYSTEM ANSWER with the quotes it cited.

Judge the SYSTEM ANSWER:
- "correct": every key fact the question needs (values, units, conditions, yes/no) is \
present and consistent with the EVIDENCE. Different wording, other language, or extra \
correct detail is fine.
- "partial": the core is right, but a key fact or condition is missing or imprecise, or \
it adds a claim the EVIDENCE does not support.
- "incorrect": a key fact is wrong, contradicts the EVIDENCE, or the question is not answered.

Flags (independent of the verdict):
- "overstated": the answer claims something stronger or broader than the EVIDENCE \
supports (e.g. "X is required" when only space for X is required; "all" when "some").
- "missing_condition": the answer omits a condition, class distinction or exception that \
the EVIDENCE attaches to its answer.
- "contradicts_reference": the answer states something the REFERENCE ANSWER or EVIDENCE \
says is false.

If the REFERENCE ANSWER and the EVIDENCE disagree, the EVIDENCE wins; say so in the \
explanation. Do not reward confident tone, and do not penalise a correct answer for \
being shorter than the reference.

Reply with a json object exactly like:
{"verdict": "correct|partial|incorrect", "overstated": false, "missing_condition": false, \
"contradicts_reference": false, "explanation": "one or two sentences"}
"""


class JudgeConfig(Model):
    model_config = ConfigDict(frozen=True)

    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    # Thinking on: judging needs careful reading. DeepSeek ignores temperature in
    # thinking mode, so verdicts are not deterministic; hence caching + consistency checks.
    thinking: bool = True
    reasoning_effort: Literal["low", "high", "max"] = "high"
    max_tokens: int = Field(8192, gt=0)  # reasoning + the short JSON verdict


class JudgeVerdict(Model):
    verdict: Verdict
    overstated: bool = False
    missing_condition: bool = False
    contradicts_reference: bool = False
    explanation: str = ""

    @property
    def score(self) -> float:
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[self.verdict]


class JudgeInput(Model):
    question: str
    reference_answer: str
    evidence: str
    system_answer: str
    quotes: list[str]

    def render(self) -> str:
        quotes = "\n".join(f"- {q}" for q in self.quotes) or "(none)"
        return (
            f"QUESTION:\n{self.question}\n\n"
            f"REFERENCE ANSWER:\n{self.reference_answer}\n\n"
            f"EVIDENCE:\n{self.evidence}\n\n"
            f"SYSTEM ANSWER:\n{self.system_answer}\n\n"
            f"QUOTES CITED BY THE SYSTEM:\n{quotes}"
        )


class LLMJudge:
    """Judges answers through an OpenAI-compatible API (DeepSeek), with an on-disk cache.

    `client` is an `openai.OpenAI` pointed at the judge provider (or a fake in tests).
    """

    def __init__(self, client: Any, config: JudgeConfig, cache_path: Path | None = None) -> None:
        self.client = client
        self.cfg = config
        self.cache_path = cache_path
        self._cache: dict[str, JudgeVerdict] = {}
        self.calls = 0  # API calls actually made (cache misses)
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._cache[row["key"]] = JudgeVerdict.model_validate(row["verdict"])

    def cache_key(self, inp: JudgeInput) -> str:
        """Everything that can change a verdict is part of the key."""
        material = json.dumps(
            [JUDGE_PROMPT_VERSION, self.cfg.model, self.cfg.thinking, inp.model_dump()],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def judge(self, inp: JudgeInput, use_cache: bool = True) -> JudgeVerdict:
        key = self.cache_key(inp)
        if use_cache and key in self._cache:
            return self._cache[key]

        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": inp.render()},
            ],
            max_tokens=self.cfg.max_tokens,
            response_format={"type": "json_object"},
            reasoning_effort=self.cfg.reasoning_effort,
            extra_body={"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
        )
        self.calls += 1
        content = response.choices[0].message.content or ""
        try:
            verdict = JudgeVerdict.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            # Not cached: a failed judgment should be retried next run, not frozen.
            raise JudgeError(f"unparseable judge output: {content[:200]!r}") from e

        if use_cache:
            self._cache[key] = verdict
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    row = {"key": key, "verdict": verdict.model_dump()}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return verdict


class JudgeError(RuntimeError):
    pass
