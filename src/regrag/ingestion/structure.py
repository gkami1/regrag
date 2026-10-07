"""Structure analysis: scopes, section tree and breadcrumbs from numbering.

In regulations the numbering *is* the hierarchy: 7.6.1 is a child of 7.6, which
is a child of 7. We walk body blocks in reading order with a stack of open
sections:

    "7.6. Exits"          stack = [7.6]
    "7.6.1. Number of…"   stack = [7.6, 7.6.1]
    "(a) ..."             no number -> inherits the stack
    "7.7. Interior…"      pop until the top is a prefix of 7.7, push -> [7.7]

Numbering restarts in every annex/appendix, so a section number only means
something inside its *scope*: (scope, number) is the real key. A scope starts at
a block that is both set in a heading style and matches the profile's scope
pattern (e.g. "Annex 3") — two independent signals, so a table cell reading
"Appendix 1)" can't open a scope.

Each block gets a breadcrumb such as:
    ["Annex 3: Requirements to be met by all vehicles", "7.6 Exits", "7.6.1 Number of exits"]

Deciding whether a numbered block is a *label* ("7.6.1. Number of exits") or
*content* ("7.6.1.1. The number of doors shall be…") is heuristic and will
sometimes be wrong. The design keeps that cheap: the tree comes from numbers
alone, and the label/content decision only affects how a node is named.
"""

import logging
import re
from dataclasses import dataclass, field

from pydantic import ConfigDict

from regrag.ingestion.models import Model, ParsedBlock, ScopeInfo, StructureReport

logger = logging.getLogger(__name__)


class DocProfile(Model):
    """Conventions of a document family. Everything else in the pipeline is generic.

    Patterns use inline flags like `(?i)` rather than `re.IGNORECASE`, so that the
    pattern string alone (which is what gets serialised) reproduces the regex.
    """

    model_config = ConfigDict(frozen=True)

    name: str  # recorded in the processed-document manifest
    # Group 1 must capture the number ("7.6.1").
    numbering_re: re.Pattern[str]
    # Matches the start of a scope heading; group 1 is the scope id.
    scope_re: re.Pattern[str]
    # Scope for numbered text before the first annex.
    main_scope: str
    # Placeholder sections that are never labels.
    placeholder_re: re.Pattern[str] = re.compile(r"(?i)^[(\[]?(reserved|deleted)[)\]]?\.?$")
    max_label_chars: int = 80  # a label is short...
    max_title_chars: int = 200  # cap for merged multi-block scope titles


_DASH = r"\s*[-–—]\s*"

UNECE_PROFILE = DocProfile(
    name="unece",
    # "1.", "7.6.1." followed by text. UNECE always writes the trailing dot.
    numbering_re=re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+\S"),
    # "Annex 3", "Annex 1 – Part 1 – Appendix 2", "Annex 3 - Appendix"
    scope_re=re.compile(rf"(?i)^\s*(Annex\s+\d+[A-Z]?(?:{_DASH}(?:Part|Appendix)(?:\s+\d+)?)*)"),
    main_scope="Regulation",
)

FRONT_MATTER = "Front matter"


@dataclass
class _Section:
    number: tuple[int, ...]
    label: str

    def crumb(self) -> str:
        return f"{'.'.join(map(str, self.number))} {self.label}"


@dataclass
class _State:
    scope: str = FRONT_MATTER
    scope_title: str = ""
    stack: list[_Section] = field(default_factory=list)
    last_number: tuple[int, ...] | None = None
    # Heading level of the scope heading. Right after it, until body text appears,
    # unnumbered headings at most one level below it continue the scope title
    # ("Annex 3" / "Requirements to be met…"); deeper ones are captions
    # ("Figure 1 …", "Table 1 …"). None = not collecting.
    title_level: int | None = None


def _normalize_scope(raw: str) -> str:
    return re.sub(_DASH, " - ", raw.strip())


def _join_lines(left: str, right: str) -> str:
    """Join title fragments, repairing words hyphenated across lines ("power-" + "operated")."""
    if left.endswith("-") and right[:1].islower():
        return left + right
    return f"{left} {right}" if left else right


def _is_label(rest: str, profile: DocProfile) -> bool:
    """Does the text after the number read like a heading rather than a sentence?"""
    rest = rest.strip().strip('"“”')
    if not rest or profile.placeholder_re.match(rest):
        return False
    if len(rest) > profile.max_label_chars:
        return False
    if rest[-1] in ".;:,":
        return False
    return not re.search(r"\b(and|or)$", rest)


def _short(text: str, words: int = 8) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def _breadcrumb(state: _State) -> list[str]:
    head = f"{state.scope}: {state.scope_title}" if state.scope_title else state.scope
    return [head, *(s.crumb() for s in state.stack)]


def annotate_structure(
    blocks: list[ParsedBlock],
    profile: DocProfile = UNECE_PROFILE,
) -> StructureReport:
    """Set scope, section_number, block_type and breadcrumb on body blocks, in place."""
    report = StructureReport()
    state = _State()

    for b in blocks:
        text = b.text
        scope_match = profile.scope_re.match(text) if b.heading_level else None
        num_match = profile.numbering_re.match(text)

        if scope_match:
            # New annex/appendix: numbering restarts, close all open sections.
            state = _State(
                scope=_normalize_scope(scope_match.group(1)), title_level=b.heading_level
            )
            state.scope_title = text[scope_match.end() :].strip(" -–—:")
            report.scopes.append(
                ScopeInfo(id=state.scope, title=state.scope_title, first_page=b.page_number)
            )
            b.block_type = "heading"

        elif num_match:
            if state.scope == FRONT_MATTER:
                state.scope = profile.main_scope
                report.scopes.append(ScopeInfo(id=state.scope, title="", first_page=b.page_number))
            number = tuple(int(n) for n in num_match.group(1).split("."))
            rest = text[num_match.end(1) + 1 :]
            is_label = bool(b.heading_level) or _is_label(rest, profile)

            if state.last_number is not None and number < state.last_number:
                report.backward_jumps.append(
                    f"p.{b.page_number} {state.scope}: "
                    f"{'.'.join(map(str, state.last_number))} -> {num_match.group(1)}"
                )
            # Close sections that are not ancestors of this one, then open it.
            while state.stack and state.stack[-1].number != number[: len(state.stack[-1].number)]:
                state.stack.pop()
            if state.stack and state.stack[-1].number == number:
                state.stack.pop()  # same number again: replace, don't nest under itself
            label = rest.strip() if is_label else _short(rest)
            state.stack.append(_Section(number, label))

            state.last_number = number
            state.title_level = None
            b.section_number = num_match.group(1)
            b.block_type = "heading" if is_label else "text"
            report.numbered_blocks += 1

        elif (
            b.heading_level
            and state.title_level is not None
            and b.heading_level <= state.title_level + 1
        ):
            # Continuation of the scope title ("Annex 3" / "Requirements to be met…").
            merged = _join_lines(state.scope_title, text)
            state.scope_title = merged[: profile.max_title_chars]
            report.scopes[-1].title = state.scope_title
            b.block_type = "heading"

        else:
            # Unnumbered: body text, a list item "(a) …", or a caption. Inherits context.
            b.block_type = "heading" if b.heading_level else "text"
            state.title_level = None

        b.scope = state.scope
        b.breadcrumb = _breadcrumb(state)
        if b.block_type == "heading":
            report.heading_blocks += 1

    logger.info(
        "Structure: %d scopes, %d numbered blocks, %d backward jumps",
        len(report.scopes),
        report.numbered_blocks,
        len(report.backward_jumps),
    )
    return report
