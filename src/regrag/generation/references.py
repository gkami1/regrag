"""Find cross-references in regulation text ("see Annex 4, Figure 6").

Regulations answer many questions in two places: the requirement, and the figure,
table or paragraph it points to. The second place rarely resembles the question,
so retrieval misses it; following the reference from the retrieved text finds it.

Measured in R107: ~270 bare "paragraph N" references, ~60 "Annex X, Figure N"
and ~50 "paragraph N of Annex X / of this annex". Cross-scope references are the
ones retrieval cannot reach on its own, so they get priority.
"""

import re
from dataclasses import dataclass

_NUM = r"(\d+(?:\.\d+)*)\.?"
# "paragraph 3.3.3. of Annex 5", "paragraph 7.7.5.1. of this annex",
# "paragraph 2.18 of this Regulation"
_PARA_SCOPED = re.compile(
    rf"paragraphs?\s+{_NUM}\s+(?:of|to|in)\s+(Annex\s+\d+|this\s+annex|this\s+Regulation)", re.I
)
# "Annex 4, Figure 6", "Annex 4, Figures 12A"
_FIGURE = re.compile(r"Annex\s+(\d+),?\s+Figures?\s+(\d+[A-Za-z]?)", re.I)
# "according to Annex 5": a whole annex. Only worth following when the annex is
# small (a self-contained procedure); the resolver enforces that.
_ANNEX_BARE = re.compile(r"Annex\s+(\d+)\b(?!\s*[-–]?\s*(?:Part|Appendix))", re.I)
# "paragraph 7.6.5.1." (same scope as the text it appears in). `(?![\d.])` stops the
# number from backtracking to a shorter match ("3.3" out of "3.3.3. of Annex 5"),
# which would otherwise slip past the "not followed by 'of Annex'" check.
_PARA_BARE = re.compile(
    r"paragraphs?\s+(\d+(?:\.\d+)*)\.?(?![\d.])(?!\s*(?:of|to|in)\s+(?:Annex|this))", re.I
)


@dataclass(frozen=True)
class Reference:
    scope: str  # resolved scope, e.g. "Annex 5"
    section: str | None = None  # "3.3.3"
    figure: str | None = None  # "Figure 6"; section and figure both None = whole annex
    cross_scope: bool = False  # points outside the citing text's scope
    via: str = ""  # the matched text, for display

    @property
    def priority(self) -> int:
        return 0 if self.cross_scope else 1


def _scope_of(target: str, source_scope: str) -> str:
    t = target.lower()
    if t == "this annex":
        return source_scope
    if t == "this regulation":
        return "Regulation"
    return re.sub(r"\s+", " ", target.strip()).title()  # "annex 5" -> "Annex 5"


def find_references(text: str, source_scope: str) -> list[Reference]:
    """All references in `text`, cross-scope first, de-duplicated, in order of appearance."""
    found: list[Reference] = []
    for m in _PARA_SCOPED.finditer(text):
        scope = _scope_of(m.group(2), source_scope)
        found.append(
            Reference(scope, section=m.group(1), cross_scope=scope != source_scope, via=m.group(0))
        )
    for m in _FIGURE.finditer(text):
        scope = f"Annex {m.group(1)}"
        found.append(
            Reference(
                scope,
                figure=f"Figure {m.group(2)}",
                cross_scope=scope != source_scope,
                via=m.group(0),
            )
        )
    for m in _PARA_BARE.finditer(text):
        found.append(Reference(source_scope, section=m.group(1), via=m.group(0)))
    # Bare annex mentions that are not part of a more specific reference above.
    taken = [m.span() for p in (_PARA_SCOPED, _FIGURE) for m in p.finditer(text)]
    for m in _ANNEX_BARE.finditer(text):
        scope = f"Annex {m.group(1)}"
        inside = any(start <= m.start() < end for start, end in taken)
        if not inside and scope != source_scope:
            found.append(Reference(scope, cross_scope=True, via=m.group(0)))

    unique: dict[tuple, Reference] = {}
    for ref in found:
        unique.setdefault((ref.scope, ref.section, ref.figure), ref)
    return sorted(unique.values(), key=lambda r: r.priority)  # stable: keeps text order
