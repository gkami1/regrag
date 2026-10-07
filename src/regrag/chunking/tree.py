"""Rebuild the section tree from the breadcrumbs stored on each block.

The parser wrote, for every body block, the path of sections it lives in. Walking
the blocks in document order with a stack turns those paths back into a tree:

    Annex 3: Requirements…                 (scope root)
      7 Requirements
        7.6 Exits
          7.6.1 Number of exits            own pieces: "7.6.1. Number of exits"
            7.6.1.1 The minimum number…    own pieces: "7.6.1.1. …", "(a) …"

Unnumbered blocks ("(a) …", captions) always belong to the deepest open section,
so a node's own pieces come before its children: walking own pieces then
children reproduces document order exactly.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from regrag.ingestion.models import ParsedBlock


@dataclass(frozen=True)
class Piece:
    """A unit of text for chunking: a whole block, or one sentence of a block too big to keep."""

    block: ParsedBlock
    text: str
    path: tuple[str, ...]  # breadcrumb, with the scope head normalised (see build_forest)


@dataclass
class SectionNode:
    path: tuple[str, ...]
    pieces: list[Piece] = field(default_factory=list)  # own text, before any child
    children: list["SectionNode"] = field(default_factory=list)

    def all_pieces(self) -> Iterator[Piece]:
        yield from self.pieces
        for child in self.children:
            yield from child.all_pieces()


def _scope_heads(blocks: list[ParsedBlock]) -> dict[str, str]:
    """Final display head for each scope.

    The parser builds a scope's title incrementally ("Annex 3", then
    "Annex 3: Requirements to be met…"), so early blocks of a scope carry a
    shorter head. Use the last (complete) one for every block of the scope.
    """
    heads: dict[str, str] = {}
    for b in blocks:
        if b.scope and b.breadcrumb:
            heads[b.scope] = b.breadcrumb[0]
    return heads


def build_forest(blocks: list[ParsedBlock]) -> list[SectionNode]:
    """One tree per scope, in document order."""
    heads = _scope_heads(blocks)
    roots: list[SectionNode] = []
    stack: list[SectionNode] = []

    for b in blocks:
        path = (heads.get(b.scope or "", b.scope or ""), *b.breadcrumb[1:])
        # Close sections that don't contain this block.
        while stack and path[: len(stack[-1].path)] != stack[-1].path:
            stack.pop()
        if not stack:
            root = SectionNode(path=path[:1])
            roots.append(root)
            stack.append(root)
        # Open any sections between the deepest open one and this block's own.
        for depth in range(len(stack[-1].path) + 1, len(path) + 1):
            node = SectionNode(path=path[:depth])
            stack[-1].children.append(node)
            stack.append(node)
        stack[-1].pieces.append(Piece(block=b, text=b.text, path=path))

    return roots
