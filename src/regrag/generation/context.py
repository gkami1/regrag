"""Assemble the LLM context: top reranked hits + chunks they reference.

    reranked hits (top context_k)
      + follow references found in them (cross-scope first, at most max_expansions)
      -> de-duplicate -> order by document position -> label C1..Cn

Document order (not score order) because a regulation reads coherently in the
order it was written: "7.6.1" before "7.6.1.1", the requirement before its figure.
"""

import re
from collections import defaultdict
from pathlib import Path

from regrag.chunking.models import Chunk
from regrag.chunking.storage import CHUNKS_MANIFEST_FILE, load_chunks
from regrag.generation.models import ContextChunk, GenerationConfig
from regrag.generation.references import Reference, find_references
from regrag.retrieval.models import Hit


class ChunkStore:
    """All chunks by id and by (doc, scope), for resolving references.

    Read from chunks.jsonl; the same lookups could be Qdrant payload queries.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.by_id = {c.id: c for c in chunks}
        self.by_scope: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for c in chunks:
            self.by_scope[(c.doc_id, c.scope)].append(c)

    @classmethod
    def from_processed(cls, root: Path) -> "ChunkStore":
        chunks: list[Chunk] = []
        for doc_dir in sorted(p for p in root.iterdir() if (p / CHUNKS_MANIFEST_FILE).exists()):
            chunks.extend(load_chunks(doc_dir))
        return cls(chunks)

    def resolve(self, doc_id: str, ref: Reference, max_whole_scope_chunks: int = 2) -> Chunk | None:
        """The chunk a reference points to, or None if it cannot be found."""
        candidates = self.by_scope.get((doc_id, ref.scope), [])
        if ref.section is not None:
            exact = [c for c in candidates if ref.section in c.sections]
            if exact:
                return exact[0]
            # Section not numbered on its own (e.g. merged into a parent's chunk):
            # take the first chunk covering one of its subsections.
            prefix = ref.section + "."
            return next(
                (c for c in candidates if any(s.startswith(prefix) for s in c.sections)), None
            )
        if ref.figure is not None:
            # Figures live in unnumbered diagram annexes; match the caption text.
            # Word boundary: "Figure 2" must not match "Figure 28".
            caption = re.compile(rf"\b{re.escape(ref.figure)}(?![0-9A-Za-z])", re.I)
            return next((c for c in candidates if caption.search(c.text)), None)
        # Whole annex: only a small, self-contained one (e.g. a measurement procedure).
        if 0 < len(candidates) <= max_whole_scope_chunks:
            return candidates[0]
        return None


def _from_chunk(chunk: Chunk, origin: str, **extra) -> ContextChunk:
    return ContextChunk(
        label="",
        chunk_id=chunk.id,
        doc_id=chunk.doc_id,
        scope=chunk.scope,
        sections=chunk.sections,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        header=chunk.header,
        text=chunk.text,
        index=chunk.index,
        origin=origin,
        **extra,
    )


def build_context(
    hits: list[Hit], store: ChunkStore, config: GenerationConfig | None = None
) -> list[ContextChunk]:
    cfg = config or GenerationConfig()
    selected: dict[str, ContextChunk] = {}
    for hit in hits[: cfg.context_k]:
        chunk = store.by_id.get(hit.chunk_id)
        if chunk is not None:
            selected[chunk.id] = _from_chunk(chunk, "retrieved", rerank_score=hit.rerank_score)

    # Collect references from all retrieved chunks, best-ranked chunks first,
    # then cross-scope before same-scope across the whole set.
    refs: list[tuple[Reference, str]] = []
    for ctx in list(selected.values()):
        refs.extend((ref, ctx.doc_id) for ref in find_references(ctx.text, ctx.scope))
    refs.sort(key=lambda pair: pair[0].priority)  # stable

    added = 0
    for ref, doc_id in refs:
        if added >= cfg.max_expansions:
            break
        target = store.resolve(doc_id, ref)
        if target is not None and target.id not in selected:
            selected[target.id] = _from_chunk(target, "reference", via=ref.via)
            added += 1

    ordered = sorted(selected.values(), key=lambda c: (c.doc_id, c.index))
    for i, ctx in enumerate(ordered, start=1):
        ctx.label = f"C{i}"
    return ordered
