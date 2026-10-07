"""Structure-aware recursive chunking.

Recursion runs over the *section tree*, not over characters:

    chunk(section):
        whole section fits max_tokens        -> one chunk
        otherwise: its own text, then each child (recursively chunked),
                   packed greedily into chunks of ~target_tokens
        a single block bigger than max       -> split into sentences (last resort)

Packing merges *consecutive* units of one parent, so a chunk never starts in the
middle of a paragraph, never crosses a scope, and covers a contiguous run of
sections (e.g. 7.6.1.1-7.6.1.4).

Every size is measured on the final `embed_text`, i.e. *including* the
breadcrumb header, with the embedding model's tokenizer, so `max_tokens` is a
real guarantee rather than an estimate.
"""

import logging
import re
import statistics

from regrag.chunking.models import Chunk, ChunkerConfig, ChunkingReport
from regrag.chunking.tokens import TokenCounter
from regrag.chunking.tree import Piece, SectionNode, build_forest
from regrag.ingestion.models import ParsedDocument

logger = logging.getLogger(__name__)

# Sentence boundary: after . ; : followed by whitespace. Regulation sentences end
# with these; list items often end with ";".
_SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")

Unit = list[Piece]  # pieces that must end up in the same chunk


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    prefix = paths[0]
    for p in paths[1:]:
        n = 0
        while n < min(len(prefix), len(p)) and prefix[n] == p[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


class Chunker:
    def __init__(self, config: ChunkerConfig, counter: TokenCounter) -> None:
        self.cfg = config
        self.counter = counter
        self._split_blocks = 0

    # --- rendering ---------------------------------------------------------

    def _header(self, doc_id: str, path: tuple[str, ...]) -> str:
        head, *sections = path
        scope_id, sep, title = head.partition(": ")
        words = title.split()
        if len(words) > self.cfg.max_scope_title_words:
            title = " ".join(words[: self.cfg.max_scope_title_words]) + "…"
        head = f"{scope_id}{sep}{title}"
        return " > ".join([f"{doc_id} | {head}", *sections])

    def _render(self, doc_id: str, unit: Unit) -> tuple[str, str, str, tuple[str, ...]]:
        path = _common_prefix([p.path for p in unit])
        header = self._header(doc_id, path)
        body = "\n".join(p.text for p in unit)
        return header, body, f"{header}\n\n{body}", path

    def _tokens(self, doc_id: str, unit: Unit) -> int:
        return self.counter.count(self._render(doc_id, unit)[2])

    # --- splitting / packing -------------------------------------------------

    def _split_piece(self, doc_id: str, piece: Piece) -> list[Unit]:
        """A single block over max_tokens: sentences, and as a last resort word windows."""
        if self._tokens(doc_id, [piece]) <= self.cfg.max_tokens:
            return [[piece]]
        self._split_blocks += 1
        units: list[Unit] = []
        for sentence in _SENTENCE_END_RE.split(piece.text):
            part = Piece(block=piece.block, text=sentence, path=piece.path)
            if self._tokens(doc_id, [part]) <= self.cfg.max_tokens:
                units.append([part])
                continue
            # A "sentence" this long is usually a flattened table: cut by words.
            window: list[str] = []
            for word in sentence.split():
                candidate = Piece(piece.block, " ".join([*window, word]), piece.path)
                if window and self._tokens(doc_id, [candidate]) > self.cfg.max_tokens:
                    units.append([Piece(piece.block, " ".join(window), piece.path)])
                    window = []
                window.append(word)
            if window:
                units.append([Piece(piece.block, " ".join(window), piece.path)])
        return units

    def _pack(self, doc_id: str, units: list[Unit]) -> list[Unit]:
        """Greedily merge consecutive units.

        Merge while the result stays within target_tokens; additionally, a unit
        smaller than min_tokens may be merged up to max_tokens, so headings and
        one-line paragraphs don't end up as chunks of their own.
        """
        packed: list[Unit] = []
        current: Unit | None = None
        for unit in units:
            if current is None:
                current = unit
                continue
            merged = current + unit
            size = self._tokens(doc_id, merged)
            small = min(self._tokens(doc_id, current), self._tokens(doc_id, unit))
            if size <= self.cfg.target_tokens or (
                size <= self.cfg.max_tokens and small < self.cfg.min_tokens
            ):
                current = merged
            else:
                packed.append(current)
                current = unit
        if current is not None:
            packed.append(current)
        return packed

    def _chunk_node(self, doc_id: str, node: SectionNode) -> list[Unit]:
        whole = list(node.all_pieces())
        if self._tokens(doc_id, whole) <= self.cfg.max_tokens:
            return [whole]
        units: list[Unit] = []
        for piece in node.pieces:
            units.extend(self._split_piece(doc_id, piece))
        for child in node.children:
            units.extend(self._chunk_node(doc_id, child))
        return self._pack(doc_id, units)

    # --- public API ------------------------------------------------------------

    def chunk_document(self, doc: ParsedDocument) -> tuple[list[Chunk], ChunkingReport]:
        self._split_blocks = 0
        body = doc.body_blocks()
        included = [b for b in body if b.scope not in self.cfg.exclude_scopes]

        units: list[Unit] = []
        for root in build_forest(included):
            units.extend(self._chunk_node(doc.doc_id, root))

        chunks = [self._to_chunk(doc.doc_id, i, unit) for i, unit in enumerate(units)]
        report = self._report(chunks, body, included)
        logger.info(
            "Chunked %s: %d chunks, tokens median=%d p90=%d max=%d",
            doc.doc_id,
            report.chunk_count,
            report.tokens_median,
            report.tokens_p90,
            report.tokens_max,
        )
        return chunks, report

    def _to_chunk(self, doc_id: str, index: int, unit: Unit) -> Chunk:
        header, body, embed_text, path = self._render(doc_id, unit)
        blocks = [p.block for p in unit]
        block_ids: list[str] = []
        sections: list[str] = []
        for b in blocks:
            if not block_ids or block_ids[-1] != b.id:  # a split block repeats per sentence
                block_ids.append(b.id or "")
            if b.section_number and b.section_number not in sections:
                sections.append(b.section_number)
        return Chunk(
            id=f"{doc_id}:c{index:05d}",
            doc_id=doc_id,
            index=index,
            embed_text=embed_text,
            header=header,
            text=body,
            token_count=self.counter.count(embed_text),
            scope=blocks[0].scope or "",
            section_path=list(path),
            sections=sections,
            page_start=min(b.page_number for b in blocks),
            page_end=max(b.page_number for b in blocks),
            block_ids=block_ids,
        )

    def _report(self, chunks: list[Chunk], body: list, included: list) -> ChunkingReport:
        sizes = sorted(c.token_count for c in chunks) or [0]
        return ChunkingReport(
            chunk_count=len(chunks),
            input_blocks=len(included),
            excluded_blocks=len(body) - len(included),
            split_blocks=self._split_blocks,
            below_min=sum(1 for s in sizes if s < self.cfg.min_tokens) if chunks else 0,
            tokens_min=sizes[0],
            tokens_median=int(statistics.median(sizes)),
            tokens_p90=sizes[min(len(sizes) - 1, int(0.9 * len(sizes)))],
            tokens_max=sizes[-1],
        )
