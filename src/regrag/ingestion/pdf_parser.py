"""PDF parsing with structure preservation and document filters."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # было fitz — убирает deprecation warning

logger = logging.getLogger(__name__)


# --- Filters ---

# Running header: E/ECE/324/Rev.3/Add.106/Rev.9 (повторяется на каждой странице)
_RUNNING_HEADER_RE = re.compile(
    r"E/ECE/(?:324|TRANS/505)/Rev\.\d+/Add\.\d+/Rev\.\d+",
    re.IGNORECASE,
)

# ToC entry: текст + минимум 4 точки подряд + номер страницы в конце
_TOC_RE = re.compile(r"\.{4,}\s*\d+\s*$")

# Нумерация: "1.", "1.1.", "3.10.12."
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+\S")

# Annex heading: "Annex 12", "Annex 1 - Part 1 - Appendix 2"
_ANNEX_RE = re.compile(r"^\s*Annex\s+\d+", re.IGNORECASE)


@dataclass
class ParsedBlock:
    text: str
    page_number: int
    block_type: str  # "text" | "title" | "subtitle"
    font_size: float = 0.0
    is_bold: bool = False
    section_number: str | None = None  # "3.10.12" или None


@dataclass
class ParsedDocument:
    source_path: str
    total_pages: int
    blocks: list[ParsedBlock] = field(default_factory=list)

    def to_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def _normalize(text: str) -> str:
    """Схлопнуть whitespace, убрать пустые строки, соединить многострочники."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _is_running_header(text: str) -> bool:
    return bool(_RUNNING_HEADER_RE.search(text)) and len(text) < 250


def _is_toc_entry(text: str) -> bool:
    return bool(_TOC_RE.search(text))


def _extract_section_number(text: str) -> str | None:
    m = _NUMBERED_RE.match(text)
    return m.group(1) if m else None


def _block_font_info(block: dict) -> tuple[float, bool]:
    sizes: list[float] = []
    bold_count = 0
    total = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(span.get("size", 0))
            total += 1
            # PyMuPDF flags: bit 4 (значение 16) = bold
            if span.get("flags", 0) & 16:
                bold_count += 1
    avg = sum(sizes) / len(sizes) if sizes else 0.0
    is_bold = (bold_count / total > 0.5) if total else False
    return avg, is_bold


def _classify(text: str, avg_size: float, is_bold: bool) -> tuple[str, str | None]:
    section_number = _extract_section_number(text)

    # 1. Крупный болд — заголовок верхнего уровня
    if avg_size >= 13 and is_bold:
        return "title", section_number
    # 2. Средний болд — подзаголовок
    if avg_size >= 12 and is_bold:
        return "subtitle", section_number
    # 3. Нумерованный болд и короткий — заголовок раздела ("1. Scope")
    if section_number and is_bold and len(text) < 120:
        return "title", section_number
    # 4. Annex-заголовок без нумерации
    if _ANNEX_RE.match(text) and len(text) < 150:
        return "title", None
    # 5. Нумерованный, короткий, без точки в конце — похоже на заголовок
    #    (напр. "2.2. Definition of type(s)", но не "1.2.3. Off-road vehicles.")
    if (
        section_number
        and len(text) < 60
        and not text.rstrip().endswith(".")
    ):
        return "subtitle", section_number
    # 6. Всё остальное — тело с номером (важно: section_number сохраняем!)
    return "text", section_number


def parse_pdf(path: Path, min_block_chars: int = 5) -> ParsedDocument:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = pymupdf.open(path)
    parsed = ParsedDocument(source_path=str(path), total_pages=len(doc))

    logger.info("Parsing %s (%d pages)", path.name, len(doc))

    skipped_headers = skipped_toc = skipped_short = 0

    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            raw = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw += span.get("text", "")
                raw += "\n"

            text = _normalize(raw)
            if len(text) < min_block_chars:
                skipped_short += 1
                continue
            if _is_running_header(text):
                skipped_headers += 1
                continue
            if _is_toc_entry(text):
                skipped_toc += 1
                continue

            avg_size, is_bold = _block_font_info(block)
            block_type, section_number = _classify(text, avg_size, is_bold)

            parsed.blocks.append(
                ParsedBlock(
                    text=text,
                    page_number=page_num,
                    block_type=block_type,
                    font_size=avg_size,
                    is_bold=is_bold,
                    section_number=section_number,
                )
            )

    doc.close()

    logger.info(
        "Parsed %d blocks (skipped: headers=%d, toc=%d, short=%d)",
        len(parsed.blocks), skipped_headers, skipped_toc, skipped_short,
    )
    return parsed