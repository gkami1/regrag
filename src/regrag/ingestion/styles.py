"""Style analysis: rank font styles into heading levels, relative to this document.

"Big" only means something relative to the document's own body text: 14pt is a
heading in a 10pt document and body text in a 14pt one. So we build a style
inventory (character counts per (size, bold)), take the dominant style as the
body baseline, and treat a style as a heading style when it is:

- more prominent than body: at least `min_size_delta` larger, or same size but bold;
- rare: a small share of all characters (otherwise it is just a second body style);
- substantial: enough characters in total, so a handful of formula or diagram
  fragments set in some odd size don't become a heading level.

Heading styles are ranked by (size, bold), largest first: level 1, 2, 3, ...
Fonts only identify the top few heading levels; deeper headings in regulations
are set in body font and are recognised by numbering in `structure.py`.
"""

import logging
from collections import Counter

from pydantic import ConfigDict

from regrag.ingestion.models import Model, ParsedBlock, StyleReport

logger = logging.getLogger(__name__)

Style = tuple[float, bool]  # (size rounded to 0.5pt, bold)


class StyleConfig(Model):
    model_config = ConfigDict(frozen=True)

    min_size_delta: float = 1.0  # pt larger than body to count as "larger"
    max_char_share: float = 0.05  # heading styles cover at most this share of text
    min_style_chars: int = 30  # ...and at least this many characters in total


def style_of(block: ParsedBlock) -> Style:
    # Rounding to 0.5pt absorbs rendering jitter (13.9 vs 14.0).
    return round(block.font_size * 2) / 2, block.is_bold


def style_name(style: Style) -> str:
    size, bold = style
    return f"{size:.1f} {'bold' if bold else 'regular'}"


def _is_heading_style(style: Style, body: Style, chars: int, total: int, cfg: StyleConfig) -> bool:
    size, bold = style
    body_size, body_bold = body
    larger = size >= body_size + cfg.min_size_delta
    emphasized = abs(size - body_size) < cfg.min_size_delta and bold and not body_bold
    return (
        (larger or emphasized)
        and chars <= cfg.max_char_share * total
        and chars >= cfg.min_style_chars
    )


def annotate_styles(blocks: list[ParsedBlock], config: StyleConfig | None = None) -> StyleReport:
    """Set `heading_level` on blocks set in a heading style. Expects body blocks only."""
    cfg = config or StyleConfig()
    inventory: Counter[Style] = Counter()
    for b in blocks:
        inventory[style_of(b)] += len(b.text)
    if not inventory:
        return StyleReport(body_style="none")

    total = sum(inventory.values())
    body = inventory.most_common(1)[0][0]
    heading_styles = [
        s for s, chars in inventory.items() if _is_heading_style(s, body, chars, total, cfg)
    ]
    # Largest first; at equal size, bold outranks regular.
    heading_styles.sort(key=lambda s: (s[0], s[1]), reverse=True)
    levels = {s: i for i, s in enumerate(heading_styles, start=1)}

    for b in blocks:
        b.heading_level = levels.get(style_of(b))

    report = StyleReport(
        body_style=style_name(body),
        heading_levels={style_name(s): lvl for s, lvl in levels.items()},
    )
    logger.info("Styles: body=%s headings=%s", report.body_style, report.heading_levels)
    return report
