"""Best-effort downloader for UN regulations and other public documents.

UNECE (and many .org/.gov sites) block non-browser User-Agents.
This module tries to look like a browser. If it still fails — the caller
should fall back to manual download.
"""

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://unece.org/",
}

DOCUMENT_URLS = {
    "r107": {
        # Прямой URL ты вставишь после того, как один раз скачаешь вручную
        # и посмотришь в DevTools → Network, куда реально идёт запрос.
        "url": "https://unece.org/docbinder/add/425532",
        "filename": "r107r9e.pdf",
        "description": "UN Regulation No. 107 Rev.9 (Addendum 106)",
    },
}


def download_document(
    doc_key: str,
    output_dir: Path = Path("data/raw"),
    force: bool = False,
) -> Path:
    """Try to download a document. Raises RuntimeError if blocked."""
    if doc_key not in DOCUMENT_URLS:
        raise ValueError(f"Unknown document: {doc_key}. Available: {list(DOCUMENT_URLS)}")

    doc = DOCUMENT_URLS[doc_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / doc["filename"]

    if output_path.exists() and not force:
        logger.info("File already exists: %s", output_path)
        return output_path

    logger.info("Downloading %s from %s", doc_key, doc["url"])

    with httpx.Client(
        headers=_BROWSER_HEADERS,
        follow_redirects=True,
        timeout=60.0,
    ) as client:
        response = client.get(doc["url"])
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        is_pdf = "pdf" in content_type or response.content.startswith(b"%PDF")
        if not is_pdf:
            raise RuntimeError(
                f"Expected PDF, got content-type={content_type!r}. "
                f"Скорее всего UNECE блокирует автозагрузку. "
                f"Скачай вручную и положи в {output_path}"
            )

        output_path.write_bytes(response.content)

    logger.info("Saved %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path