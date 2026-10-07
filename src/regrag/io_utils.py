"""File helpers shared by every pipeline stage that persists its output."""

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Fingerprint of the file's bytes, read in 1 MB chunks to keep memory flat."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_lines_atomic(path: Path, lines: Iterable[str]) -> None:
    """Write lines to a temp file, then atomically rename it over `path`.

    Readers see either the old file or the complete new one, never a partial write.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    os.replace(tmp, path)  # atomic on the same filesystem, on Windows and POSIX
