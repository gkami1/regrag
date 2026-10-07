"""Token counting with the embedding model's own tokenizer.

The chunker only needs "how many tokens is this text?", so it depends on the
small `TokenCounter` protocol rather than on a concrete tokenizer. Production
uses the real bge-m3 tokenizer; tests pass a trivial word counter, so they run
offline, instantly and with easy-to-reason-about numbers.
"""

from functools import lru_cache
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HFTokenCounter:
    """Counts tokens with a Hugging Face `tokenizers` tokenizer (fast, Rust-based).

    `from_pretrained` downloads tokenizer.json once into the Hugging Face cache
    (~/.cache/huggingface); later runs load it from disk.
    """

    def __init__(self, name: str) -> None:
        from tokenizers import Tokenizer  # imported lazily: tests never need it

        self.name = name
        self._tokenizer = Tokenizer.from_pretrained(name)
        # The chunker re-measures the same strings while packing; cache them.
        self.count = lru_cache(maxsize=65_536)(self._count)

    def _count(self, text: str) -> int:
        # Special tokens (<s> ... </s>) are part of every model input, so they
        # count against the budget too.
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)


class WordCounter:
    """One token per whitespace-separated word. For tests only."""

    def count(self, text: str) -> int:
        return len(text.split())
