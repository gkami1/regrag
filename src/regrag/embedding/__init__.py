from regrag.embedding.models import (
    Embedder,
    EmbedderConfig,
    Embedding,
    LazyEmbedder,
    SparseVector,
)

__all__ = ["Embedder", "EmbedderConfig", "Embedding", "LazyEmbedder", "SparseVector"]

# BGEM3Embedder is imported from regrag.embedding.bge_m3 explicitly: it pulls in
# torch and transformers, which code that only needs the protocol should not pay for.
