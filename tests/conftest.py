import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def client():
    """In-process Qdrant (local mode): same client API as the server, no Docker needed."""
    c = QdrantClient(":memory:")
    yield c
    c.close()
