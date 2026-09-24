"""Generate embeddings for chunks using sentence-transformers (local)."""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunk import Chunk
from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level cache so the model is loaded only once
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts and return an (N, dim) numpy array.

    Vectors are L2-normalized for cosine similarity.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_chunks(chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
    """
    Embed the text of each chunk.

    Args:
        chunks: List of Chunk objects.
        batch_size: Encoding batch size.

    Returns:
        numpy array of shape (len(chunks), embedding_dim).
    """
    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks...", len(texts))
    embeddings = embed_texts(texts, batch_size=batch_size)
    logger.info("Embedding complete: shape %s", embeddings.shape)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query. Returns a 1-D array of shape (dim,)."""
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return np.asarray(embedding, dtype=np.float32)
