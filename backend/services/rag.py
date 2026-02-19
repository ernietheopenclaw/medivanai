"""RAG module — FAISS + sentence-transformers for clinical guideline retrieval in MediVan AI."""
import os
import glob
import logging
from backend.config import MOCK_MODE, EMBEDDING_MODEL, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

_index = None
_chunks = []
_chunk_sources = []  # track which file each chunk came from
_embed_model = None


def _load():
    """Load embedding model and build FAISS index over clinical guidelines."""
    global _index, _chunks, _chunk_sources, _embed_model
    if _index is not None or (MOCK_MODE and _chunks):
        return

    _load_chunks()

    if MOCK_MODE:
        logger.info(f"RAG in mock mode — loaded {len(_chunks)} chunks for keyword matching")
        return

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}'")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)

        if not _chunks:
            logger.warning("No knowledge chunks found — RAG will return empty results")
            return

        logger.info(f"Building FAISS index over {len(_chunks)} chunks...")
        embeddings = _embed_model.encode(_chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        _index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        _index.add(embeddings)
        logger.info(f"FAISS index built: {_index.ntotal} vectors, dim={dim}")

    except ImportError as e:
        logger.warning(f"RAG dependencies not available ({e}), falling back to keyword matching")
    except Exception as e:
        logger.error(f"Failed to build RAG index: {e}", exc_info=True)


def _load_chunks():
    """Load and chunk clinical guideline markdown files."""
    global _chunks, _chunk_sources
    if _chunks:
        return

    if not os.path.isdir(KNOWLEDGE_DIR):
        logger.warning(f"Knowledge directory not found: {KNOWLEDGE_DIR}")
        return

    for fpath in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")
            continue

        # Split by ## headers into sections
        sections = text.split("\n## ")
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) < 30:
                continue
            # Restore ## prefix for non-first sections
            if i > 0:
                section = "## " + section
            _chunks.append(section)
            _chunk_sources.append(fname)

    logger.info(f"Loaded {len(_chunks)} knowledge chunks from {len(set(_chunk_sources))} files")


def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieve top-k relevant clinical guideline chunks for a query."""
    _load()

    if _index is not None and _embed_model is not None:
        return _semantic_retrieve(query, k)
    else:
        return _keyword_retrieve(query, k)


def _semantic_retrieve(query: str, k: int) -> list[str]:
    """FAISS-based semantic retrieval."""
    import numpy as np
    import faiss

    try:
        q_emb = _embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        n = min(k, len(_chunks))
        scores, indices = _index.search(q_emb, n)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(_chunks) and score > 0.1:  # minimum relevance threshold
                results.append(_chunks[idx])

        if not results:
            return _keyword_retrieve(query, k)
        return results

    except Exception as e:
        logger.error(f"Semantic retrieval failed: {e}")
        return _keyword_retrieve(query, k)


def _keyword_retrieve(query: str, k: int = 3) -> list[str]:
    """Keyword-based fallback retrieval."""
    _load_chunks()
    if not _chunks:
        return []

    query_words = set(query.lower().split())
    scored = []
    for chunk in _chunks:
        chunk_lower = chunk.lower()
        # Score by number of matching query words
        score = sum(1 for w in query_words if w in chunk_lower and len(w) > 2)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: -x[0])
    results = [chunk for _, chunk in scored[:k]]

    # If no matches, return first few chunks as general context
    if not results:
        results = _chunks[:min(k, len(_chunks))]
    return results


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "RAG (Clinical Guidelines)", "status": "ready (mock)", "chunks": len(_chunks)}
    return {
        "name": "RAG (Clinical Guidelines)",
        "status": "loaded" if _index is not None else ("keyword-only" if _chunks else "not_loaded"),
        "chunks": len(_chunks),
        "model": EMBEDDING_MODEL,
    }
