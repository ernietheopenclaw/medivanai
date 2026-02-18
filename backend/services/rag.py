"""RAG module — FAISS + sentence-transformers for clinical guideline retrieval."""
import os
import glob
from backend.config import MOCK_MODE, EMBEDDING_MODEL, KNOWLEDGE_DIR

_index = None
_chunks = []
_embed_model = None


def _load():
    global _index, _chunks, _embed_model
    if _index is not None:
        return
    if MOCK_MODE:
        _load_chunks_only()
        return
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    _load_chunks_only()
    if not _chunks:
        return
    embeddings = _embed_model.encode(_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    _index.add(embeddings)


def _load_chunks_only():
    global _chunks
    if _chunks:
        return
    for f in glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")):
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read()
        # Split by sections
        sections = text.split("\n## ")
        for s in sections:
            s = s.strip()
            if len(s) > 50:
                _chunks.append(s)


def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieve top-k relevant guideline chunks."""
    _load()
    if MOCK_MODE or _index is None:
        return _mock_retrieve(query)
    import numpy as np
    q_emb = _embed_model.encode([query], convert_to_numpy=True)
    import faiss
    faiss.normalize_L2(q_emb)
    scores, indices = _index.search(q_emb, min(k, len(_chunks)))
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]


def _mock_retrieve(query: str) -> list[str]:
    _load_chunks_only()
    q = query.lower()
    results = []
    for c in _chunks:
        if any(w in c.lower() for w in q.split()[:3]):
            results.append(c)
        if len(results) >= 3:
            break
    return results[:3] if results else _chunks[:2]
