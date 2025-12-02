import pickle
from pathlib import Path
import faiss
from ..config import settings

def load_index_and_docs():
    idx_path = Path(settings.FAISS_INDEX_PATH)
    docs_path = Path(settings.DOCS_META_PATH)
    if not idx_path.exists() or not docs_path.exists():
        return None, None
    idx = faiss.read_index(str(idx_path))
    with open(docs_path, 'rb') as f:
        docs = pickle.load(f)
    return idx, docs

def faiss_search(idx, docs, q_emb, top_k=4):
    D, I = idx.search(q_emb, top_k)
    results = []
    for score, i in zip(D[0], I[0]):
        if i < 0 or i >= len(docs):
            continue
        results.append({'id': docs[i]['id'], 'score': float(score), 'text': docs[i]['text'], 'meta': docs[i]['meta']})
    return results
