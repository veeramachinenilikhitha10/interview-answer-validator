from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from ..config import settings
from pathlib import Path

VEC_PATH = Path(settings.DATA_DIR) / 'tfidf_vectorizer.joblib'
MATRIX_PATH = Path(settings.DATA_DIR) / 'doc_matrix.joblib'
DOCS_PATH = Path(settings.DATA_DIR) / 'docs_meta.joblib'

def load_index_and_docs():
    if not VEC_PATH.exists() or not MATRIX_PATH.exists() or not DOCS_PATH.exists():
        return None, None, None
    vec = joblib.load(VEC_PATH)
    doc_matrix = joblib.load(MATRIX_PATH)
    docs = joblib.load(DOCS_PATH)
    return vec, doc_matrix, docs

async def retrieve(query: str, top_k: int = 4):
    vec, doc_matrix, docs = load_index_and_docs()
    if vec is None:
        raise RuntimeError('Index not found. Run POST /ingest to build index.')
    q_vec = vec.transform([query])  # shape (1, n_features)
    sims = cosine_similarity(q_vec, doc_matrix)[0]  # shape (n_docs,)
    # get top_k indices (descending order)
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idxs:
        results.append({'id': docs[i]['id'], 'score': float(sims[i]), 'text': docs[i]['text'], 'meta': docs[i]['meta']})
    return results
