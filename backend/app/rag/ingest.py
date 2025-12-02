from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ..config import settings

DATA_DIR = Path(settings.DATA_DIR)

VEC_PATH = Path(settings.DATA_DIR) / 'tfidf_vectorizer.joblib'
MATRIX_PATH = Path(settings.DATA_DIR) / 'doc_matrix.joblib'
DOCS_PATH = Path(settings.DATA_DIR) / 'docs_meta.joblib'

def load_corpus():
    docs = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {'.txt', '.md'}:
            text = p.read_text(encoding='utf-8', errors='ignore')
            docs.append({'id': p.name, 'text': text, 'meta': {'path': str(p)}})
    return docs

def build_index():
    docs = load_corpus()
    if not docs:
        raise RuntimeError('No documents found in data/. Add .txt or .md files.')
    texts = [d['text'] for d in docs]
    # Create TF-IDF vectorizer
    vec = TfidfVectorizer(stop_words='english', max_features=20000)
    doc_matrix = vec.fit_transform(texts)  # shape (n_docs, n_features)
    # Save artifacts
    VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(doc_matrix, MATRIX_PATH)
    joblib.dump(docs, DOCS_PATH)
    return {'indexed': len(docs)}

def load_index():
    if not VEC_PATH.exists() or not MATRIX_PATH.exists() or not DOCS_PATH.exists():
        return None, None, None
    vec = joblib.load(VEC_PATH)
    doc_matrix = joblib.load(MATRIX_PATH)
    docs = joblib.load(DOCS_PATH)
    return vec, doc_matrix, docs
