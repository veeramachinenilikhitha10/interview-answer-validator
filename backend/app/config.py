from pathlib import Path

class _SimpleSettings:
    # data directory (relative to repo root)
    DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
    EMBED_MODEL = "all-MiniLM-L6-v2"
    OPENAI_API_KEY = None
    FAISS_INDEX_PATH = str(Path(DATA_DIR) / "index.faiss")
    DOCS_META_PATH = str(Path(DATA_DIR) / "docs.pkl")

settings = _SimpleSettings()
