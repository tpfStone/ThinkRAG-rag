import hashlib
import os
import pickle
from pathlib import Path


def get_cache_dir() -> Path:
    return Path(os.getenv("EMBEDDING_CACHE_DIR", ".embedding_cache"))


def _key(text: str, model: str) -> str:
    return hashlib.md5(f"{model}|{text}".encode("utf-8")).hexdigest()


def get_cached_embedding(text: str, model: str):
    path = get_cache_dir() / f"{_key(text, model)}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return None


def save_embedding(text: str, model: str, vector) -> None:
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_key(text, model)}.pkl"
    with path.open("wb") as f:
        pickle.dump(vector, f)
