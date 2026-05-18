from src.embeddings.cache import get_cached_embedding, save_embedding


def test_embedding_cache_write_and_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))
    vector = [0.1, 0.2, 0.3]

    assert get_cached_embedding("hello", "model-a") is None
    save_embedding("hello", "model-a", vector)

    assert get_cached_embedding("hello", "model-a") == vector


def test_embedding_cache_uses_text_and_model_in_key(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))
    save_embedding("hello", "model-a", [1.0])
    save_embedding("hello", "model-b", [2.0])
    save_embedding("world", "model-a", [3.0])

    assert get_cached_embedding("hello", "model-a") == [1.0]
    assert get_cached_embedding("hello", "model-b") == [2.0]
    assert get_cached_embedding("world", "model-a") == [3.0]
