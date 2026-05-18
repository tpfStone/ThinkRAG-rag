from llama_index.core.embeddings import BaseEmbedding

from config import ALIYUN_EMBEDDING_DIMENSIONS, ALIYUN_EMBEDDING_MODEL
from src.embeddings.cache import get_cached_embedding, save_embedding
from src.llm.aliyun_client import get_aliyun_client


class CachedAliyunEmbedding(BaseEmbedding):
    model_name: str = ALIYUN_EMBEDDING_MODEL
    dimensions: int = ALIYUN_EMBEDDING_DIMENSIONS

    def _get_text_embedding(self, text: str) -> list[float]:
        cached = get_cached_embedding(text, self.model_name)
        if cached is not None:
            return cached

        client = get_aliyun_client()
        resp = client.embeddings.create(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
        )
        vector = resp.data[0].embedding
        save_embedding(text, self.model_name, vector)
        return vector

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
