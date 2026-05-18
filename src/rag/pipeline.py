from __future__ import annotations

from typing import Any

from server.retriever import SimpleFusionRetriever
from src.llm.aliyun_client import get_aliyun_client
from src.prompts.basic_prompt import BASIC_DIRECT_PROMPT, BASIC_RAG_PROMPT
from src.prompts.strict_prompt import STRICT_RAG_PROMPT
from src.rag.consistency_check import verify_consistency
from src.rag.reranker import aliyun_rerank
from src.rag.score_gating import should_refuse


DEFAULT_CONFIG = {
    "retrieval": {"enabled": True, "top_k": 5, "use_reranker": False, "initial_top_k": 20},
    "prompt": {"strict_mode": False},
    "refuse_gate": {"enabled": False, "max_threshold": 0.5, "spread_threshold": 0.05},
    "consistency_check": {"enabled": False},
    "llm": {"model": "qwen-plus", "temperature": 0},
}


class RAGPipeline:
    def __init__(self, config: dict[str, Any], index=None, client=None):
        self.config = self._merge_config(config)
        self.index = index
        self.client = client or get_aliyun_client()

    def set_index(self, index) -> None:
        self.index = index

    def answer(self, question: str) -> dict[str, Any]:
        result = {
            "answer": "",
            "sources": [],
            "source_details": [],
            "max_score": 0.0,
            "refused": False,
            "refusal_reason": "",
            "consistency_label": "N/A",
            "consistency_reason": "",
        }

        retrieval_cfg = self.config["retrieval"]
        if not retrieval_cfg["enabled"]:
            result["answer"] = self._llm_call(question, context=None)
            return result

        nodes = self._retrieve(question, self._initial_top_k())
        scores = [self._node_score(node) for node in nodes]

        if retrieval_cfg["use_reranker"] and nodes:
            ranked = aliyun_rerank(
                question,
                [self._node_text(node) for node in nodes],
                retrieval_cfg["top_k"],
            )
            filtered = [(idx, score) for idx, score in ranked if 0 <= idx < len(nodes)]
            nodes = [nodes[idx] for idx, _ in filtered]
            scores = [score for _, score in filtered]

        result["max_score"] = max(scores) if scores else 0.0
        result["sources"] = [self._node_file_name(node) for node in nodes]
        result["source_details"] = [self._source_detail(node, score) for node, score in zip(nodes, scores)]

        if self.config["refuse_gate"]["enabled"]:
            refused, reason = should_refuse(scores, **self.config["refuse_gate"])
            if refused:
                result["refused"] = True
                result["refusal_reason"] = reason
                result["answer"] = "根据现有知识库，未找到相关信息。"
                return result

        context = self._format_context(nodes, scores)
        result["answer"] = self._llm_call(
            question,
            context=context,
            strict=self.config["prompt"]["strict_mode"],
        )

        if self.config["consistency_check"]["enabled"]:
            label, reason = verify_consistency(result["answer"], context, self.client)
            result["consistency_label"] = label
            result["consistency_reason"] = reason

        return result

    def _retrieve(self, question: str, top_k: int):
        if self.index is None:
            raise ValueError("RAGPipeline requires an index when retrieval is enabled.")
        retriever = SimpleFusionRetriever(vector_index=self.index, top_k=top_k)
        return retriever.retrieve(question)

    def _format_context(self, nodes, scores: list[float] | None = None) -> str:
        scores = scores or [self._node_score(node) for node in nodes]
        chunks = []
        for idx, (node, score) in enumerate(zip(nodes, scores), start=1):
            metadata = self._node_metadata(node)
            file_name = metadata.get("file_name") or metadata.get("title") or "N/A"
            page = metadata.get("page_label", "N/A")
            chunks.append(
                f"[{idx}] file={file_name}; page={page}; score={score:.4f}\n"
                f"{self._node_text(node)}"
            )
        return "\n\n".join(chunks)

    def _llm_call(self, question: str, context: str | None, strict: bool = False) -> str:
        if context is None:
            prompt = BASIC_DIRECT_PROMPT.format(question=question)
        elif strict:
            prompt = STRICT_RAG_PROMPT.format(context=context, question=question)
        else:
            prompt = BASIC_RAG_PROMPT.format(context=context, question=question)

        llm_cfg = self.config["llm"]
        response = self.client.chat.completions.create(
            model=llm_cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_cfg["temperature"],
        )
        return response.choices[0].message.content.strip()

    def _initial_top_k(self) -> int:
        retrieval_cfg = self.config["retrieval"]
        if retrieval_cfg["use_reranker"]:
            return int(retrieval_cfg.get("initial_top_k", 20))
        return int(retrieval_cfg["top_k"])

    def _source_detail(self, node, score: float) -> dict[str, Any]:
        metadata = self._node_metadata(node)
        text = self._node_text(node)
        return {
            "file": metadata.get("file_name") or metadata.get("title") or "N/A",
            "page": metadata.get("page_label", "N/A"),
            "score": float(score) if score is not None else 0.0,
            "text": text[:300],
            "node_id": getattr(getattr(node, "node", node), "node_id", ""),
        }

    def _node_score(self, node) -> float:
        score = getattr(node, "score", 0.0)
        return float(score) if score is not None else 0.0

    def _node_text(self, node) -> str:
        source_node = getattr(node, "node", node)
        return getattr(source_node, "text", "") or ""

    def _node_metadata(self, node) -> dict[str, Any]:
        source_node = getattr(node, "node", node)
        metadata = getattr(source_node, "metadata", {}) or {}
        return metadata if isinstance(metadata, dict) else {}

    def _node_file_name(self, node) -> str:
        metadata = self._node_metadata(node)
        return metadata.get("file_name") or metadata.get("title") or "N/A"

    def _merge_config(self, config: dict[str, Any]) -> dict[str, Any]:
        merged = {}
        for key, default_value in DEFAULT_CONFIG.items():
            current_value = config.get(key, {}) if config else {}
            if isinstance(default_value, dict):
                merged[key] = {**default_value, **current_value}
            else:
                merged[key] = current_value if current_value is not None else default_value
        return merged
