import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_INDEX_NAME
from server.index import IndexManager
from server.models.embedding import create_aliyun_embedding
from src.rag.pipeline import RAGPipeline

CSV_FIELDS = [
    "id",
    "question",
    "answerable",
    "gold_answer",
    "predicted_answer",
    "retrieved_sources",
    "max_score",
    "refused",
    "consistency_label",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"q{line_no:04d}")
            rows.append(row)
    return rows


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_index_if_needed(config: dict):
    if not config.get("retrieval", {}).get("enabled", True):
        return None

    create_aliyun_embedding()
    index_manager = IndexManager(DEFAULT_INDEX_NAME)
    if not index_manager.check_index_exists():
        raise ValueError("No index found. Build the knowledge base before running retrieval configs.")
    return index_manager.load_index()


def run_eval(config_path: Path, eval_set_path: Path, output_path: Path) -> None:
    load_dotenv()
    config = load_config(config_path)
    rows = load_jsonl(eval_set_path)
    index = load_index_if_needed(config)
    pipeline = RAGPipeline(config, index=index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            result = pipeline.answer(row["question"])
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "answerable": row.get("answerable", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "predicted_answer": result.get("answer", ""),
                    "retrieved_sources": json.dumps(
                        result.get("source_details") or result.get("sources", []),
                        ensure_ascii=False,
                    ),
                    "max_score": result.get("max_score", 0.0),
                    "refused": result.get("refused", False),
                    "consistency_label": result.get("consistency_label", "N/A"),
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run ThinkRAG batch evaluation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--eval_set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    run_eval(args.config, args.eval_set, args.output)
    print(f"Wrote evaluation CSV to {args.output}")


if __name__ == "__main__":
    main()
