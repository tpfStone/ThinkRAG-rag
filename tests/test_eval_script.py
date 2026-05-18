import csv
import importlib


def test_run_eval_writes_expected_csv(tmp_path, monkeypatch):
    eval_module = importlib.import_module("eval.eval")

    class FakePipeline:
        def __init__(self, config, index=None):
            pass

        def answer(self, question):
            return {
                "answer": "predicted",
                "source_details": [{"file": "doc.pdf", "score": 0.9}],
                "max_score": 0.9,
                "refused": False,
                "consistency_label": "Y",
            }

    config_path = tmp_path / "config.yaml"
    config_path.write_text("retrieval:\n  enabled: false\n", encoding="utf-8")
    eval_set_path = tmp_path / "questions.jsonl"
    eval_set_path.write_text(
        '{"id":"q1","question":"question?","answerable":true,"gold_answer":"gold"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "out.csv"

    monkeypatch.setattr(eval_module, "RAGPipeline", FakePipeline)
    monkeypatch.setattr(eval_module, "load_index_if_needed", lambda config: None)

    eval_module.run_eval(config_path, eval_set_path, output_path)

    with output_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["id"] == "q1"
    assert rows[0]["predicted_answer"] == "predicted"
    assert rows[0]["consistency_label"] == "Y"
    assert "retrieved_sources" in rows[0]
