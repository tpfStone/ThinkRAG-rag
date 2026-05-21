import importlib

import pandas as pd


def _write_result(path):
    pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "answerable",
                "answerable": True,
                "predicted_answer": "a",
                "refused": False,
                "consistency_label": "Y",
                "correct": True,
                "citation_ok": True,
                "has_hallucination": False,
            },
            {
                "id": "q2",
                "question": "unanswerable",
                "answerable": False,
                "predicted_answer": "refused",
                "refused": True,
                "consistency_label": "N",
                "correct": True,
                "citation_ok": False,
                "has_hallucination": False,
            },
        ]
    ).to_csv(path, index=False)


def test_summarize_csv_reports_refusal_accuracy(tmp_path):
    summarize = importlib.import_module("results.summarize")
    path = tmp_path / "results_C4.csv"
    _write_result(path)

    summary = summarize.summarize_csv(path)

    assert summary["count"] == 2
    assert summary["refusal_accuracy"] == 1.0
    assert summary["correct"] == 1.0


def test_plot_results_generates_three_charts(tmp_path):
    importlib.import_module("matplotlib")
    plot_results = importlib.import_module("results.plot_results")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for cfg in ["C1", "C2", "C3", "C4"]:
        _write_result(results_dir / f"results_{cfg}.csv")

    plot_results.main(results_dir)

    charts_dir = results_dir / "charts"
    assert (charts_dir / "core_metrics.png").exists()
    assert (charts_dir / "refusal_accuracy.png").exists()
    assert (charts_dir / "consistency_distribution.png").exists()


def test_plot_results_skips_core_metrics_without_judged_columns(tmp_path):
    plot_results = importlib.import_module("results.plot_results")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame(
        [
            {
                "id": "q1",
                "answerable": True,
                "refused": False,
                "consistency_label": "N/A",
            },
            {
                "id": "q2",
                "answerable": False,
                "refused": True,
                "consistency_label": "N/A",
            },
        ]
    ).to_csv(results_dir / "results_C1.csv", index=False)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir()
    stale_core_metrics = charts_dir / "core_metrics.png"
    stale_core_metrics.write_text("stale", encoding="utf-8")

    summary = plot_results.load_summaries(results_dir)
    c1 = summary[summary["config"] == "C1"].iloc[0]

    assert c1["refusal_accuracy"] == 1.0
    assert pd.isna(c1["correct_rate"])
    assert pd.isna(c1["citation_ok_rate"])
    assert pd.isna(c1["hallucination_free_rate"])

    plot_results.main(results_dir)

    assert not stale_core_metrics.exists()
    assert (charts_dir / "refusal_accuracy.png").exists()
    assert (charts_dir / "consistency_distribution.png").exists()
