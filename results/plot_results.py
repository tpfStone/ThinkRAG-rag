from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

CONFIGS = ["C1", "C2", "C3", "C4"]


def _as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _safe_mean(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return float("nan")
    return float(_as_bool(df[column]).mean())


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    answerable = _as_bool(df["answerable"]) if "answerable" in df.columns else pd.Series([], dtype=bool)
    refused = _as_bool(df["refused"]) if "refused" in df.columns else pd.Series([], dtype=bool)
    unanswerable_mask = ~answerable if len(answerable) else pd.Series([], dtype=bool)
    refusal_accuracy = float(refused[unanswerable_mask].mean()) if len(refused[unanswerable_mask]) else 0.0
    return {
        "correct_rate": _safe_mean(df, "correct"),
        "citation_ok_rate": _safe_mean(df, "citation_ok"),
        "hallucination_free_rate": (
            1.0 - _safe_mean(df, "has_hallucination")
            if "has_hallucination" in df.columns
            else float("nan")
        ),
        "refusal_accuracy": refusal_accuracy,
    }


def load_summaries(results_dir: Path) -> pd.DataFrame:
    rows = []
    for cfg in CONFIGS:
        path = results_dir / f"results_{cfg}.csv"
        if path.exists():
            row = {"config": cfg, **summarize_file(path)}
        else:
            row = {
                "config": cfg,
                "correct_rate": float("nan"),
                "citation_ok_rate": float("nan"),
                "hallucination_free_rate": float("nan"),
                "refusal_accuracy": float("nan"),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_core_metrics(summary: pd.DataFrame, charts_dir: Path) -> None:
    output_path = charts_dir / "core_metrics.png"
    metric_columns = [
        "correct_rate",
        "citation_ok_rate",
        "hallucination_free_rate",
    ]
    available_columns = [
        column for column in metric_columns if column in summary and summary[column].notna().any()
    ]
    if available_columns:
        ax = summary.set_index("config")[available_columns].plot(kind="bar", ylim=(0, 1), figsize=(10, 5))
        ax.set_ylabel("Rate")
        ax.set_title("Core Evaluation Metrics")
        ax.legend(loc="lower right")
    else:
        output_path.unlink(missing_ok=True)
        print("Skipped core metrics chart: judged metric columns are not available.")
        return
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_refusal_accuracy(summary: pd.DataFrame, charts_dir: Path) -> None:
    ax = summary.plot(x="config", y="refusal_accuracy", kind="bar", ylim=(0, 1), legend=False)
    ax.set_ylabel("Refusal Accuracy")
    ax.set_title("Unanswerable Question Refusal Accuracy")
    plt.tight_layout()
    plt.savefig(charts_dir / "refusal_accuracy.png", dpi=200)
    plt.close()


def plot_consistency_distribution(results_dir: Path, charts_dir: Path) -> None:
    path = results_dir / "results_C4.csv"
    if not path.exists():
        labels = pd.Series({"N/A": 1})
    else:
        df = pd.read_csv(path)
        labels = df.get("consistency_label", pd.Series(["N/A"])).fillna("N/A").value_counts()
    ax = labels.plot(kind="pie", autopct="%1.1f%%", figsize=(5, 5))
    ax.set_ylabel("")
    ax.set_title("C4 Consistency Label Distribution")
    plt.tight_layout()
    plt.savefig(charts_dir / "consistency_distribution.png", dpi=200)
    plt.close()


def main(results_dir: Path = Path("results")) -> None:
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summaries(results_dir)
    plot_core_metrics(summary, charts_dir)
    plot_refusal_accuracy(summary, charts_dir)
    plot_consistency_distribution(results_dir, charts_dir)
    print(f"Wrote charts to {charts_dir}")


if __name__ == "__main__":
    main()
