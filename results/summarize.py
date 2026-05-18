from pathlib import Path

import pandas as pd


def _as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def summarize_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    answerable = _as_bool(df["answerable"]) if "answerable" in df.columns else pd.Series([], dtype=bool)
    refused = _as_bool(df["refused"]) if "refused" in df.columns else pd.Series([], dtype=bool)
    unanswerable_mask = ~answerable if len(answerable) else pd.Series([], dtype=bool)

    summary = {
        "file": path.name,
        "count": len(df),
        "refusal_accuracy": float(refused[unanswerable_mask].mean()) if len(refused[unanswerable_mask]) else None,
    }
    for column in ["correct", "citation_ok", "has_hallucination"]:
        if column in df.columns:
            summary[column] = float(_as_bool(df[column]).mean())
    return summary


def summarize_results(results_dir: Path = Path("results")) -> pd.DataFrame:
    rows = [summarize_csv(path) for path in sorted(results_dir.glob("results_C*.csv"))]
    return pd.DataFrame(rows)


def main() -> None:
    summary = summarize_results()
    if summary.empty:
        print("No results_C*.csv files found.")
        return
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
