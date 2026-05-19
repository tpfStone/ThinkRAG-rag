<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Framework: LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-purple.svg)](https://www.llamaindex.ai/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

</div>

# ThinkRAG

ThinkRAG is a RAG question-answering and evaluation system adapted from the original ThinkRAG project. The current workflow is centered on Alibaba Cloud Bailian / DashScope APIs and supports local knowledge-base ingestion, retrieval-augmented answering, C1-C4 comparison experiments, refusal gating, consistency checks, and result reporting.

The UI is built with Streamlit and the indexing/retrieval layer uses LlamaIndex. In development mode, ThinkRAG uses local file storage and persists indexes under `storage/`, so Redis, Elasticsearch, and ChromaDB are not required for the default workflow.

See [docs/deployment.md](docs/deployment.md) for the full deployment, runtime, and evaluation workflow.

## Features

- Local knowledge-base ingestion through Streamlit file upload for PDF, DOCX, PPTX, and other documents.
- API document parsing for PDF pages whose native text extraction is empty or too short.
- DashScope integration through `DASHSCOPE_API_KEY` for generation, embedding, and reranking.
- A configurable `RAGPipeline` that separates retrieval, reranking, strict prompting, refusal gating, and consistency checking.
- C1-C4 experiment configs under `configs/*.yaml` for baseline, basic RAG, enhanced RAG, and the full pipeline.
- Batch evaluation through `eval/eval.py`, using `eval/questions.jsonl` and writing CSV outputs.
- Result summaries and charts through `results/summarize.py` and `results/plot_results.py`.
- Pytest coverage for core logic in `tests/`; unit tests should not require real external API calls.

## Current Models And Config

| Capability | Default |
|---|---|
| Generation model | `qwen-plus` |
| Consistency-check model | `qwen-flash` |
| Embedding | `text-embedding-v4` |
| Reranker | `qwen3-rerank` |
| API key | `DASHSCOPE_API_KEY` |
| Default storage | Local file storage in development mode |

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`, then set your DashScope API key.

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
THINKRAG_PARSE_PROVIDER=auto
THINKRAG_PARSE_CACHE_DIR=storage/parsed
DASHSCOPE_WORKSPACE_ID=
DASHSCOPE_CATEGORY_ID=
```

### 3. Run Tests

```bash
python -m pytest tests -q
```

### 4. Start The App

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Document Parsing Scope

ThinkRAG keeps native text extraction as the primary ingestion path. When complex PDFs, scanned PDFs, or PDFs with poor native extraction have empty or too-short text, `THINKRAG_PARSE_PROVIDER=auto` routes the file to DashScopeParse / Alibaba Cloud Document Mind and indexes the returned text nodes with `parse_provider` metadata.

Parsed results are cached under `storage/parsed/{sha256}.json`, so the same file content is not submitted repeatedly. Full rebuilds write to `storage_staging/` first and only replace the active index after the new index is persisted successfully.

The current document repair path uses Alibaba Cloud APIs and does not require local OCR dependencies.

## Evaluation

Prepare the evaluation set:

```text
eval/questions.jsonl
```

Run the four experiment configs:

```bash
python eval/eval.py --config configs/C1_baseline.yaml --eval_set eval/questions.jsonl --output results/results_C1.csv
python eval/eval.py --config configs/C2_basic.yaml --eval_set eval/questions.jsonl --output results/results_C2.csv
python eval/eval.py --config configs/C3_enhanced.yaml --eval_set eval/questions.jsonl --output results/results_C3.csv
python eval/eval.py --config configs/C4_full.yaml --eval_set eval/questions.jsonl --output results/results_C4.csv
```

Config meanings:

| Config | Meaning |
|---|---|
| C1 | No retrieval; direct LLM answer |
| C2 | Basic RAG retrieval |
| C3 | RAG + reranker + strict prompt |
| C4 | C3 + score-based refusal gate + consistency check |

Generate summaries and charts:

```bash
python results/summarize.py
python results/plot_results.py
```

## Project Structure

```text
ThinkRAG/
├── app.py                 # Streamlit entry point
├── config.py              # Model, storage, and runtime settings
├── configs/               # C1-C4 experiment configs
├── eval/                  # Batch evaluation script and JSONL eval set
├── results/               # Summary and chart scripts
├── src/
│   ├── embeddings/        # Aliyun embedding and cache
│   ├── llm/               # Aliyun client
│   ├── prompts/           # Basic and strict prompts
│   └── rag/               # RAGPipeline, reranker, refusal gate, consistency check
├── server/                # Indexing, retrieval, storage, and document parsing modules
├── frontend/              # Streamlit pages
├── tests/                 # Automated tests
└── docs/                  # Deployment guide, images, and legacy docs
```

## Documentation

- [Deployment and Evaluation Guide](docs/deployment.md)
- [Legacy Hugging Face model download guide](docs/legacy/HowToDownloadModels.md)
- [Legacy Python virtual environment guide](docs/legacy/HowToUsePythonVirtualEnv.md)
- [Security Policy](SECURITY.md)
- [Code of Conduct](docs/Code_of_Conduct.md)

## Legacy Notes

This repository keeps parts of the original ThinkRAG code related to Ollama, Hugging Face local models, and multiple provider configuration for compatibility and reference. The public workflow documented here is the Aliyun / DashScope workflow. Legacy usage notes live under `docs/legacy/`.

## License

ThinkRAG uses the [MIT License](LICENSE).
