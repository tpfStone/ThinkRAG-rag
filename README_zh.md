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

ThinkRAG 是一个基于原 ThinkRAG 改造的 RAG 问答与评测系统，当前主线使用阿里云百炼 DashScope API，支持本地知识库导入、检索增强问答、C1-C4 对比实验、拒答门控、一致性验证和结果统计出图。

项目前端沿用 Streamlit，索引和检索能力基于 LlamaIndex。默认开发模式使用本地文件存储，知识库索引写入 `storage/`，不需要额外启动 Redis、Elasticsearch 或 ChromaDB 服务。

完整部署、运行和评测流程见 [docs/deployment.md](docs/deployment.md)。

## 主要功能

- 本地知识库导入：通过 Streamlit 上传 PDF、DOCX、PPTX 等文档，并生成本地索引。
- API 文档解析：当 PDF 页面原生抽取为空或过短时，调用 DashScopeParse / 阿里云文档智能修复。
- 阿里云百炼模型接入：通过 `DASHSCOPE_API_KEY` 调用兼容 OpenAI 风格接口的生成、嵌入和重排能力。
- RAGPipeline：将检索、重排、强约束 prompt、拒答门控和一致性验证拆成可配置流程。
- C1-C4 实验配置：使用 `configs/*.yaml` 对比 baseline、基础 RAG、增强 RAG 和完整方案。
- 批量评测：使用 `eval/eval.py` 读取 `eval/questions.jsonl` 并输出 CSV。
- 结果分析：使用 `results/summarize.py` 和 `results/plot_results.py` 生成统计结果和图表。
- 自动化测试：核心逻辑通过 `tests/` 中的 pytest 用例覆盖，测试不应依赖真实外部 API 调用。

## 当前模型与配置

| 能力 | 默认配置 |
|---|---|
| 生成模型 | `qwen-plus` |
| 一致性验证模型 | `qwen-flash` |
| Embedding | `text-embedding-v4` |
| Reranker | `qwen3-rerank` |
| API Key | `DASHSCOPE_API_KEY` |
| 默认存储 | 开发模式本地文件存储 |

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并写入阿里云百炼 API Key。

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
THINKRAG_PARSE_PROVIDER=auto
THINKRAG_PARSE_CACHE_DIR=storage/parsed
DASHSCOPE_WORKSPACE_ID=
DASHSCOPE_CATEGORY_ID=
```

`DASHSCOPE_WORKSPACE_ID` 和 `DASHSCOPE_CATEGORY_ID` 是可选项。除非你的阿里云文档解析配置要求指定 workspace 或 category，否则保持为空即可。

### 3. 运行测试

```bash
python -m pytest tests -q
```

### 4. 启动应用

```bash
streamlit run app.py
```

访问：

```text
http://localhost:8501
```

## 文档解析范围

ThinkRAG 仍以原生文本抽取作为主路径。`THINKRAG_PARSE_PROVIDER=auto` 时，复杂 PDF、扫描 PDF 或原生抽取质量较差的 PDF 若页面文本为空或过短，会触发 DashScopeParse / 阿里云文档智能解析，解析后的节点会带上 `parse_provider` 元数据并进入索引。

解析结果会按文件内容 hash 缓存在 `storage/parsed/{sha256}.json`，同一文件不会重复提交 API。完整重建索引会先写入 `storage_staging/`，只有新索引持久化成功后才替换正式 `storage/`。

当前文档修复解析路径使用阿里云 API，不要求安装本地 OCR 依赖。

## 评测流程

准备评测集：

```text
eval/questions.jsonl
```

运行四组配置：

```bash
python eval/eval.py --config configs/C1_baseline.yaml --eval_set eval/questions.jsonl --output results/results_C1.csv
python eval/eval.py --config configs/C2_basic.yaml --eval_set eval/questions.jsonl --output results/results_C2.csv
python eval/eval.py --config configs/C3_enhanced.yaml --eval_set eval/questions.jsonl --output results/results_C3.csv
python eval/eval.py --config configs/C4_full.yaml --eval_set eval/questions.jsonl --output results/results_C4.csv
```

配置含义：

| 配置 | 含义 |
|---|---|
| C1 | 不检索，直接问 LLM |
| C2 | 基础 RAG 检索 |
| C3 | RAG + reranker + 强约束 prompt |
| C4 | C3 + 分数门控拒答 + 一致性验证 |

生成统计和图表：

```bash
python results/summarize.py
python results/plot_results.py
```

## 项目结构

```text
ThinkRAG/
├── app.py                 # Streamlit 入口
├── config.py              # 模型、存储和运行参数
├── configs/               # C1-C4 实验配置
├── eval/                  # 批量评测脚本和 JSONL 评测集
├── results/               # 统计和图表脚本
├── src/
│   ├── embeddings/        # 阿里云 embedding 与缓存
│   ├── llm/               # 阿里云客户端
│   ├── prompts/           # 基础和强约束 prompt
│   └── rag/               # RAGPipeline、reranker、拒答门控、一致性验证
├── server/                # 后端索引、检索、存储和文档解析模块
├── frontend/              # Streamlit 页面
├── tests/                 # 自动化测试
└── docs/                  # 部署指南、图片和 legacy 文档
```

## 文档

- [部署与评测指南](docs/deployment.md)
- [旧版 Hugging Face 模型下载说明](docs/legacy/HowToDownloadModels.md)
- [旧版 Python 虚拟环境说明](docs/legacy/HowToUsePythonVirtualEnv.md)
- [安全政策](SECURITY.md)
- [行为准则](docs/Code_of_Conduct.md)

## Legacy 说明

本仓库保留了原 ThinkRAG 中 Ollama、Hugging Face 本地模型和多 provider 配置相关代码，以便兼容和参考。当前公开主流程以阿里云百炼 DashScope 为准；旧版使用说明已移动到 `docs/legacy/`。

## License

ThinkRAG 使用 [MIT License](LICENSE)。
