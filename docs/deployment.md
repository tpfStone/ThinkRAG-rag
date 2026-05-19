# ThinkRAG Deployment and Evaluation Guide

本文档说明 ThinkRAG 的部署、运行、知识库导入、自动化测试、C1-C4 评测实验和结果生成流程。当前流程以阿里云百炼 DashScope API 为主，开发模式默认使用本地文件存储，不需要安装 Redis、Elasticsearch 或 ChromaDB 服务。

> 适用对象：开发者、实验复现者和课程/项目评测人员。若仅作为普通用户使用，请重点阅读第 4、6、7、8 节；自动化测试、C1-C4 评测、结果出图和人工打分章节可按需跳过。

## 1. 获取代码

克隆仓库并进入项目目录：

```bash
git clone <repository-url>
cd ThinkRAG
```

如果已经有本地仓库，请更新到需要运行的分支：

```bash
git fetch
git pull
```

后续命令默认在包含 `app.py` 的项目根目录下执行。

## 2. 环境要求

- Python 3.10+
- Git
- 阿里云百炼 DashScope API Key
- Windows / macOS / Linux

开发模式默认使用本地文件存储，知识库索引会写入 `storage/`。

## 3. 创建虚拟环境并安装依赖

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 配置 API Key

复制示例文件：

```powershell
copy .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

注意：

- `.env` 不要提交到 Git。
- `.env.example` 可以提交，但不能写真实 key。
- 如果已经启动过 Streamlit，修改 `.env` 后需要重启应用。

## 5. 运行自动化测试

```powershell
python -m pytest tests -q
```

测试应全部通过。实际通过数量以本地 `pytest` 输出为准。

如果看到 pydantic 的 `UnsupportedFieldAttributeWarning`，可以忽略；这是第三方依赖 warning，不影响当前测试结果。

## 6. 启动系统

在项目根目录运行：

```powershell
streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

## 7. 导入知识库

准备若干 RAG 相关 PDF 或文档资料，用于构建本地知识库。

操作步骤：

1. 打开 Streamlit 页面。
2. 进入知识库文件上传页面。
3. 上传 PDF 或文档资料。
4. 点击保存或生成索引。
5. 等待索引写入本地 `storage/`。

说明：

- `storage/` 是本地索引目录，默认被 `.gitignore` 忽略。
- C2、C3、C4 实验都依赖知识库索引。
- C1 baseline 不检索，但仍需要调用 LLM。

## 8. 手工 Smoke Test

知识库导入完成后，先手工测试 2-3 个问题，再运行完整评测。

建议测试：

1. 资料中有明确答案的问题，例如“RAG 由哪两个阶段组成？”
2. 资料中没有答案的问题，例如“明天天气怎么样？”
3. 边缘问题，用来看一致性验证是否出现黄色或红色可信度标签。

正常表现：

- 有答案的问题：返回答案 + 引用文档 + 分数 + 可信度标签。
- 不可回答问题：触发拒答，返回“根据现有知识库，未找到相关信息。”

## 9. 准备评测集

评测集文件路径：

```text
eval/questions.jsonl
```

每行一个 JSON，格式示例：

```jsonl
{"id":"Q001","question":"RAG 全称是什么？由哪两个核心阶段组成？","gold_answer":"RAG 全称是 Retrieval-Augmented Generation，由检索阶段和生成阶段组成。","gold_source":"RAG原始论文.pdf","answerable":true,"type":"factual"}
{"id":"Q024","question":"明天北京的天气怎么样？","gold_answer":"知识库中未提供该信息。","gold_source":null,"answerable":false,"type":"unanswerable"}
```

建议要求：

- 共 25 题。
- 其中 5 题为不可回答题。
- 可回答题必须能在知识库资料中找到出处。
- `answerable` 必须是 `true` 或 `false`。

## 10. 运行 C1-C4 评测

确保：

- `.env` 中有真实 `DASHSCOPE_API_KEY`
- 知识库索引已生成
- `eval/questions.jsonl` 已替换为正式评测集

然后运行：

```powershell
python eval/eval.py --config configs/C1_baseline.yaml --eval_set eval/questions.jsonl --output results/results_C1.csv
python eval/eval.py --config configs/C2_basic.yaml --eval_set eval/questions.jsonl --output results/results_C2.csv
python eval/eval.py --config configs/C3_enhanced.yaml --eval_set eval/questions.jsonl --output results/results_C3.csv
python eval/eval.py --config configs/C4_full.yaml --eval_set eval/questions.jsonl --output results/results_C4.csv
```

四组配置含义：

| 配置 | 含义 |
|---|---|
| C1 | 不检索，直接问 LLM |
| C2 | 基础 RAG 检索 |
| C3 | RAG + reranker + 强约束 prompt |
| C4 | C3 + 分数门控拒答 + 一致性验证 |

### C4 拒答阈值说明

当前 `configs/C4_full.yaml` 中拒答门控使用：

- `max_threshold: 0.3`
- `spread_threshold: 0.0`

这是基于当前样例和手工试运行得到的初步经验值。原计划中的 `0.5 / 0.05` 在当前知识库和相似度分布下偏严格，容易误拒答可回答问题。该阈值不是最终实验结论，正式 25 题评测集跑完后，应根据可回答题误拒率、不可回答题拒答率和引用支持情况重新校准。

## 11. 统计与出图

自动统计：

```powershell
python results/summarize.py
```

生成图表：

```powershell
python results/plot_results.py
```

图表默认输出到：

```text
results/charts/
```

注意：

- `results/results_C*.csv` 和 `results/charts/` 默认被 `.gitignore` 忽略。
- 如果需要保存实验结果，请单独归档 CSV 和图表。
- 如果需要把实验结果提交进仓库，请先确认 `.gitignore` 策略，避免提交 API Key、本地索引或缓存。

## 12. 人工打分字段

`eval.py` 自动输出的 CSV 字段包括：

```text
id, question, answerable, gold_answer, predicted_answer, retrieved_sources, max_score, refused, consistency_label
```

人工评分时建议额外加 3 列：

```text
correct
citation_ok
has_hallucination
```

含义：

- `correct`：答案是否正确。
- `citation_ok`：引用是否真的支持答案。
- `has_hallucination`：是否包含知识库不支持的内容。

## 13. 常见问题

### 13.1 `DASHSCOPE_API_KEY is not set`

说明 `.env` 没配置或应用没重启。

处理：

1. 检查 `.env` 是否存在。
2. 检查是否写了 `DASHSCOPE_API_KEY=...`。
3. 重启 Streamlit 或重新运行命令。

### 13.2 `No index found`

说明还没有导入知识库，或本地 `storage/` 中没有可用索引。

处理：

1. 打开 Streamlit。
2. 上传 PDF 或文档。
3. 生成知识库索引。
4. 再运行 C2、C3、C4。

### 13.3 pytest 有 warning

如果测试全部通过，可以继续。部分 warning 来自第三方依赖，不影响当前功能。

### 13.4 `git add` 出现 LF/CRLF warning

这是 Windows 换行符提示，不是代码错误。

### 13.5 不要提交这些文件

不要提交：

```text
.env
storage/
.embedding_cache/
.pytest_cache/
__pycache__/
streamlit.out.log
streamlit.err.log
```
