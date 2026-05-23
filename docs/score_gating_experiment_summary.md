# 分数门控实验总结

日期：2026-05-23

本文记录 C4 评测中对拒答门控阈值的实验过程和结论。实验目标是在尽量不误拒可回答问题的前提下，提高不可回答问题的门控拒答率。

## 当前门控逻辑

实现位置：`src/rag/score_gating.py`

当前门控只使用检索或重排序后的分数：

```python
if not scores:
    refuse

top_scores = scores[:top_k]
max_score = max(top_scores)
mean_score = sum(top_scores) / len(top_scores)
spread = max_score - mean_score

if max_score < max_threshold:
    refuse

if spread < spread_threshold:
    refuse
```

当前推荐配置位于 `configs/C4_full.yaml`：

```yaml
refuse_gate:
  enabled: true
  max_threshold: 0.33
  spread_threshold: 0.02
```

## 评测设置

评测集：

```text
eval/questions.jsonl
```

数据规模：

| 类型 | 数量 |
|---|---:|
| 可回答问题 | 20 |
| 不可回答问题 | 5 |
| 总数 | 25 |

使用过的评测命令：

```powershell
.venv\Scripts\python.exe eval\eval.py --config configs\C4_full.yaml --eval_set eval\questions.jsonl --output results\results_C4_spread_002.csv
.venv\Scripts\python.exe eval\eval.py --config configs\C4_full.yaml --eval_set eval\questions.jsonl --output results\results_C4_max033_spread002.csv
```

基线结果文件：

```text
results/results_C4.csv
```

## 阈值对比

| 配置 | 结果文件 | 门控拒答总数 | 可回答问题门控误拒 | 不可回答问题门控拒答 | 不可回答问题门控拒答率 | 一致性标签 |
|---|---|---:|---:|---:|---:|---|
| `max=0.30`, `spread=0.00` | `results_C4.csv` | 1 | 0 | 1/5 | 20% | `Y=20`, `N=4`, `N/A=1` |
| `max=0.30`, `spread=0.02` | `results_C4_spread_002.csv` | 1 | 0 | 1/5 | 20% | `Y=21`, `N=3`, `N/A=1` |
| `max=0.33`, `spread=0.02` | `results_C4_max033_spread002.csv` | 3 | 0 | 3/5 | 60% | `Y=21`, `N=1`, `N/A=3` |

## 实验发现

### 1. `spread_threshold=0.02` 没有改变门控结果

在固定 `max_threshold=0.30` 的情况下，将 `spread_threshold` 从 `0.00` 调到 `0.02`，没有新增任何被门控拒答的样本。

25 题中最低的几个 `spread` 为：

| 问题 | 是否可回答 | `spread` |
|---|---:|---:|
| Q001 | true | 0.0282 |
| Q021 | false | 0.0318 |
| Q022 | false | 0.0335 |
| Q010 | true | 0.0346 |

由于所有样本的 `spread` 都高于 `0.02`，所以 `spread_threshold=0.02` 没有触发。这个参数在代码中确实生效，但在当前数据分布下只是一个很弱的保护条件，不是主要决策边界。

继续提高 `spread_threshold` 有明显风险：

| 候选 `spread_threshold` | 预期问题 |
|---:|---|
| `0.03` | 会先影响 Q001，而 Q001 是可回答问题。 |
| `0.04` | 会拒掉 Q021/Q022，但也会误拒可回答问题 Q001/Q010。 |

结论：当前评测集里，可回答问题和不可回答问题的 `spread` 分布有重叠，单独依赖 `spread_threshold` 不适合做主要拒答边界。

### 2. `max_threshold=0.33` 明显提高了不可回答问题的门控拒答率

将 `max_threshold` 从 `0.30` 提高到 `0.33` 后，新增拒掉了两个不可回答问题：

| 问题 | 是否可回答 | `max_score` | 旧配置是否门控拒答 | 新配置是否门控拒答 |
|---|---:|---:|---:|---:|
| Q021 | false | 0.3000 | false | true |
| Q024 | false | 0.3256 | false | true |

Q025 在旧配置下已经被拒答：

| 问题 | 是否可回答 | `max_score` | 是否门控拒答 |
|---|---:|---:|---:|
| Q025 | false | 0.2558 | true |

在 `0.33 / 0.02` 这组配置下，没有可回答问题被门控误拒。

### 3. 仍未被门控挡住的问题是高分不可回答问题

仍有两个不可回答问题通过了门控：

| 问题 | `max_score` | 是否门控拒答 | 说明 |
|---|---:|---:|---|
| Q022 | 0.6126 | false | 分数过高，单纯依赖 `max_threshold` 很难拒掉。 |
| Q023 | 0.3912 | false | 高于 `0.33`，需要更高阈值或额外规则。 |

Q023 在最新结果中被一致性检查标为 `Y`，但它本身是不可回答问题，且回答文本是拒答式内容。这说明一致性标签和拒答指标不能混为一谈。

### 4. 门控拒答和文本拒答不同

CSV 中的 `refused` 字段只记录门控层是否拒答。即使 `refused=false`，LLM 也可能在看到检索证据后自行输出拒答文本。

本轮观察到的文本拒答数量：

| 结果文件 | 门控拒答数 | 文本包含拒答句的数量 | 可回答问题文本拒答数 | 不可回答问题文本拒答数 |
|---|---:|---:|---:|---:|
| `results_C4_spread_002.csv` | 1 | 7 | 2 | 5 |
| `results_C4_max033_spread002.csv` | 3 | 6 | 1 | 5 |

如果要评估用户实际看到的拒答效果，建议新增有效拒答指标：

```text
text_refused = predicted_answer contains "未找到相关信息"
effective_refused = refused OR text_refused
```

## 当前结论

在已测试的三组配置中，推荐使用：

```yaml
max_threshold: 0.33
spread_threshold: 0.02
```

理由：

- 不可回答问题的门控拒答率从 20% 提升到 60%。
- 可回答问题的门控误拒率仍为 0%。
- 相比只调 `spread_threshold`，提高 `max_threshold` 对当前数据更有效。

这个配置应表述为“当前 25 题评测下的最佳经验阈值”，不能表述为全局最优。当前评测集较小，并且 Q022、Q023 仍未被门控挡住。

## 后续优化建议

1. 在保持 `spread_threshold=0.02` 的前提下，继续测试更高的 `max_threshold`：

```text
0.35 / 0.02
0.38 / 0.02
0.40 / 0.02
```

2. 扩展 `eval/eval.py` 的输出字段：

```text
refusal_reason
text_refused
effective_refused
top1_score
top2_score
score_gap
score_spread
```

3. 考虑更可解释的组合门控规则：

```text
top1 < 0.33 -> refuse
or top1 < 0.40 and top1 - top2 < 0.05 -> refuse
```

4. 排查可回答问题中 LLM 自行拒答的情况。最新结果中 Q004 仍然输出了拒答式回答，但它是可回答问题。这更可能是强约束 prompt 或检索证据质量问题，不是分数门控本身的问题。
