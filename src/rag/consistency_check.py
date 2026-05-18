CONSISTENCY_CHECK_PROMPT = """你是一个严格的事实核查员。请判断下面的“模型回答”是否完全可以从“检索证据”中得出。

【检索证据】
{evidence}

【模型回答】
{answer}

判断标准：
- Y：回答的所有内容都能在证据中找到明确依据
- P：回答大部分有依据，但有少量推测或证据未直接覆盖
- N：回答的核心内容在证据中找不到依据

只回答一个字符：Y、P 或 N。不要任何其他文字。"""


def verify_consistency(answer: str, evidence: str, client) -> tuple[str, str]:
    response = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {
                "role": "user",
                "content": CONSISTENCY_CHECK_PROMPT.format(
                    evidence=evidence,
                    answer=answer,
                ),
            }
        ],
        temperature=0,
        max_tokens=5,
    )
    label = response.choices[0].message.content.strip().upper()[:1]
    labels = {
        "Y": ("Y", "高可信：答案完全有证据支持"),
        "P": ("P", "中可信：答案部分有证据，含少量推测"),
        "N": ("N", "低可信：证据中无明确依据，请人工复核"),
    }
    return labels.get(label, ("P", "验证异常，默认标记为中可信"))
