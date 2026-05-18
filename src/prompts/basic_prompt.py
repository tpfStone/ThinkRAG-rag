BASIC_RAG_PROMPT = """请基于给定上下文回答问题。回答要简洁、准确，并尽量使用问题本身的语言。

上下文：
{context}

问题：
{question}

回答："""

BASIC_DIRECT_PROMPT = """请回答用户问题。回答要简洁、准确，并使用问题本身的语言。

问题：
{question}

回答："""
