from types import SimpleNamespace

from src.rag.consistency_check import verify_consistency


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def test_verify_consistency_maps_known_labels():
    assert verify_consistency("answer", "evidence", FakeClient("Y"))[0] == "Y"
    assert verify_consistency("answer", "evidence", FakeClient("P"))[0] == "P"
    assert verify_consistency("answer", "evidence", FakeClient("N"))[0] == "N"


def test_verify_consistency_defaults_unknown_label_to_p():
    label, reason = verify_consistency("answer", "evidence", FakeClient("maybe"))
    assert label == "P"
    assert "默认" in reason


def test_verify_consistency_uses_qwen_flash():
    client = FakeClient("Y")
    verify_consistency("answer", "evidence", client)
    assert client.chat.completions.last_kwargs["model"] == "qwen-flash"
    assert client.chat.completions.last_kwargs["temperature"] == 0
    assert client.chat.completions.last_kwargs["max_tokens"] == 5
