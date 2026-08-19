import pytest

import ollama
from app.llm.ollama_client import OllamaClient


def test_ollama_client_wraps_service_errors(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self, host=None, timeout=None):
            calls["init"] = {"host": host, "timeout": timeout}

        def chat(self, model, messages, stream=False):
            calls["chat"] = {"model": model, "messages": messages, "stream": stream}
            raise OSError("service unavailable")

    monkeypatch.setattr(ollama, "Client", FakeClient)

    client = OllamaClient(model="llama3.2", timeout=1)

    with pytest.raises(RuntimeError, match="Ollama is not available or not running"):
        client.generate("hello")

    assert calls["init"]["timeout"] == 1
