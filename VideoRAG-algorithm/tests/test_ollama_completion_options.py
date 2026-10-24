import asyncio
import importlib.util
import sys
import types
from pathlib import Path


class FakeCache:
    def __init__(self):
        self.read_keys = []
        self.writes = []
        self.flushes = 0

    async def get_by_id(self, key):
        self.read_keys.append(key)
        return None

    async def upsert(self, records):
        self.writes.append(records)

    async def index_done_callback(self):
        self.flushes += 1


class FakeOllamaClient:
    def __init__(self):
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "answer"}}


def load_llm_module(monkeypatch):
    package_path = Path(__file__).parents[1] / "videorag"
    package = types.ModuleType("videorag")
    package.__path__ = [str(package_path)]
    utils = types.ModuleType("videorag._utils")
    utils.compute_args_hash = lambda *args: "cache-key"
    utils.wrap_embedding_func_with_attrs = lambda **kwargs: lambda func: func
    utils.EmbeddingFunc = object
    base = types.ModuleType("videorag.base")
    base.BaseKVStorage = type("BaseKVStorage", (), {})
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._utils", utils)
    monkeypatch.setitem(sys.modules, "videorag.base", base)

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = type("AsyncOpenAI", (), {})
    openai.AsyncAzureOpenAI = type("AsyncAzureOpenAI", (), {})
    openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    ollama = types.ModuleType("ollama")
    ollama.AsyncClient = type("AsyncClient", (), {})
    tenacity = types.ModuleType("tenacity")
    tenacity.retry = lambda **kwargs: lambda func: func
    tenacity.stop_after_attempt = lambda attempts: attempts
    tenacity.wait_exponential = lambda **kwargs: None
    tenacity.retry_if_exception_type = lambda exception_types: None
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    numpy.array = lambda values: values
    for name, module in (
        ("openai", openai),
        ("ollama", ollama),
        ("tenacity", tenacity),
        ("numpy", numpy),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "videorag._llm"
    module_path = package_path / "_llm.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_ollama_wrappers_forward_cache_and_generation_options(monkeypatch):
    module = load_llm_module(monkeypatch)

    for completion in (module.ollama_complete, module.ollama_mini_complete):
        cache = FakeCache()
        client = FakeOllamaClient()
        monkeypatch.setattr(module, "get_ollama_async_client_instance", lambda: client)

        result = asyncio.run(
            completion(
                "local-model",
                "prompt",
                hashing_kv=cache,
                use_cache=True,
                max_tokens=128,
                temperature=0.25,
                options={"num_ctx": 2048},
            )
        )

        assert result == "answer"
        assert cache.read_keys == ["cache-key"]
        assert cache.writes == [
            {"cache-key": {"return": "answer", "model": "local-model"}}
        ]
        assert cache.flushes == 1
        assert client.calls[0]["options"] == {
            "num_ctx": 2048,
            "num_predict": 128,
            "temperature": 0.25,
        }
