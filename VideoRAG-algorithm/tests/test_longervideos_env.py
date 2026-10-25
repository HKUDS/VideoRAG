import runpy
import sys
import types
from pathlib import Path


def test_longervideos_example_preserves_configured_openai_key(monkeypatch):
    package = types.ModuleType("videorag")
    package.__path__ = []
    package.videorag = types.ModuleType("videorag.videorag")
    package.videorag.VideoRAG = type("VideoRAG", (), {})
    package.videorag.QueryParam = type("QueryParam", (), {})
    llm = types.ModuleType("videorag._llm")

    class LLMConfig:
        def __init__(self, **kwargs):
            self.options = kwargs

    llm.LLMConfig = LLMConfig
    llm.openai_embedding = object()
    llm.gpt_4o_mini_complete = object()
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._llm", llm)
    monkeypatch.setitem(sys.modules, "videorag.videorag", package.videorag)
    monkeypatch.setattr(sys, "argv", ["videorag_longervideos.py"])
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")

    script = Path(__file__).parents[1] / "videorag_longervideos.py"
    runpy.run_path(str(script), run_name="video_example_test")

    assert sys.modules["os"].environ["OPENAI_API_KEY"] == "synthetic-openai-key"
