import runpy
import sys
import types
from pathlib import Path


EXAMPLE_NAMES = ("process_videos_deepseek.py", "query_videos_deepseek.py")


def test_deepseek_examples_preserve_configured_environment_keys(monkeypatch):
    package = types.ModuleType("videorag")
    package.__path__ = []
    package.VideoRAG = type("VideoRAG", (), {})
    package.QueryParam = type("QueryParam", (), {})
    llm = types.ModuleType("videorag._llm")
    llm.deepseek_bge_config = object()
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._llm", llm)

    examples_dir = Path(__file__).parents[1] / "examples"
    for index, example_name in enumerate(EXAMPLE_NAMES):
        monkeypatch.setenv("DEEPSEEK_API_KEY", f"synthetic-deepseek-{index}")
        monkeypatch.setenv("SILICONFLOW_API_KEY", f"synthetic-siliconflow-{index}")
        runpy.run_path(
            str(examples_dir / example_name),
            run_name=f"video_example_test_{index}",
        )
        assert sys.modules["os"].environ["DEEPSEEK_API_KEY"] == (
            f"synthetic-deepseek-{index}"
        )
        assert sys.modules["os"].environ["SILICONFLOW_API_KEY"] == (
            f"synthetic-siliconflow-{index}"
        )
