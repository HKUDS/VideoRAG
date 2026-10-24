import importlib.util
import sys
import types
from pathlib import Path


def load_base_module(monkeypatch):
    videorag_path = Path(__file__).parents[1] / "videorag"
    package = types.ModuleType("videorag")
    package.__path__ = [str(videorag_path)]
    utils = types.ModuleType("videorag._utils")
    utils.EmbeddingFunc = type("EmbeddingFunc", (), {})
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._utils", utils)
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    spec = importlib.util.spec_from_file_location(
        "videorag.base", videorag_path / "base.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "videorag.base", module)
    spec.loader.exec_module(module)
    return module


def test_naive_token_budget_can_be_set_when_constructing_query_param(monkeypatch):
    query_param_class = load_base_module(monkeypatch).QueryParam

    assert query_param_class().naive_max_token_for_text_unit == 12000
    assert (
        query_param_class(naive_max_token_for_text_unit=256).naive_max_token_for_text_unit
        == 256
    )
