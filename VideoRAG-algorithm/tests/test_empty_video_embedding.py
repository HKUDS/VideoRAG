import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


def test_empty_video_upsert_does_not_load_imagebind(monkeypatch):
    package_path = Path(__file__).parents[1] / "videorag"
    storage_path = package_path / "_storage"

    package = types.ModuleType("videorag")
    package.__path__ = [str(package_path)]
    storage_package = types.ModuleType("videorag._storage")
    storage_package.__path__ = [str(storage_path)]
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._storage", storage_package)

    logger = types.SimpleNamespace(info=Mock(), warning=Mock())
    utils = types.ModuleType("videorag._utils")
    utils.logger = logger
    base = types.ModuleType("videorag.base")
    base.BaseVectorStorage = type("BaseVectorStorage", (), {})
    video_utils = types.ModuleType("videorag._videoutil")
    video_utils.encode_video_segments = Mock()
    video_utils.encode_string_query = Mock()
    monkeypatch.setitem(sys.modules, "videorag._utils", utils)
    monkeypatch.setitem(sys.modules, "videorag.base", base)
    monkeypatch.setitem(sys.modules, "videorag._videoutil", video_utils)

    torch = types.ModuleType("torch")
    numpy = types.ModuleType("numpy")
    nano_vectordb = types.ModuleType("nano_vectordb")
    nano_vectordb.NanoVectorDB = type("NanoVectorDB", (), {})
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda values, **kwargs: values
    imagebind = types.ModuleType("imagebind")
    imagebind.__path__ = []
    imagebind_models = types.ModuleType("imagebind.models")
    embedder = types.SimpleNamespace(cuda=Mock(), eval=Mock())
    imagebind_model = types.SimpleNamespace(imagebind_huge=Mock(return_value=embedder))
    imagebind_models.imagebind_model = imagebind_model
    for name, module in (
        ("torch", torch),
        ("numpy", numpy),
        ("nano_vectordb", nano_vectordb),
        ("tqdm", tqdm),
        ("imagebind", imagebind),
        ("imagebind.models", imagebind_models),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "videorag._storage.vdb_nanovectordb"
    module_path = storage_path / "vdb_nanovectordb.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    storage = object.__new__(module.NanoVectorDBVideoSegmentStorage)
    storage.namespace = "video_segments"

    assert asyncio.run(storage.upsert("sample", {}, "mp4")) == []
    imagebind_model.imagebind_huge.assert_not_called()
    embedder.cuda.assert_not_called()
