import asyncio
import importlib.util
import json
from dataclasses import dataclass
import os
import sys
import types
from pathlib import Path

import pytest


def load_storage_module(monkeypatch):
    package_path = Path(__file__).parents[1] / "videorag"
    package = types.ModuleType("videorag")
    package.__path__ = [str(package_path)]
    storage_package = types.ModuleType("videorag._storage")
    storage_package.__path__ = [str(package_path / "_storage")]

    def load_json(file_name):
        if not os.path.exists(file_name):
            return None
        with open(file_name, encoding="utf-8") as stream:
            return json.load(stream)

    def write_json(value, file_name):
        with open(file_name, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)

    utils = types.ModuleType("videorag._utils")
    utils.load_json = load_json
    utils.write_json = write_json
    utils.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    base = types.ModuleType("videorag.base")

    @dataclass
    class BaseKVStorage:
        namespace: str
        global_config: dict

    base.BaseKVStorage = BaseKVStorage
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._storage", storage_package)
    monkeypatch.setitem(sys.modules, "videorag._utils", utils)
    monkeypatch.setitem(sys.modules, "videorag.base", base)

    module_name = "videorag._storage.kv_json"
    module_path = package_path / "_storage" / "kv_json.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_interrupted_write_preserves_previous_json_snapshot(tmp_path, monkeypatch):
    module = load_storage_module(monkeypatch)
    store_path = tmp_path / "kv_store_cache.json"
    previous_data = {"existing": {"return": "valid snapshot"}}
    store_path.write_text(json.dumps(previous_data), encoding="utf-8")
    storage = module.JsonKVStorage(
        namespace="cache", global_config={"working_dir": str(tmp_path)}
    )
    storage._data["new"] = {"return": "next snapshot"}

    def interrupted_write(value, file_name):
        Path(file_name).write_text('{"partial":', encoding="utf-8")
        raise OSError("simulated interrupted JSON write")

    monkeypatch.setattr(module, "write_json", interrupted_write)
    with pytest.raises(OSError, match="simulated interrupted JSON write"):
        asyncio.run(storage.index_done_callback())

    assert json.loads(store_path.read_text(encoding="utf-8")) == previous_data
    assert not Path(f"{store_path}.tmp").exists()


def test_successful_write_replaces_snapshot(tmp_path, monkeypatch):
    module = load_storage_module(monkeypatch)
    store_path = tmp_path / "kv_store_cache.json"
    store_path.write_text('{"existing": {"return": "old"}}', encoding="utf-8")
    storage = module.JsonKVStorage(
        namespace="cache", global_config={"working_dir": str(tmp_path)}
    )
    storage._data["new"] = {"return": "next"}

    asyncio.run(storage.index_done_callback())

    assert json.loads(store_path.read_text(encoding="utf-8")) == {
        "existing": {"return": "old"},
        "new": {"return": "next"},
    }
    assert not Path(f"{store_path}.tmp").exists()
