import asyncio
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_hnsw_storage_class():
    package_path = Path(__file__).parents[1] / "videorag"
    package_name = "_videorag_hnsw_upsert_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    package.__package__ = package_name
    package.__spec__ = importlib.machinery.ModuleSpec(
        package_name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules[package_name] = package

    storage_name = f"{package_name}._storage"
    storage_package = types.ModuleType(storage_name)
    storage_path = package_path / "_storage"
    storage_package.__path__ = [str(storage_path)]
    storage_package.__package__ = storage_name
    storage_package.__spec__ = importlib.machinery.ModuleSpec(
        storage_name, loader=None, is_package=True
    )
    storage_package.__spec__.submodule_search_locations = [str(storage_path)]
    sys.modules[storage_name] = storage_package

    module = importlib.import_module(f"{storage_name}.vdb_hnswlib")
    return module.HNSWVectorStorage


class SyntheticEmbedding:
    embedding_dim = 2

    async def __call__(self, texts):
        return np.asarray(
            [[0.0, 1.0] if text == "updated" else [1.0, 0.0] for text in texts],
            dtype=np.float32,
        )


def test_upsert_existing_id_succeeds_at_capacity(tmp_path):
    pytest.importorskip("hnswlib")
    storage_class = _load_hnsw_storage_class()
    storage = storage_class(
        namespace="chunks",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 1,
            "vector_db_storage_cls_kwargs": {"max_elements": 1},
        },
        embedding_func=SyntheticEmbedding(),
    )

    async def insert_and_update():
        await storage.upsert({"same-id": {"content": "before"}})
        await storage.upsert({"same-id": {"content": "updated"}})
        with pytest.raises(ValueError, match="Cannot insert 1 new elements"):
            await storage.upsert({"new-id": {"content": "new"}})
        return await storage.query("updated", top_k=1)

    result = asyncio.run(insert_and_update())

    assert result[0]["id"] == "same-id"
    assert result[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert storage._current_elements == 1
