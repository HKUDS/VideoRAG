import asyncio
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np


def _load_networkx_storage_class():
    package_path = Path(__file__).parents[1] / "videorag"
    package_name = "_videorag_networkx_embedding_test"
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

    prompt = types.ModuleType(f"{package_name}.prompt")
    prompt.GRAPH_FIELD_SEP = "|"
    sys.modules[prompt.__name__] = prompt

    module = importlib.import_module(f"{storage_name}.gdb_networkx")
    return module.NetworkXStorage


def test_node2vec_embedding_returns_graph_node_keys(monkeypatch, tmp_path):
    storage_class = _load_networkx_storage_class()
    storage = storage_class(
        namespace="graph",
        global_config={"working_dir": str(tmp_path), "node2vec_params": {}},
    )
    asyncio.run(storage.upsert_node("ENTITY", {"entity_type": "PERSON"}))

    graspologic = types.ModuleType("graspologic")
    graspologic.embed = types.SimpleNamespace(
        node2vec_embed=lambda graph, **params: (
            np.asarray([[0.25, 0.75]], dtype=np.float32),
            ["ENTITY"],
        )
    )
    monkeypatch.setitem(sys.modules, "graspologic", graspologic)

    embeddings, node_ids = asyncio.run(storage.embed_nodes("node2vec"))

    assert node_ids == ["ENTITY"]
    assert np.array_equal(embeddings, [[0.25, 0.75]])
