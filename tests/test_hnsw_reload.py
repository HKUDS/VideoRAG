"""Offline native HNSW save/reload regression for both VideoRAG backends.

Run: python -m unittest discover -s tests -p test_hnsw_reload.py -v
Dependencies: numpy, hnswlib, xxhash. No models or API calls are needed.
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = (
    ROOT / "VideoRAG-algorithm/videorag",
    ROOT / "Vimo-desktop/python_backend/videorag",
)


def load_storage(root):
    """Load the real storage and base modules without model-loading __init__."""
    name = "_hnsw_reload_" + str(BACKENDS.index(root))
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    utils = types.ModuleType(name + "._utils")
    utils.logger = logging.getLogger(name)
    utils.EmbeddingFunc = object
    modules = {name: package, utils.__name__: utils}
    with patch.dict(sys.modules, modules):
        for relative, source in (("base", root / "base.py"),
                                 ("_storage.vdb_hnswlib", root / "_storage/vdb_hnswlib.py")):
            spec = importlib.util.spec_from_file_location(name + "." + relative, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return module.HNSWVectorStorage


class Embedder:
    embedding_dim = 2

    async def __call__(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class HNSWReloadTests(unittest.TestCase):
    def storage(self, cls, directory, ef=None):
        options = {} if ef is None else {"ef_search": ef}
        return cls(namespace="test", global_config={
            "working_dir": directory, "vector_db_storage_cls_kwargs": options,
        }, embedding_func=Embedder(), meta_fields={"content"})

    def test_new_index_honors_default_and_custom_search_ef(self):
        for backend in BACKENDS:
            cls = load_storage(backend)
            for configured, expected in ((None, 50), (83, 83)):
                with self.subTest(backend=backend, ef=configured), tempfile.TemporaryDirectory() as tmp:
                    storage = self.storage(cls, tmp, configured)
                    self.assertEqual(storage._index.ef, expected)
                    self.assertEqual(asyncio.run(storage.query("empty")), [])

    def test_reloaded_index_restores_current_search_ef_and_metadata(self):
        for backend in BACKENDS:
            cls = load_storage(backend)
            # The setting is runtime configuration, not serialized HNSW state.
            for configured, expected in ((None, 50), (83, 83), (17, 17)):
                with self.subTest(backend=backend, ef=configured), tempfile.TemporaryDirectory() as tmp:
                    original = self.storage(cls, tmp, 71)
                    asyncio.run(original.upsert({"item": {"content": "evidence"}}))
                    asyncio.run(original.index_done_callback())
                    reloaded = self.storage(cls, tmp, configured)
                    self.assertEqual(reloaded._index.ef, expected)
                    self.assertEqual(reloaded._current_elements, 1)
                    hits = asyncio.run(reloaded.query("evidence", top_k=1))
                    self.assertEqual(hits[0]["id"], "item")
                    self.assertEqual(hits[0]["content"], "evidence")
                    self.assertAlmostEqual(hits[0]["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
