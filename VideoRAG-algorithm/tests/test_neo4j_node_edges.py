import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


class FakeResult:
    def __init__(self, records):
        self.records = records

    async def __aiter__(self):
        for record in self.records:
            yield record


class FakeSession:
    def __init__(self):
        self.query = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def run(self, query, source_id):
        self.query = query
        if "-[r]->" in query:
            records = [] if source_id == "ZULU" else [
                {"source": source_id, "target": "ZULU"}
            ]
        else:
            records = [{"source": "ZULU", "target": "ALPHA"}]
        return FakeResult(records)


def test_get_node_edges_includes_incoming_relationships(monkeypatch):
    package_path = Path(__file__).parents[1] / "videorag"
    storage_path = package_path / "_storage"

    package = types.ModuleType("videorag")
    package.__path__ = [str(package_path)]
    storage_package = types.ModuleType("videorag._storage")
    storage_package.__path__ = [str(storage_path)]
    monkeypatch.setitem(sys.modules, "videorag", package)
    monkeypatch.setitem(sys.modules, "videorag._storage", storage_package)

    base = types.ModuleType("videorag.base")
    base.BaseGraphStorage = type("BaseGraphStorage", (), {})
    base.SingleCommunitySchema = dict
    utils = types.ModuleType("videorag._utils")
    utils.logger = types.SimpleNamespace(info=Mock())
    prompt = types.ModuleType("videorag.prompt")
    prompt.GRAPH_FIELD_SEP = "<SEP>"
    for name, module in (
        ("videorag.base", base),
        ("videorag._utils", utils),
        ("videorag.prompt", prompt),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    neo4j = types.ModuleType("neo4j")
    neo4j.AsyncGraphDatabase = types.SimpleNamespace(driver=Mock())
    monkeypatch.setitem(sys.modules, "neo4j", neo4j)

    module_name = "videorag._storage.gdb_neo4j"
    module_path = storage_path / "gdb_neo4j.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    session = FakeSession()
    storage = object.__new__(module.Neo4jStorage)
    storage.namespace = "test_graph"
    storage.async_driver = types.SimpleNamespace(session=lambda: session)

    edges = asyncio.run(storage.get_node_edges("ZULU"))

    assert edges == [("ZULU", "ALPHA")]
    assert "-[r]-" in session.query
    assert "-[r]->" not in session.query
