import os
import runpy
import sys
import types
from pathlib import Path


def test_quantitative_batch_uploader_preserves_configured_key(monkeypatch):
    openai = types.ModuleType("openai")
    openai.__path__ = []
    openai.OpenAI = type("OpenAI", (), {})
    openai_lib = types.ModuleType("openai.lib")
    openai_lib.__path__ = []
    pydantic_utils = types.ModuleType("openai.lib._pydantic")
    pydantic_utils.to_strict_json_schema = lambda model: {}
    parsing_package = types.ModuleType("openai.lib._parsing")
    parsing_package.__path__ = []
    completions = types.ModuleType("openai.lib._parsing._completions")
    completions.type_to_response_format_param = lambda model: {}
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "openai.lib", openai_lib)
    monkeypatch.setitem(sys.modules, "openai.lib._pydantic", pydantic_utils)
    monkeypatch.setitem(sys.modules, "openai.lib._parsing", parsing_package)
    monkeypatch.setitem(
        sys.modules, "openai.lib._parsing._completions", completions
    )

    tiktoken = types.ModuleType("tiktoken")
    tiktoken.encoding_for_model = lambda name: types.SimpleNamespace(
        encode=lambda text: []
    )
    jsonlines = types.ModuleType("jsonlines")
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda values, *args, **kwargs: values
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda **kwargs: None
    for name, module in (
        ("tiktoken", tiktoken),
        ("jsonlines", jsonlines),
        ("tqdm", tqdm),
        ("pydantic", pydantic),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    script = (
        Path(__file__).parents[1]
        / "reproduce"
        / "quantitative_comparison"
        / "batch_quant_eval_upload.py"
    )
    runpy.run_path(str(script), run_name="quant_eval_import_test")

    assert os.environ["OPENAI_API_KEY"] == "synthetic-openai-key"
