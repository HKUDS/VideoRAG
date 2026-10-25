import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "reproduce"
    / "quantitative_comparison"
    / "batch_quant_eval_parse.py"
)


def load_parser(monkeypatch):
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    setproctitle = types.ModuleType("setproctitle")
    setproctitle.setproctitle = lambda name: None
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "setproctitle", setproctitle)

    spec = importlib.util.spec_from_file_location("quant_parse_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_batch_result(path, custom_id="query-1", content="invalid"):
    path.write_text(
        json.dumps(
            [
                {
                    "custom_id": custom_id,
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": content}}]
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


def valid_score_result():
    return {
        metric: {"Score": 4, "Explanation": "synthetic"}
        for metric in (
            "Comprehensiveness",
            "Empowerment",
            "Trustworthiness",
            "Depth",
            "Density",
            "Overall Score",
        )
    }


def test_import_preserves_environment_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    load_parser(monkeypatch)
    assert os.environ["OPENAI_API_KEY"] == "synthetic-openai-key"


def test_retry_exhaustion_is_bounded_without_partial_output(
    monkeypatch, tmp_path
):
    module = load_parser(monkeypatch)
    module.base_dir = "fixture"
    monkeypatch.chdir(tmp_path)
    result_dir = tmp_path / "batch_requests" / "fixture"
    result_dir.mkdir(parents=True)
    write_batch_result(result_dir / "bad-output.json")
    calls = []

    class FailingCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise OSError("synthetic provider failure")

    module.OpenAI = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    request_dict = {
        "query-1": {
            "model": "synthetic-model",
            "messages": [],
            "response_format": {},
        }
    }

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        module.process_file("bad-output", request_dict)

    assert len(calls) == 3
    assert not (result_dir / "bad-output-parse-result.json").exists()


def test_successful_retry_writes_valid_score_result(monkeypatch, tmp_path):
    module = load_parser(monkeypatch)
    module.base_dir = "fixture"
    monkeypatch.chdir(tmp_path)
    result_dir = tmp_path / "batch_requests" / "fixture"
    result_dir.mkdir(parents=True)
    write_batch_result(result_dir / "recoverable-output.json")
    expected = valid_score_result()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(expected))
            )
        ]
    )
    calls = []

    def create_completion(**kwargs):
        calls.append(kwargs)
        return response

    module.OpenAI = lambda: SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_completion)
        )
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    module.process_file(
        "recoverable-output",
        {
            "query-1": {
                "model": "synthetic-model",
                "messages": [],
                "response_format": {},
            }
        },
    )

    assert len(calls) == 1
    assert json.loads(
        (result_dir / "recoverable-output-parse-result.json").read_text(
            encoding="utf-8"
        )
    ) == {"query-1": expected}


def test_main_surfaces_worker_exceptions(monkeypatch, tmp_path):
    module = load_parser(monkeypatch)
    module.base_dir = "fixture"
    module.request_file = "requests.jsonl"
    module.result_files = ["broken-output"]
    monkeypatch.chdir(tmp_path)
    result_dir = tmp_path / "batch_requests" / "fixture"
    result_dir.mkdir(parents=True)
    (result_dir / "requests.jsonl").write_text("", encoding="utf-8")

    def fail_worker(file_id, request_dict):
        raise OSError(f"synthetic failure for {file_id}")

    module.process_file = fail_worker

    with pytest.raises(RuntimeError, match="broken-output"):
        module.main()
