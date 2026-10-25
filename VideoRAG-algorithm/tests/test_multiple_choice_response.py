import asyncio
import importlib.util
from pathlib import Path

import pytest


_response_path = Path(__file__).parents[1] / "videorag" / "_multiple_choice.py"
_response_spec = importlib.util.spec_from_file_location(
    "videorag_test_multiple_choice", _response_path
)
_response_module = importlib.util.module_from_spec(_response_spec)
_response_spec.loader.exec_module(_response_module)
request_multiple_choice_response = (
    _response_module.request_multiple_choice_response
)


def test_invalid_model_responses_retry_then_return_required_json():
    responses = iter(
        [
            "not json",
            '{"Answer": "B"}',
            '{"Answer": "B", "Explanation": "Supported by the context."}',
        ]
    )
    calls = []
    invalid_attempts = []

    async def use_model_func(query, **kwargs):
        calls.append((query, kwargs))
        return next(responses)

    result = asyncio.run(
        request_multiple_choice_response(
            "question",
            "system prompt",
            use_model_func,
            on_invalid_response=lambda error, attempt, limit: invalid_attempts.append(
                (attempt, limit)
            ),
        )
    )

    assert result == {"Answer": "B", "Explanation": "Supported by the context."}
    assert invalid_attempts == [(1, 3), (2, 3)]
    assert len(calls) == 3
    assert all(call[1]["use_cache"] is False for call in calls)


def test_invalid_model_responses_stop_after_the_attempt_limit():
    calls = []
    invalid_attempts = []

    async def use_model_func(query, **kwargs):
        calls.append(query)
        return "not json"

    with pytest.raises(ValueError, match="after 3 attempts"):
        asyncio.run(
            request_multiple_choice_response(
                "question",
                "system prompt",
                use_model_func,
                on_invalid_response=lambda error, attempt, limit: invalid_attempts.append(
                    (attempt, limit)
                ),
            )
        )

    assert calls == ["question", "question", "question"]
    assert invalid_attempts == [(1, 3), (2, 3), (3, 3)]
