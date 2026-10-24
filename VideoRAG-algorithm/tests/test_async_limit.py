import asyncio
import importlib.util
from pathlib import Path

import pytest


_utils_path = Path(__file__).parents[1] / "videorag" / "_utils.py"
_utils_spec = importlib.util.spec_from_file_location("videorag_test_utils", _utils_path)
_utils_module = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_module)
limit_async_func_call = _utils_module.limit_async_func_call


def test_async_call_exception_releases_concurrency_slot():
    @limit_async_func_call(max_size=1)
    async def operation(should_fail):
        if should_fail:
            raise RuntimeError("synthetic provider failure")
        return "ok"

    async def run_calls():
        with pytest.raises(RuntimeError, match="synthetic provider failure"):
            await operation(True)
        return await asyncio.wait_for(operation(False), timeout=0.1)

    assert asyncio.run(run_calls()) == "ok"
