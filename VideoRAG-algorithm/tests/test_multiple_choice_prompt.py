import importlib.util
import json
from pathlib import Path


_PROMPT_PATH = Path(__file__).parents[1] / "videorag" / "prompt.py"
_PROMPT_SPEC = importlib.util.spec_from_file_location("videorag_prompt", _PROMPT_PATH)
_PROMPT_MODULE = importlib.util.module_from_spec(_PROMPT_SPEC)
_PROMPT_SPEC.loader.exec_module(_PROMPT_MODULE)
PROMPTS = _PROMPT_MODULE.PROMPTS


def test_multiple_choice_prompt_contains_valid_json_example():
    rendered = PROMPTS["videorag_response_for_multiple_choice_question"].format(
        response_type="one answer",
        video_data="synthetic video context",
        chunk_data="synthetic text context",
    )
    json_example = rendered.split(
        "Please provide your answer in JSON format as follows:", 1
    )[1].split("Key points:", 1)[0]

    parsed = json.loads(json_example)
    assert set(parsed) == {"Answer", "Explanation"}
