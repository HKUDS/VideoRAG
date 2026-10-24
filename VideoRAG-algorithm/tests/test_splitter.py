import importlib.util
from pathlib import Path


_splitter_path = Path(__file__).parents[1] / "videorag" / "_splitter.py"
_splitter_spec = importlib.util.spec_from_file_location(
    "videorag_test_splitter", _splitter_path
)
_splitter_module = importlib.util.module_from_spec(_splitter_spec)
_splitter_spec.loader.exec_module(_splitter_module)
SeparatorSplitter = _splitter_module.SeparatorSplitter


def test_overlap_does_not_discard_current_split_tail():
    splitter = SeparatorSplitter(
        separators=[[99]],
        chunk_size=5,
        chunk_overlap=2,
    )

    chunks = splitter.split_tokens([0, 1, 2, 99, 3, 4, 5, 6, 99])

    assert chunks == [[0, 1, 2, 99], [3, 4, 5, 6, 99]]
