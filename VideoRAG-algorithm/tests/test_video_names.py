import importlib.util
from pathlib import Path


_paths_path = Path(__file__).parents[1] / "videorag" / "_videoutil" / "paths.py"
_paths_spec = importlib.util.spec_from_file_location("videorag_test_paths", _paths_path)
_paths_module = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(_paths_module)
video_name_from_path = _paths_module.video_name_from_path


def test_video_names_preserve_dots_before_final_extension(tmp_path):
    first = tmp_path / "lecture.part1.mp4"
    second = tmp_path / "lecture.part2.mp4"

    assert video_name_from_path(first) == "lecture.part1"
    assert video_name_from_path(second) == "lecture.part2"
    assert video_name_from_path(first) != video_name_from_path(second)
