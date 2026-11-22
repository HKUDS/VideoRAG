"""Exercise both retrieval-context serializers without model dependencies."""

import csv
import io
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
UTILS = (
    ROOT / "VideoRAG-algorithm" / "videorag" / "_utils.py",
    ROOT / "Vimo-desktop" / "python_backend" / "videorag" / "_utils.py",
)


def load_serializer(path):
    # These imports are only needed for unrelated tensor and token utilities.
    dependencies = {name: ModuleType(name) for name in ("numpy", "tiktoken", "torch")}
    dependencies["numpy"].ndarray = object
    with patch.dict(sys.modules, dependencies):
        return runpy.run_path(str(path))["list_of_list_to_csv"]


class RetrievalContextCsvTests(unittest.TestCase):
    def test_captions_preserve_exact_text_and_column_boundaries(self):
        rows = [
            ["video_name", "start_time", "end_time", "content"],
            ["board, take 2", "0:01:00", "0:01:30", 'Presenter said "start, then stop".'],
            ["lecture", "0:02:00", "0:02:30", '"原文有引号"\n第二行, new slide'],
            ["lecture", "0:03:00", "0:03:30", "  'keep these quotes'\t  "],
            ["lecture", "0:04:00", "0:04:30", "first\rsecond\r\nthird"],
        ]
        for path in UTILS:
            with self.subTest(backend=path.relative_to(ROOT).parts[0]):
                rendered = load_serializer(path)(rows)
                self.assertFalse(rendered.endswith("\n"))
                parsed = list(csv.reader(io.StringIO(rendered, newline=""), strict=True))
                self.assertEqual(parsed, rows)

    def test_empty_sections_and_numeric_values(self):
        for path in UTILS:
            with self.subTest(backend=path.relative_to(ROOT).parts[0]):
                serialize = load_serializer(path)
                self.assertEqual(serialize([]), "")
                rows = [["lecture", 0, 25.5, ""]]
                parsed = list(csv.reader(io.StringIO(serialize(rows)), strict=True))
                self.assertEqual(parsed, [["lecture", "0", "25.5", ""]])


if __name__ == "__main__":
    unittest.main()
