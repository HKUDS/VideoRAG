"""Cloud-free regression checks for the desktop splitter and ASR routing.

From Vimo-desktop/python_backend (MoviePy 1.x and ffmpeg are required):
    python -m unittest discover -s tests -p test_silent_video.py -v
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from moviepy.config import get_setting


def load_media_modules():
    # Import the production files without initializing unrelated GPU/LLM models.
    # MoviePy and FFmpeg are real; only the hosted recognition provider is replaced.
    root = Path(__file__).resolve().parents[1] / "videorag" / "_videoutil"
    logger = types.ModuleType("_media_test._utils")
    logger.logger = logging.getLogger(__name__)
    provider = types.ModuleType("dashscope.audio.asr")
    provider.Recognition = Mock(side_effect=AssertionError("no hosted calls in tests"))
    with patch.dict(sys.modules, {
        "_media_test._utils": logger,
        "dashscope": types.ModuleType("dashscope"),
        "dashscope.audio": types.ModuleType("dashscope.audio"),
        "dashscope.audio.asr": provider,
    }):
        modules = []
        for name in ("split", "asr"):
            spec = importlib.util.spec_from_file_location(f"_media_test._videoutil.{name}", root / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


split, asr = load_media_modules()


class DesktopSilentVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(self.loop.close)
        self.addCleanup(asyncio.set_event_loop, None)

    def make_video(self, with_audio):
        video = self.work / "fixture.mp4"
        args = [get_setting("FFMPEG_BINARY"), "-v", "error", "-nostdin",
                "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10:d=2"]
        if with_audio:
            args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2", "-c:a", "aac"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)]
        subprocess.run(args, check=True, capture_output=True, timeout=20)
        return video

    def test_silent_mp4_keeps_timing_and_frames_without_asr_or_credentials(self):
        video = self.make_video(with_audio=False)
        names, times = split.split_video(str(video), self.tmp.name, 10, 2)
        self.assertEqual(set(names), {"0"})
        self.assertEqual(times["0"]["timestamp"], (0, 2))
        self.assertEqual(list(times["0"]["frame_times"]), [0, 1])
        self.assertFalse(times["0"]["has_audio"])
        self.assertEqual(list((self.work / "_cache" / "fixture").glob("*.mp3")), [])
        with patch.object(asr, "speech_to_text_async", new_callable=AsyncMock) as recognize:
            result = asr.speech_to_text("fixture", self.tmp.name, names, "mp3", {}, segment_times_info=times)
        recognize.assert_not_called()
        self.assertEqual(result, {"0": ""})

    def test_mp4_with_audio_still_exports_and_dispatches_existing_asr(self):
        video = self.make_video(with_audio=True)
        names, times = split.split_video(str(video), self.tmp.name, 10, 2)
        self.assertTrue(times["0"]["has_audio"])
        clip = self.work / "_cache" / "fixture" / f"{names['0']}.mp3"
        self.assertGreater(clip.stat().st_size, 0)
        config = {"ali_dashscope_api_key": "fixture-not-a-real-key"}
        with patch.object(asr, "speech_to_text_async", new_callable=AsyncMock) as recognize:
            recognize.return_value = {"0": "Existing transcript"}
            result = asr.speech_to_text("fixture", self.tmp.name, names, "mp3", config, segment_times_info=times)
        recognize.assert_awaited_once_with("fixture", self.tmp.name, names, "mp3", config)
        self.assertEqual(result, {"0": "Existing transcript"})

    def test_only_explicit_no_audio_is_skipped(self):
        names = {"0": "silent", "1": "expected", "2": "unknown"}
        times = {"0": {"has_audio": False}, "1": {"has_audio": True}, "2": {"timestamp": (20, 30)}}
        config = {"ali_dashscope_api_key": "fixture-not-a-real-key"}
        with patch.object(asr, "speech_to_text_async", new_callable=AsyncMock) as recognize:
            recognize.return_value = {"2": "Existing transcript"}
            result = asr.speech_to_text("fixture", self.tmp.name, names, "mp3", config, segment_times_info=times)
        recognize.assert_awaited_once_with("fixture", self.tmp.name, {"1": "expected", "2": "unknown"}, "mp3", config)
        # An expected segment omitted by the provider is not disguised as silence.
        self.assertEqual(result, {"0": "", "2": "Existing transcript"})

    def test_no_metadata_keeps_existing_required_credentials(self):
        with self.assertRaisesRegex(ValueError, "ali_dashscope_api_key"):
            asr.speech_to_text("fixture", self.tmp.name, {"0": "expected"}, "mp3", {})


if __name__ == "__main__":
    unittest.main()
