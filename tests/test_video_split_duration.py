"""Exercise both split implementations without importing model-backed packages."""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ColorClip
from moviepy.video.io.VideoFileClip import VideoFileClip


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["VideoRAG-algorithm", "Vimo-desktop/python_backend"])
def split_module(request, monkeypatch):
    package_dir = ROOT / request.param / "videorag"
    name = "_video_duration_test"
    package = ModuleType(name)
    package.__path__ = [str(package_dir)]
    video_util = ModuleType(f"{name}._videoutil")
    video_util.__path__ = [str(package_dir / "_videoutil")]
    utils = ModuleType(f"{name}._utils")
    utils.logger = logging.getLogger(name)
    spec = importlib.util.spec_from_file_location(
        f"{name}._videoutil.split", package_dir / "_videoutil" / "split.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Only the unused package bootstrap and logger are replaced. Execute the
    # actual split module with real NumPy and MoviePy, not copied function code.
    with patch.dict(
        sys.modules,
        {
            name: package,
            f"{name}._videoutil": video_util,
            f"{name}._utils": utils,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "tqdm", lambda values, **kwargs: values)
    return module


def _mock_video(module, monkeypatch, duration):
    video = MagicMock()
    video.duration = duration
    video.__enter__.return_value = video
    clips = []

    def subclip(start, end):
        clip = MagicMock()
        clip.duration = end - start
        clips.append(clip)
        return clip

    video.subclip.side_effect = subclip
    monkeypatch.setattr(module, "VideoFileClip", lambda path: video)
    return video, clips


@pytest.mark.parametrize(
    "duration, expected",
    [
        (0.25, [(0, 0.25)]),
        (1.25, [(0, 1.25)]),
        (29.75, [(0, 29.75)]),
        (30.25, [(0, 30.25)]),
        (34.75, [(0, 34.75)]),
        (35.25, [(0, 30), (30, 35.25)]),
        (60.25, [(0, 30), (30, 60.25)]),
        (65.25, [(0, 30), (30, 60), (60, 65.25)]),
    ],
)
def test_fractional_duration_is_fully_covered(
    split_module, monkeypatch, tmp_path, duration, expected
):
    video, clips = _mock_video(split_module, monkeypatch, duration)

    names, segments = split_module.split_video("clip.mp4", str(tmp_path), 30, 4)

    assert [row["timestamp"] for row in segments.values()] == expected
    assert video.subclip.call_args_list == [call(start, end) for start, end in expected]
    assert list(names) == list(segments) == [str(i) for i in range(len(expected))]
    for (index, row), clip in zip(segments.items(), clips):
        start, end = row["timestamp"]
        assert start < end
        assert len(row["frame_times"]) == 4
        assert np.all(row["frame_times"] >= start)
        assert np.all(row["frame_times"] < end)
        assert float(names[index].split("-")[-1]) == end
        clip.audio.write_audiofile.assert_called_once()
    video.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "duration, expected",
    [
        (30.0, [(0, 30)]),
        (35.0, [(0, 30), (30, 35)]),
        (60.0, [(0, 30), (30, 60)]),
        (64.0, [(0, 30), (30, 64)]),
    ],
)
def test_whole_second_windows_and_names_are_unchanged(
    split_module, monkeypatch, tmp_path, duration, expected
):
    _mock_video(split_module, monkeypatch, duration)

    names, segments = split_module.split_video("clip.mp4", str(tmp_path), 30, 4)

    assert [row["timestamp"] for row in segments.values()] == expected
    for index, (start, end) in enumerate(expected):
        assert names[str(index)].endswith(f"-{index}-{start}-{end}")


@pytest.mark.parametrize("duration", [0.5, 1.25])
def test_real_media_keeps_the_decoded_duration(split_module, tmp_path, duration):
    source = tmp_path / "synthetic.mp4"
    audio = AudioClip(
        lambda t: 0.2 * np.sin(2 * np.pi * 440 * t), duration=duration, fps=16000
    )
    clip = ColorClip((16, 16), color=(20, 40, 60), duration=duration).set_audio(audio)
    try:
        clip.write_videofile(
            str(source),
            fps=8,
            codec="libx264",
            audio_codec="aac",
            audio_fps=16000,
            temp_audiofile=str(tmp_path / "temporary.m4a"),
            verbose=False,
            logger=None,
        )
    finally:
        clip.close()
        audio.close()
    with VideoFileClip(str(source)) as decoded:
        decoded_duration = decoded.duration

    names, segments = split_module.split_video(str(source), str(tmp_path), 30, 4)

    assert len(segments) == 1
    assert segments["0"]["timestamp"] == (0, decoded_duration)
    extracted = tmp_path / "_cache" / "synthetic" / f"{names['0']}.mp3"
    assert extracted.is_file() and extracted.stat().st_size > 0
    with AudioFileClip(str(extracted)) as decoded_audio:
        # MP3 padding can extend the stream, but its duration must cover the tail.
        assert decoded_audio.duration >= decoded_duration - 0.05

    errors = MagicMock()
    parameters = inspect.signature(split_module.saving_video_segments).parameters
    kwargs = {"error_queue": errors} if "error_queue" in parameters else {}
    split_module.saving_video_segments(
        "synthetic", str(source), str(tmp_path), names, segments, **kwargs
    )
    errors.put.assert_not_called()
    saved = extracted.with_suffix(".mp4")
    assert saved.is_file() and saved.stat().st_size > 0
    with VideoFileClip(str(saved)) as decoded_segment:
        assert decoded_segment.duration == pytest.approx(decoded_duration, abs=0.05)
