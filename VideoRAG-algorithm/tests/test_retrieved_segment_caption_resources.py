import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "videorag"
    / "_videoutil"
    / "caption.py"
)


def load_caption_module(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)

    numpy = types.ModuleType("numpy")
    numpy.linspace = lambda start, end, count, endpoint=False: [start, end]

    pil = types.ModuleType("PIL")
    pil.__path__ = []
    pil.Image = types.ModuleType("PIL.Image")

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda values, *args, **kwargs: values

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = object
    transformers.AutoTokenizer = object

    moviepy = types.ModuleType("moviepy")
    moviepy.__path__ = []
    video = types.ModuleType("moviepy.video")
    video.__path__ = []
    io = types.ModuleType("moviepy.video.io")
    io.__path__ = []
    video_file_clip = types.ModuleType("moviepy.video.io.VideoFileClip")
    video_file_clip.VideoFileClip = object

    for name, module in (
        ("torch", torch),
        ("numpy", numpy),
        ("PIL", pil),
        ("PIL.Image", pil.Image),
        ("tqdm", tqdm),
        ("transformers", transformers),
        ("moviepy", moviepy),
        ("moviepy.video", video),
        ("moviepy.video.io", io),
        ("moviepy.video.io.VideoFileClip", video_file_clip),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "caption_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("caption_raises", [False, True])
def test_retrieved_segment_caption_closes_video_on_success_and_error(
    monkeypatch, caption_raises
):
    module = load_caption_module(monkeypatch)
    videos = []

    class FakeVideo:
        def __init__(self, path):
            self.path = path
            self.closed = False
            videos.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.closed = True

    module.VideoFileClip = FakeVideo
    module.encode_video = lambda video, frame_times: ["synthetic-frame"]

    class FakeCaptionModel:
        def chat(self, **kwargs):
            if caption_raises:
                raise RuntimeError("synthetic caption failure")
            return "synthetic caption\n<|endoftext|>"

    video_path_db = SimpleNamespace(_data={"video": "synthetic-video.mp4"})
    video_segments = SimpleNamespace(
        _data={
            "video": {
                "0": {"time": "0-1", "transcript": "synthetic transcript"}
            }
        }
    )

    if caption_raises:
        with pytest.raises(RuntimeError, match="synthetic caption failure"):
            module.retrieved_segment_caption(
                FakeCaptionModel(),
                object(),
                "synthetic question",
                ["video_0"],
                video_path_db,
                video_segments,
                num_sampled_frames=2,
            )
    else:
        captions = module.retrieved_segment_caption(
            FakeCaptionModel(),
            object(),
            "synthetic question",
            ["video_0"],
            video_path_db,
            video_segments,
            num_sampled_frames=2,
        )
        assert captions["video_0"].startswith("Caption:\nsynthetic caption")

    assert len(videos) == 1
    assert videos[0].closed
