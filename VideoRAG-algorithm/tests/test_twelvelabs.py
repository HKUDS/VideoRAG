"""Tests for the optional TwelveLabs (Marengo + Pegasus) integration.

The no-network tests use a small fake TwelveLabs client and only require
``torch``; the live tests are skipped unless ``TWELVELABS_API_KEY`` is set.

Run from ``VideoRAG-algorithm/``::

    pytest tests/test_twelvelabs.py
"""
import importlib.util
import os
import sys
import types

import pytest

# The TwelveLabs helpers only need ``torch`` at runtime, but they live in modules
# whose top-level imports also pull in the heavy local-inference stack (ImageBind,
# transformers, moviepy, PIL). Stub those out so the no-network tests can run in a
# lightweight CI environment, then load the two leaf modules straight from file.
_REPO = os.path.dirname(os.path.dirname(__file__))


def _stub(name, **attrs):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


for _m in ["imagebind", "imagebind.data", "moviepy", "moviepy.video",
           "moviepy.video.io", "transformers", "PIL"]:
    _stub(_m)
_stub("imagebind.models")
_stub("imagebind.models.imagebind_model", ImageBindModel=object, ModalityType=object,
      imagebind_huge=lambda *a, **k: None)
sys.modules["imagebind"].data = sys.modules["imagebind.data"]
sys.modules["imagebind"].models = sys.modules["imagebind.models"]
sys.modules["imagebind.models"].imagebind_model = sys.modules["imagebind.models.imagebind_model"]
_stub("moviepy.video.io.VideoFileClip", VideoFileClip=object)
sys.modules["moviepy.video.io"].VideoFileClip = sys.modules["moviepy.video.io.VideoFileClip"]
sys.modules["transformers"].AutoModel = object
sys.modules["transformers"].AutoTokenizer = object
sys.modules["PIL"].Image = object


def _load(rel_path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_REPO, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_feature = _load("videorag/_videoutil/feature.py", "_tl_feature")
_caption = _load("videorag/_videoutil/caption.py", "_tl_caption")
tl_encode_string_query = _feature.tl_encode_string_query
tl_encode_video_segments = _feature.tl_encode_video_segments
tl_segment_caption = _caption.tl_segment_caption

EMBED_DIM = 512


class _FakeSegment:
    def __init__(self, vector, scope):
        self.float_ = vector
        self.embedding_scope = scope


class _FakeTextEmbedding:
    def __init__(self):
        self.segments = [_FakeSegment([0.1] * EMBED_DIM, "text")]


class _FakeVideoEmbedding:
    def __init__(self):
        self.segments = [
            _FakeSegment([0.2] * EMBED_DIM, "clip"),
            _FakeSegment([0.4] * EMBED_DIM, "video"),
        ]


class _FakeTask:
    id = "task-123"


class _FakeEmbedTasks:
    def create(self, **kwargs):
        assert kwargs["model_name"] == "marengo3.0"
        assert "video_file" in kwargs
        return _FakeTask()

    def wait_for_done(self, task_id):
        assert task_id == "task-123"

    def retrieve(self, task_id, embedding_option):
        assert task_id == "task-123"
        assert embedding_option == "visual"

        class _R:
            video_embedding = _FakeVideoEmbedding()

        return _R()


class _FakeEmbed:
    def __init__(self):
        self.tasks = _FakeEmbedTasks()

    def create(self, model_name, text):
        assert model_name == "marengo3.0"
        assert isinstance(text, str)

        class _R:
            text_embedding = _FakeTextEmbedding()

        return _R()


class _FakeAnalyzeResult:
    data = "A cyclist rides through a city street.\n<|endoftext|>"


class _FakeClient:
    def __init__(self):
        self.embed = _FakeEmbed()

    def analyze(self, model_name, video, prompt, max_tokens):
        assert model_name == "pegasus1.5"
        assert prompt
        return _FakeAnalyzeResult()


def test_tl_encode_string_query_shape():
    out = tl_encode_string_query("a person riding a bicycle", _FakeClient())
    assert tuple(out.shape) == (1, EMBED_DIM)


def test_tl_encode_video_segments_shape():
    out = tl_encode_video_segments(["seg-0.mp4", "seg-1.mp4"], _FakeClient())
    # one vector per input video file, picking the "video"-scope embedding
    assert tuple(out.shape) == (2, EMBED_DIM)


def test_tl_segment_caption_cleans_output():
    caption = tl_segment_caption(_FakeClient(), "https://example.com/clip.mp4", "Describe this clip")
    assert "\n" not in caption
    assert "<|endoftext|>" not in caption
    assert caption.startswith("A cyclist")


@pytest.mark.skipif(
    not os.environ.get("TWELVELABS_API_KEY"),
    reason="TWELVELABS_API_KEY not set; skipping live Marengo call",
)
def test_marengo_text_embedding_live_512_dim():
    from twelvelabs import TwelveLabs

    client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
    out = tl_encode_string_query("a person riding a bicycle", client)
    assert tuple(out.shape) == (1, EMBED_DIM)
