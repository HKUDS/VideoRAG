import os
import torch
import pickle
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ImageBindModel, ModalityType


def encode_video_segments(video_paths, embedder: ImageBindModel):
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.VISION]
    embeddings = embeddings.cpu()
    return embeddings

def encode_string_query(query:str, embedder: ImageBindModel):
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([query], device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.TEXT]
    embeddings = embeddings.cpu()
    return embeddings


# ---------------------------------------------------------------------------
# TwelveLabs Marengo embeddings (optional, cloud-based alternative to ImageBind)
#
# Marengo produces video-segment and text-query embeddings in a *shared* 512-d
# space, so it is a drop-in replacement for the ImageBind functions above: video
# clips and text queries can be compared directly with cosine similarity. Using
# it removes the local GPU / ImageBind checkpoint requirement for visual
# retrieval. Enabled by selecting ``TwelveLabsVideoSegmentStorage`` as the video
# segment feature store (see ``videorag/_storage/vdb_twelvelabs.py``).
# ---------------------------------------------------------------------------
def tl_encode_video_segments(video_paths, client, model_name: str = "marengo3.0"):
    """Embed each video-segment file with Marengo.

    Returns a ``torch.Tensor`` of shape ``(len(video_paths), embedding_dim)``,
    matching ``encode_video_segments`` so it can feed the same vector DB.
    """
    embeddings = []
    for video_path in video_paths:
        task = client.embed.tasks.create(
            model_name=model_name,
            video_file=video_path,
            video_embedding_scope=["clip", "video"],
        )
        client.embed.tasks.wait_for_done(task_id=task.id)
        result = client.embed.tasks.retrieve(task_id=task.id, embedding_option="visual")
        segments = result.video_embedding.segments
        # One vector per segment file: use the whole-video ("video" scope) vector
        # when present, otherwise average the clip vectors.
        video_scope = [s.float_ for s in segments if s.embedding_scope == "video"]
        vectors = video_scope if video_scope else [s.float_ for s in segments]
        embeddings.append(torch.tensor(vectors, dtype=torch.float32).mean(dim=0))
    return torch.stack(embeddings, dim=0)


def tl_encode_string_query(query: str, client, model_name: str = "marengo3.0"):
    """Embed a text query with Marengo into the shared video/text space.

    Returns a ``torch.Tensor`` of shape ``(1, embedding_dim)``, matching
    ``encode_string_query``.
    """
    result = client.embed.create(model_name=model_name, text=query)
    vector = result.text_embedding.segments[0].float_
    return torch.tensor([vector], dtype=torch.float32)