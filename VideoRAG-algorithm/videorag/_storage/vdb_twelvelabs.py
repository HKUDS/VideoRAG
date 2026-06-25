import os
from dataclasses import dataclass

import numpy as np
import torch
from nano_vectordb import NanoVectorDB
from tqdm import tqdm

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import tl_encode_video_segments, tl_encode_string_query


@dataclass
class TwelveLabsVideoSegmentStorage(BaseVectorStorage):
    """Visual-retrieval index backed by TwelveLabs Marengo embeddings.

    Drop-in replacement for ``NanoVectorDBVideoSegmentStorage``: video segments
    and text queries are embedded into the same Marengo space, so retrieval is
    plain cosine similarity over a ``nano-vectordb`` index, exactly like the
    ImageBind path. The difference is that embedding happens via the TwelveLabs
    API instead of a local ImageBind checkpoint, so visual retrieval no longer
    needs a local GPU.

    Enable it (opt-in, non-breaking) by overriding the storage class on
    ``VideoRAG`` and setting the Marengo embedding dim (512)::

        from videorag._storage import TwelveLabsVideoSegmentStorage

        videorag = VideoRAG(
            llm=openai_4o_mini_config,
            vs_vector_db_storage_cls=TwelveLabsVideoSegmentStorage,
            video_embedding_dim=512,
            working_dir="./videorag-workdir",
        )

    Requires ``pip install twelvelabs`` and the ``TWELVELABS_API_KEY`` env var
    (grab a free key at https://twelvelabs.io).
    """

    segment_retrieval_top_k: float = 2
    embedding_model_name: str = "marengo3.0"

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
        self.embedding_model_name = self.global_config.get(
            "tl_embedding_model_name", self.embedding_model_name
        )
        self._tl_client = self._get_tl_client()

    def _get_tl_client(self):
        from twelvelabs import TwelveLabs

        api_key = os.environ.get("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TWELVELABS_API_KEY is not set. Get a free key at https://twelvelabs.io"
            )
        return TwelveLabs(api_key=api_key)

    async def upsert(self, video_name, segment_index2name, video_output_format):
        logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
        if not len(segment_index2name):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data, video_paths = [], []
        cache_path = os.path.join(self.global_config["working_dir"], "_cache", video_name)
        index_list = list(segment_index2name.keys())
        for index in index_list:
            list_data.append(
                {
                    "__id__": f"{video_name}_{index}",
                    "__video_name__": video_name,
                    "__index__": index,
                }
            )
            segment_name = segment_index2name[index]
            video_file = os.path.join(cache_path, f"{segment_name}.{video_output_format}")
            video_paths.append(video_file)
        embeddings = []
        for video_path in tqdm(video_paths, desc=f"Encoding Video Segments {video_name}"):
            batch_embeddings = tl_encode_video_segments(
                [video_path], self._tl_client, self.embedding_model_name
            )
            embeddings.append(batch_embeddings)
        embeddings = torch.concat(embeddings, dim=0).numpy()
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str):
        embedding = tl_encode_string_query(
            query, self._tl_client, self.embedding_model_name
        )
        embedding = embedding[0].numpy()
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=-1,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()
