import os
from dotenv import load_dotenv

load_dotenv()

import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# All text LLM + embeddings via Google Gemini API (same key):
#   GEMINI_API_KEY or GOOGLE_API_KEY
# Embeddings: gemini-embedding-001 (embedContent), 768-d by default.
# Chat / entity extraction: GEMINI_LLM_MODEL (default gemini-flash-latest), generateContent.
# Optional: GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_DIMENSION
# CUDA: use spawn so caption subprocesses see the GPU.

from VideoRAG_algorithm.videorag._llm import gemini_embed_and_chat_config
from VideoRAG_algorithm.videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    video_paths = [
        'videos/Amritsar Woman Stops Robbers.mp4',
        'videos/Human Activity Recognition I3D Demo 720P.mp4'
    ]
    videorag = VideoRAG(
        llm=gemini_embed_and_chat_config,
        working_dir="./videorag-workdir-activity",
    )
    videorag.insert_video(video_path_list=video_paths)
