import os
from dotenv import load_dotenv

load_dotenv()

import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Same as construct_graph.py: GEMINI_API_KEY or GOOGLE_API_KEY; optional GEMINI_LLM_MODEL.
# Must use the same working_dir you used when building the graph.

from VideoRAG_algorithm.videorag._llm import gemini_embed_and_chat_config
from VideoRAG_algorithm.videorag import VideoRAG, QueryParam


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    query = "Summarize the list of events that happened in the video"
    param = QueryParam(mode="videorag")
    # If param.wo_reference is False, VideoRAG adds references to video clips in the response.
    param.wo_reference = True

    videorag = VideoRAG(
        llm=gemini_embed_and_chat_config,
        working_dir="./videorag-workdir-gemini-embed",
    )
    videorag.load_caption_model(debug=False)
    response = videorag.query(query=query, param=param)
    print(response)
