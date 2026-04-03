import os
from dotenv import load_dotenv

load_dotenv()

import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from VideoRAG_algorithm.videorag._llm import gemini_embed_and_chat_config
from VideoRAG_algorithm.videorag import VideoRAG

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize VideoRAG referencing the existing working directory
    videorag = VideoRAG(
        llm=gemini_embed_and_chat_config,
        working_dir="./videorag-workdir-activity"
    )
    
    # We don't necessarily need the caption model here if we're just doing traversal
    # but we load it just in case any downstream operation expects it
    videorag.load_caption_model(debug=False)

    print("Starting activity summarization traversal for Human Activity Recognition...")
    response2 = videorag.generate_activity_summary(target_video_name="Human Activity Recognition I3D Demo 720P")
    print("\n--- Final Activity Summary (Human Activity Recognition) ---\n")
    print(response2)
