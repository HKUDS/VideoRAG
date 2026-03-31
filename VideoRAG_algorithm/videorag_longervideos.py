import os
import json
import logging
import warnings
import multiprocessing
import sys

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

import argparse
parser = argparse.ArgumentParser(description="Set sub-category and CUDA device.")
parser.add_argument('--collection', type=str, default='4-rag-lecture')
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
sub_category = args.collection

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ["OPENAI_API_KEY"] = ""

from videorag._llm import *
from videorag.videorag import VideoRAG, QueryParam

longervideos_llm_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM (we utilize gpt-4o-mini for all experiments)   
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini", 
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    ## learn
    video_base_path = f'./longervideos/{sub_category}/videos/'
    video_files = sorted(os.listdir(video_base_path))
    video_paths = [os.path.join(video_base_path, f) for f in video_files]
    videorag = VideoRAG(llm=longervideos_llm_config, working_dir=f"./longervideos/videorag-workdir/{sub_category}")    
    videorag.insert_video(video_path_list=video_paths)
    
    ## inference
    with open(f'./longervideos/dataset.json', 'r') as f:
        longervideos = json.load(f)
    
    videorag = VideoRAG(llm=longervideos_llm_config, working_dir=f"./longervideos/videorag-workdir/{sub_category}")        
    videorag.load_caption_model(debug=False)
    
    answer_folder = f'./longervideos/videorag-answers/{sub_category}'
    os.makedirs(answer_folder, exist_ok=True)
    
    collection_id = sub_category.split('-')[0]
    querys = longervideos[collection_id][0]['questions']
    for i in range(len(querys)):
        query_id = querys[i]['id']
        query = querys[i]['question']
        param = QueryParam(mode="videorag")
        param.wo_reference = True
        print("Query: ", query)
        
        response = videorag.query(query=query, param=param)
        print(response)
        with open(os.path.join(answer_folder, f'answer_{query_id}.md'), 'w') as f:
            f.write(response)
