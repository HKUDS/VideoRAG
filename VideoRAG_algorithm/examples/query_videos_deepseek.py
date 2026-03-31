import os
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# 设置DeepSeek和硅基流动的API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-*******"
os.environ["SILICONFLOW_API_KEY"] = "sk-*******"

from videorag._llm import deepseek_bge_config
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    # 必须的设置
    multiprocessing.set_start_method('spawn')

    # 你想问的问题
    query = '请描述两个视频的主要内容是什么？'
    
    param = QueryParam(mode="videorag")
    # 如果设置为 False，返回的答案会附带视频片段的引用
    param.wo_reference = True

    # 初始化 VideoRAG，并确保工作目录与上一步骤中的一致
    videorag = VideoRAG(llm=deepseek_bge_config, working_dir=f"./videorag-workdir")
    videorag.load_caption_model(debug=False)
    
    # 执行查询
    response = videorag.query(query=query, param=param)
    
    # 打印结果
    print(response)