# VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos

<a href='https://arxiv.org/abs/2502.01549'><img src='https://img.shields.io/badge/arXiv-2502.01549-b31b1b'></a>
<a href='https://github.com/HKUDS/VideoRAG/issues/1'><img src='https://img.shields.io/badge/群聊-wechat-green'></a>
<a href='https://discord.gg/ZzU55kz3'><img src='https://discordapp.com/api/guilds/1296348098003734629/widget.png?style=shield'></a>


<img src='VideoRAG_cover.png' />

 This is the PyTorch implementation for VideoRAG proposed in this paper:

 >**VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos**  
 >Xubin Ren*, Lingrui Xu*, Long Xia, Shuaiqiang Wang, Dawei Yin, Chao Huang†

\* denotes equal contribution.
† denotes corresponding author

 In this paper, we proposed a retrieval-augmented generation framework specifically designed for processing and understanding **extremely long-context videos**.

## VideoRAG Framework

<p align="center">
<img src="VideoRAG.png" alt="VideoRAG" />
</p>

VideoRAG introduces a novel dual-channel architecture that synergistically combines graph-driven textual knowledge grounding for modeling cross-video semantic relationships with hierarchical multimodal context encoding to preserve spatiotemporal visual patterns, enabling unbounded-length video understanding through dynamically constructed knowledge graphs that maintain semantic coherence across multi-video contexts while optimizing retrieval efficiency via adaptive multimodal fusion mechanisms.

💻 **Efficient Extreme Long-Context Video Processing**
- Leveraging a Single NVIDIA RTX 3090 GPU (24G) to comprehend Hundreds of Hours of video content 💪

🗃️ **Structured Video Knowledge Indexing**
- Multi-Modal Knowledge Indexing Framework distills hundreds of hours of video into a concise, structured knowledge graph 🗂️

🔍 **Multi-Modal Retrieval for Comprehensive Responses**
- Multi-Modal Retrieval Paradigm aligns textual semantics and visual content to identify the most relevant video for comprehensive responses 💬

📚 **The New Established LongerVideos Benchmark**
- The new established LongerVideos Benchmark features over 160 Videos totaling 134+ Hours across lectures, documentaries, and entertainment 🎬

## Installation

To utilize VideoRAG, please first create a conda environment with the following commands:
```bash
conda create --name videorag python=3.11
conda activate videorag

pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install accelerate==0.30.1
pip install bitsandbytes==0.43.1
pip install moviepy==1.0.3
pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
pip install timm ftfy regex einops fvcore eva-decord==0.6.1 iopath matplotlib types-regex cartopy
pip install ctranslate2==4.4.0 faster_whisper==1.0.3 neo4j hnswlib xxhash nano-vectordb
pip install transformers==4.37.1
pip install tiktoken openai tenacity

# Install ImageBind using the provided code in this repository, where we have removed the requirements.txt to avoid environment conflicts.
cd ImageBind
pip install .
```

Then, please download the necessary checkpoints in **the repository's root folder** for MiniCPM-V, Whisper, and ImageBind as follows:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# minicpm-v
git lfs clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4

# whisper
git lfs clone https://huggingface.co/Systran/faster-distil-whisper-large-v3

# imagebind
mkdir .checkpoints
cd .checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
cd ../
```

Your final directory structure after downloading all checkpoints should look like this:
```shell
VideoRAG
├── .checkpoints
├── faster-distil-whisper-large-v3
├── ImageBind
├── LICENSE
├── longervideos
├── MiniCPM-V-2_6-int4
├── README.md
├── reproduce
├── videorag
├── VideoRAG_cover.png
└── VideoRAG.png
```

## Quick Start

VideoRAG is capable of extracting knowledge from multiple videos and answering queries based on those videos. Now, try VideoRAG with your own videos 🤗.

> [!NOTE]
> Currently, VideoRAG has only been tested in an English environment. To process videos in multiple languages, it is recommended to modify the  ```WhisperModel``` in [asr.py](VideoRAG/videorag/_videoutil/asr.py). For more details, please refer to [faster-whisper](https://github.com/systran/faster-whisper).

**At first**, let the VideoRAG extract and indexing the knowledge from given videos (Only one GPU with 24GB of memory is sufficient, such as the RTX 3090):
```python
import os
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Please enter your openai key
os.environ["OPENAI_API_KEY"] = ""

from videorag._llm import *
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Please enter your video file path in this list; there is no limit on the length.
    # Here is an example; you can use your own videos instead.
    video_paths = [
        'movies/Iron-Man.mp4',
        'movies/Spider-Man.mkv',
    ]
    videorag = VideoRAG(cheap_model_func=gpt_4o_mini_complete, best_model_func=gpt_4o_mini_complete, working_dir=f"./videorag-workdir")
    videorag.insert_video(video_path_list=video_paths)
```

**Then**, ask any questions about the videos! Here is an exmaple:
```python
import os
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Please enter your openai key
os.environ["OPENAI_API_KEY"] = ""

from videorag._llm import *
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    query = 'What is the relationship between Iron Man and Spider-Man? How do they meet, and how does Iron Man help Spider-Man?'
    param = QueryParam(mode="videorag")
    # if param.wo_reference = False, VideoRAG will add reference to video clips in the response
    param.wo_reference = True

    videorag = VideoRAG(cheap_model_func=gpt_4o_mini_complete, best_model_func=gpt_4o_mini_complete, working_dir=f"./videorag-workdir")
    videorag.load_caption_model(debug=False)
    response = videorag.query(query=query, param=param)
    print(response)
```

## Evaluation

### LongerVideos
We constructed the LongerVideos benchmark to evaluate the model's performance in comprehending multiple long-context videos and answering open-ended queries. All the videos are open-access videos on YouTube, and we record the URLs of the collections of videos as well as the corresponding queries in the [JSON](longervideos/dataset.json) file.

| Video Type       | #video list | #video | #query | #avg. queries per list | #overall duration      |
|------------------|------------:|-------:|-------:|-----------------------:|-------------------------|
| **Lecture**      | 12          | 135    | 376    | 31.3                   | ~ 64.3 hours           |
| **Documentary**  | 5           | 12     | 114    | 22.8                   | ~ 28.5 hours           |
| **Entertainment**| 5           | 17     | 112    | 22.4                   | ~ 41.9 hours           |
| **All**          | 22          | 164    | 602    | 27.4                   | ~ 134.6 hours          |

### Process LongerVideos with VideoRAG

First, we need to download all the videos in 720p resolution using the `yt-dlp` library. You can use the following commands to download all the videos in LongerVideos:

```shell
cd longervideos
python prepare_videos.py # create collection folders
sh download.sh # downloading videos
```

Then, you can run the following example command to process and answer queries for LongerVideos with VideoRAG:

```shell
# Please enter your openai_key in line 18 at first
python videorag_experiment.py --collection 4-rag-lecture --cuda 0
```

### Evaluation

We conduct win-rate comparisons as well as quantitative comparisons with RAG-based baselines and long-context video understanding methods separately. **NaiveRAG, GraphRAG and LightRAG** are implemented using the `nano-graphrag` library, which is consistent with our VideoRAG, ensuring a fair comparison.

In this part, we directly provided the **answers from all the methods** (including VideoRAG) as well as the evaluation codes for experiment reproduction. Please utilize the following commands to download the answers:

```shell
cd reproduce
wget https://archive.org/download/videorag/all_answers.zip
unzip all_answers
```

#### Win-Rate Comparison

We conduct the win-rate comparison with RAG-based baselines. To reproduce the results, please follow these steps:

```shell
cd reproduce/winrate_comparison

# First Step: Upload the batch request to OpenAI (remember to enter your key in the file, same for the following steps).
python batch_winrate_eval_upload.py

# Second Step: Download the results. Please enter the batch ID and then the output file ID in the file. Generally, you need to run this twice: first to obtain the output file ID, and then to download it.
python batch_winrate_eval_download.py

# Third Step: Parsing the results. Please the output file ID in the file.
python batch_winrate_eval_download.py

# Fourth Step: Calculate the results. Please enter the parsed result file name in the file.
python batch_winrate_eval_calculate.py

```

#### Quantitative Comparison

We conduct a quantitative comparison, which extends the win-rate comparison by assigning a 5-point score to long-context video understanding methods. We use the answers from NaiveRAG as the baseline response for scoring each query. To reproduce the results, please follow these steps:

```shell
cd reproduce/quantitative_comparison

# First Step: Upload the batch request to OpenAI (remember to enter your key in the file, same for the following steps).
python batch_winrate_quant_upload.py

# Second Step: Download the results. Please enter the batch ID and then the output file ID in the file. Generally, you need to run this twice: first to obtain the output file ID, and then to download it.
python batch_winrate_quant_download.py

# Third Step: Parsing the results. Please the output file ID in the file.
python batch_winrate_quant_download.py

# Fourth Step: Calculate the results. Please enter the parsed result file name in the file.
python batch_winrate_quant_calculate.py
```

### Ollama Support

This project also supports ollama.  To use, edit the ollama_config in _llm.py.
Adjust the paramters of the models being used

```
ollama_config = LLMConfig(
    embedding_func_raw = ollama_embedding,
    embedding_model_name = "nomic-embed-text",
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = ollama_complete ,
    best_model_name = "gemma2:latest", # need to be a solid instruct model
    best_model_max_token_size = 32768,
    best_model_max_async  = 1,
    cheap_model_func_raw = ollama_mini_complete,
    cheap_model_name = "olmo2",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1)
```
And specify the config when creating your VideoRag instance

### Jupyter Notebook

To  test the solution on a single video, just load the notebook in the notebook folder and
update the paramters to fit your situation.

YouTube video for example can be downloaded as

```
yt-dlp -o "%(id)s.%(ext)s" -S "res:720" https://www.youtube.com/live/DPa2iRgzadM?si=8cf8WbYtqiglrwtN  -P .
```


## Citation
If you find this work is helpful to your research, please consider citing our paper:
```bibtex
@article{VideoRAG,
  title={VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos},
  author={Ren, Xubin and Xu, Lingrui and Xia, Long and Wang, Shuaiqiang and Yin, Dawei and Huang, Chao},
  journal={arXiv preprint arXiv:2502.01549},
  year={2025}
}
```

**Thank you for your interest in our work!**

### Acknowledgement
You may refer to related work that serves as foundations for our framework and code repository, 
[nano-graphrag](https://github.com/gusye1234/nano-graphrag) and [LightRAG](https://github.com/HKUDS/LightRAG). Thanks for their wonderful works.
