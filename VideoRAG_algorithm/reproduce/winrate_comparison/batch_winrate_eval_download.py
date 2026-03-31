import os
os.environ["OPENAI_API_KEY"] = ""
import re
import time
import json
import jsonlines
import tiktoken

from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

def obtain_ouput_file_id(batches):
    for batch in batches:
        print(client.batches.retrieve(batch))
        print(client.batches.retrieve(batch).output_file_id)

def download_result(result_files, base_dir):
    for _file in result_files:
        content = client.files.content(_file).content
        with open(f"batch_requests/{base_dir}/{_file}.temp", "wb") as f:
            f.write(content)
        results = []
        with open(f"batch_requests/{base_dir}/{_file}.temp", 'r') as f:
            for line in tqdm(f):
                json_object = json.loads(line.strip())
                results.append(json_object)
        with open(f"batch_requests/{base_dir}/{_file}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
        os.remove(f"batch_requests/{base_dir}/{_file}.temp")

# ================================

# Please enter the relevant batch ID here to obtain the output file ID.
batches = [
    '',
    '',
    '',
    '',
    ''
]
obtain_ouput_file_id(batches)

# Second Step: Please enter the output file ID below to download the output files.
# result_files = [
#     '',
#     '',
#     '',
#     '',
#     ''
# ]
# download_result(result_files, base_dir='overall_comparison_rag')