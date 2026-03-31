import os
os.environ["OPENAI_API_KEY"] = ""
import re
import time
import json
import jsonlines
import tiktoken
from pydantic import BaseModel, Field
from typing import Literal

from tqdm import tqdm
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.lib._parsing._completions import type_to_response_format_param

encoding = tiktoken.encoding_for_model('gpt-4o-mini')

sys_prompt = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.
"""

prompt = """
You will evaluate two answers to the same question based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
- **Trustworthiness**: Does the answer provide sufficient detail and align with common knowledge, enhancing its credibility?
- **Depth**: Does the answer provide in-depth analysis or details, rather than just superficial information?
- **Density**: Does the answer contain relevant information without less informative or redundant content?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these criteria.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Trustworthiness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Depth": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Density": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner on the above criteria]"
    }}
}}
"""

class Criterion(BaseModel):
    Winner: Literal["Answer 1", "Answer 2"]
    Explanation: str

class Result(BaseModel):
    Comprehensiveness: Criterion
    Empowerment: Criterion
    Trustworthiness: Criterion
    Depth: Criterion
    Density: Criterion
    Overall_Winner: Criterion = Field(alias="Overall Winner")

result_response_format = type_to_response_format_param(Result)

if __name__ == "__main__":
    with open('../../longervideos/dataset.json', 'r') as f:
        questions = json.load(f)
    our_answer_dir = 'answers-videorag'
    
    # overall comparsion - rag
    base_dir = 'overall_comparison_rag'
    com_answer_dir = [
        'answers-naiverag', 
        'answers-graphrag-local', 
        'answers-graphrag-global',
        'answers-lightrag-hybrid',
    ]
    
    requests = []
    total_token_count = 0
    for _id in questions:
        video_list_name = questions[_id][0]['description']
        video_querys = questions[_id][0]['questions']
        data_path = f"../all_answers/{_id}-{video_list_name}"
        for _com_answer_dir in com_answer_dir:
            our_work_dir = os.path.join(data_path, our_answer_dir)
            com_work_dir = os.path.join(data_path, _com_answer_dir)
            for i in range(len(questions[_id][0]['questions'])):
                # query
                query_id = questions[_id][0]['questions'][i]["id"]
                query = questions[_id][0]['questions'][i]["question"]
                # our answer
                with open(os.path.join(our_work_dir, f'answer_{query_id}.md'), 'r') as f:
                    our_answer = f.read()
                # com answer
                with open(os.path.join(com_work_dir, f'answer_{query_id}.md'), 'r') as f:
                    com_answer = f.read()
                ori_prompt = prompt.format(query=query, answer1=our_answer, answer2=com_answer)
                rev_prompt = prompt.format(query=query, answer1=com_answer, answer2=our_answer)
                
                ori_request_data = {
                    "custom_id": f"{_id}-{video_list_name}++query{query_id}++{our_answer_dir}++{_com_answer_dir}++ori",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": ori_prompt},
                        ],
                        "response_format": result_response_format
                    },
                }
                rev_request_data = {
                    "custom_id": f"{_id}-{video_list_name}++query{query_id}++{_com_answer_dir}++{our_answer_dir}++rev",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": rev_prompt},
                        ],
                        "response_format": result_response_format
                    },
                }
                requests.append(ori_request_data)
                requests.append(rev_request_data)
                
                total_token_count += len(encoding.encode(ori_prompt))
                total_token_count += len(encoding.encode(rev_prompt))
    
    run_time = 5
    os.makedirs(f'batch_requests/{base_dir}', exist_ok=True)
    request_json_file_path = f'batch_requests/{base_dir}/{int(time.time())}.json'
    with jsonlines.open(request_json_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)
    print(f"Batch API requests written to {request_json_file_path}")
    print(f"Price: {total_token_count / 1000000 * 0.075 * run_time}$")
    
    for k in range(run_time):
        client = OpenAI()
        batch_input_file = client.files.create(
            file=open(request_json_file_path, "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"runtime{k}: {request_json_file_path}"},
        )
        print(f"RunTime {k}: Batch {batch.id} has been created.")