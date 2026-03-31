import os
os.environ["OPENAI_API_KEY"] = ""
import re
import time
import json
import jsonlines
import tiktoken
import itertools
from pydantic import BaseModel, Field
from typing import Literal

from tqdm import tqdm
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.lib._parsing._completions import type_to_response_format_param

encoding = tiktoken.encoding_for_model('gpt-4o-mini')

sys_prompt = """
---Role---
You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.
"""

prompt = """
You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
- **Trustworthiness**: Does the answer provide sufficient detail and align with common knowledge, enhancing its credibility?
- **Depth**: Does the answer provide in-depth analysis or details, rather than just superficial information?
- **Density**: Does the answer contain relevant information without less informative or redundant content?

For the evaluated answer labeled "Evaluation Answer," assign a score from 1 to 5 for each criterion compared to the baseline answer labeled "Baseline Answer." Then, assign an overall score based on these criteria.
The evaluation scores are defined as follows:
- 1: Strongly worse than the baseline answer
- 2: Weakly worse than the baseline answer
- 3: Moderate compared to the baseline answer
- 4: Weakly better than the baseline answer
- 5: Strongly better than the baseline answer


Here is the question:
{query}

Here are the answers:

**Baseline Answer:**
{baseline_answer}

**Evaluation Answer:**
{evaluation_answer}


Evaluate the answer using the criteria listed above and provide detailed explanations for the scores.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Trustworthiness": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Depth": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Density": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }}
    "Overall Score": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }}
}}
"""

class Criterion(BaseModel):
    Score: int
    Explanation: str

class Result(BaseModel):
    Comprehensiveness: Criterion
    Empowerment: Criterion
    Trustworthiness: Criterion
    Depth: Criterion
    Density: Criterion
    Overall_Score: Criterion = Field(alias="Overall Score")

result_response_format = type_to_response_format_param(Result)

if __name__ == "__main__":
    with open('../../longervideos/dataset.json', 'r') as f:
        questions = json.load(f)
        
    baseline_answer_dir = 'answers-naiverag'
    base_dir = 'overall_comparison_video_understanding'
    evaluation_answer_dir = [ 
        'answers-videorag',
        'answers-notebooklm',
        'answers-llamavid',
        'answers-videoagent'
    ]
    
    requests = []
    total_token_count = 0
    for _id in questions:
        video_list_name = questions[_id][0]['description']
        video_querys = questions[_id][0]['questions']
        data_path = f"../all_answers/{_id}-{video_list_name}"
        for _evaluation_answer_dir in evaluation_answer_dir:
            baseline_work_dir = os.path.join(data_path, baseline_answer_dir)
            evaluation_work_dir = os.path.join(data_path, _evaluation_answer_dir)
            for i in range(len(questions[_id][0]['questions'])):
                # query
                query_id = questions[_id][0]['questions'][i]["id"]
                query = questions[_id][0]['questions'][i]["question"]
                # baseline answer
                with open(os.path.join(baseline_work_dir, f'answer_{query_id}.md'), 'r') as f:
                    baseline_answer = f.read()
                # evaluation answer
                with open(os.path.join(evaluation_work_dir, f'answer_{query_id}.md'), 'r') as f:
                    evaluation_answer = f.read()
                request_prompt = prompt.format(query=query, baseline_answer=baseline_answer, evaluation_answer=evaluation_answer)
                
                request_data = {
                    "custom_id": f"{_id}-{video_list_name}++query{query_id}++base++{baseline_answer_dir}++evaluate++{_evaluation_answer_dir}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": request_prompt},
                        ],
                        "response_format": result_response_format
                    },
                }
                requests.append(request_data)
                total_token_count += len(encoding.encode(request_prompt))
    
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
            metadata={"description": f"runtime{k} - a very nice and successful eval job: {request_json_file_path}"},
        )
        print(f"RunTime {k}: Batch {batch.id} has been created.")