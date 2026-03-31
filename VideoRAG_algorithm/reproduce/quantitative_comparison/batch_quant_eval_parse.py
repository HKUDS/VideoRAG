import os
os.environ["OPENAI_API_KEY"] = ""
import time
import json
import threading
from tqdm import tqdm
from openai import OpenAI
from setproctitle import setproctitle

base_dir = 'overall_comparison_video_understanding'
# The JSON file contains the batch of requests created when the batch request was uploaded.
request_file = ''
# Please enter the output file ID below, which corresponds to the downloaded output files.
result_files = [
    '',
    '',
    '',
    '',
    ''
]

setproctitle(f"parse-result-{base_dir}")
print(f"Start parsing result files in {base_dir}...")

def check_response_valid(data):
    valid_keys = ['Comprehensiveness', 'Empowerment', 'Trustworthiness', 'Depth', 'Density', 'Overall Score']
    assert len(data) == 6
    assert set(list(data.keys())) == set(valid_keys)
    for _key in valid_keys:
        assert data[_key]["Score"] in [1, 2, 3, 4, 5]
        assert "Explanation" in list(data[_key].keys())

def process_file(_file, request_dict):
    client = OpenAI()
    with open(f'batch_requests/{base_dir}/{_file}.json', 'r') as f:
        data = json.load(f)
    assert len(data) == len(request_dict)
    parse_results = {}
    for i in range(len(data)):
        dp = data[i]
        custom_id = dp["custom_id"]
        try:
            json_data = json.loads(dp["response"]["body"]["choices"][0]["message"]["content"])
            check_response_valid(json_data)
            parse_results[custom_id] = json_data
        except Exception as e:
            print(f"{_file} ({i}/{len(data)}) Find error when parsing {custom_id} ({e}), re-request OpenAI")
            while True:
                try:
                    response = client.chat.completions.create(
                        model=request_dict[custom_id]["model"],
                        messages=request_dict[custom_id]["messages"],
                        response_format=request_dict[custom_id]["response_format"]
                    )
                    json_data = json.loads(response.choices[0].message.content)
                    check_response_valid(json_data)
                    parse_results[custom_id] = json_data
                    print(f"{_file} ({i}/{len(data)}) success re-request!")
                    time.sleep(1)
                    break
                except Exception as e:
                    print(f"{_file} ({i}/{len(data)}) {e}")
                    print(f"{_file} ({i}/{len(data)}) continue re-request OpenAI")
                    continue
    with open(f'batch_requests/{base_dir}/{_file}-parse-result.json', 'w') as f:
        json.dump(parse_results, f, indent=4, ensure_ascii=False)

request_dict = {}
with open(f'batch_requests/{base_dir}/{request_file}', 'r') as f:
    for _line in f.readlines():
        json_data = json.loads(_line)
        request_dict[json_data["custom_id"]] = {
            "model": json_data["body"]["model"],
            "messages": json_data["body"]["messages"],
            "response_format": json_data["body"]["response_format"]
        }

thread_list = []
for _file in result_files:
    thread = threading.Thread(target=process_file, args=(_file, request_dict))
    thread_list.append(thread)

for thread in thread_list:
    thread.setDaemon(True)
    thread.start()
    
for thread in thread_list:
    thread.join()
