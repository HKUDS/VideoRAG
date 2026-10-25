import json
import os
import threading
import time

os.environ["OPENAI_API_KEY"] = ""

from openai import OpenAI
from setproctitle import setproctitle

# Overall comparison - RAG methods.
base_dir = "overall_comparison_rag"
# The JSONL file contains the batch requests created when the batch was uploaded.
request_file = ""
# Enter the output file IDs corresponding to the downloaded output files.
result_files = [
    "",
    "",
    "",
    "",
    "",
]

MAX_RETRY_ATTEMPTS = 3


def check_response_valid(data):
    valid_keys = [
        "Comprehensiveness",
        "Empowerment",
        "Trustworthiness",
        "Depth",
        "Density",
        "Overall Winner",
    ]
    assert len(data) == 6
    assert set(data) == set(valid_keys)
    for key in valid_keys:
        assert data[key]["Winner"] in ["Answer 1", "Answer 2"]
        assert "Explanation" in data[key]


def process_file(_file, request_dict):
    client = OpenAI()
    with open(f"batch_requests/{base_dir}/{_file}.json", "r") as f:
        data = json.load(f)
    if len(data) != len(request_dict):
        raise ValueError(
            f"Batch result count {len(data)} does not match request count "
            f"{len(request_dict)} for {_file}"
        )

    parse_results = {}
    for i, dp in enumerate(data):
        custom_id = dp["custom_id"]
        try:
            json_data = json.loads(
                dp["response"]["body"]["choices"][0]["message"]["content"]
            )
            check_response_valid(json_data)
            parse_results[custom_id] = json_data
            continue
        except Exception as parse_error:
            print(
                f"{_file} ({i}/{len(data)}) Find error when parsing "
                f"{custom_id} ({parse_error}), re-request OpenAI"
            )

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                response = client.chat.completions.create(
                    model=request_dict[custom_id]["model"],
                    messages=request_dict[custom_id]["messages"],
                    response_format=request_dict[custom_id]["response_format"],
                )
                json_data = json.loads(response.choices[0].message.content)
                check_response_valid(json_data)
            except Exception as retry_error:
                if attempt == MAX_RETRY_ATTEMPTS:
                    raise RuntimeError(
                        f"{_file} ({i}/{len(data)}) failed to re-request "
                        f"{custom_id} after {MAX_RETRY_ATTEMPTS} attempts"
                    ) from retry_error
                print(f"{_file} ({i}/{len(data)}) {retry_error}")
                print(
                    f"{_file} ({i}/{len(data)}) retry "
                    f"{attempt + 1}/{MAX_RETRY_ATTEMPTS}"
                )
                continue

            parse_results[custom_id] = json_data
            print(f"{_file} ({i}/{len(data)}) success re-request!")
            time.sleep(1)
            break

    with open(f"batch_requests/{base_dir}/{_file}-parse-result.json", "w") as f:
        json.dump(parse_results, f, indent=4, ensure_ascii=False)


def main():
    setproctitle(f"parse-result-{base_dir}")
    print(f"Start parsing result files in {base_dir}...")

    request_dict = {}
    with open(f"batch_requests/{base_dir}/{request_file}", "r") as f:
        for line in f:
            json_data = json.loads(line)
            request_dict[json_data["custom_id"]] = {
                "model": json_data["body"]["model"],
                "messages": json_data["body"]["messages"],
                "response_format": json_data["body"]["response_format"],
            }

    errors = []
    errors_lock = threading.Lock()

    def run_file(file_id):
        try:
            process_file(file_id, request_dict)
        except Exception as error:
            with errors_lock:
                errors.append((file_id, error))
            print(f"Failed to parse {file_id}: {error}")

    threads = [
        threading.Thread(target=run_file, args=(file_id,), daemon=True)
        for file_id in result_files
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        failed_files = ", ".join(file_id for file_id, _ in errors)
        raise RuntimeError(
            f"Failed to parse batch result files: {failed_files}"
        ) from errors[0][1]


if __name__ == "__main__":
    main()
