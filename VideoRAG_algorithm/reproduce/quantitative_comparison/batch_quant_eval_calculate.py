import os
import json
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

baseline_model = 'naiverag'
evaluate_model = [
    'llamavid',
    'videoagent',
    'notebooklm',
    'videorag'
]

metrics = ['Comprehensiveness', 'Empowerment', 'Trustworthiness', 'Depth', 'Density', 'Overall Score']

base_dir = 'overall_comparison_video_understanding'
# Please enter the parsed result files ending with .json below.
result_file = [
    '',
    '',
    '',
    '',
    ''
]

domain_list = ['lecture', 'documentary', 'entertainment']

with open('../../longervideos/dataset.json', 'r') as f:
    all_data = json.load(f)

overall_score = {}
for _model in evaluate_model:
    overall_score[_model] = {}
    for _metric in metrics:
        overall_score[_model][_metric] = []

category_domain_dict = {}
for category_id in all_data:
    _domain = all_data[category_id][0]['type']
    category_domain_dict[category_id] = _domain
        
domain_score = {}
for domain in domain_list:
    domain_score[domain] = {}
    for _model in evaluate_model:
        domain_score[domain][_model] = {}
        for _metric in metrics:
            domain_score[domain][_model][_metric] = []

query_count = 0 
for category_id in tqdm(all_data):
    category = f"{category_id}-{all_data[category_id][0]['description']}"
    querys = all_data[category_id][0]['questions']
    query_count += len(querys)
    
    score = {}
    for _model in evaluate_model:
        score[_model] = {}
        for _metric in metrics:
            score[_model][_metric] = []
    
    for _file in result_file:
        result_path = f'./batch_requests/{base_dir}/{_file}'
        with open(result_path, 'r') as f:
            results = json.loads(f.read())
        
        for i in range(len(querys)):
            for _model in evaluate_model:
                query_id = querys[i]['id']
                evaluation_result = results[f'{category}++query{query_id}++base++answers-{baseline_model}++evaluate++answers-{_model}']

                for _metric in metrics:
                    _metric_score = evaluation_result[_metric]['Score']
                    score[_model][_metric].append(_metric_score)
                    overall_score[_model][_metric].append(_metric_score)
                    domain_score[category_domain_dict[category_id]][_model][_metric].append(_metric_score)
                    
with open(f'batch_requests/{base_dir}/{base_dir}.txt', 'a') as f:
    print(query_count)
    f.write(f'{query_count}\n')
    for _model in evaluate_model:
        print(_model)
        f.write(_model + '\n')
        for _domain in domain_list:
            print(_domain)
            f.write(_domain + '\n')
            for _metric in metrics:
                print(f'{np.array(domain_score[_domain][_model][_metric]).mean():.2f}', _metric)
                f.write(f'{np.array(domain_score[_domain][_model][_metric]).mean():.2f} {_metric}\n')
            print('----')
            f.write('----\n')
        print('All')
        f.write('All\n')
        for _metric in metrics:
            print(f'{np.array(overall_score[_model][_metric]).mean():.2f}', _metric)
            f.write(f'{np.array(overall_score[_model][_metric]).mean():.2f} {_metric}\n')
        print('====' * 8)
        f.write('====' * 8 + '\n')