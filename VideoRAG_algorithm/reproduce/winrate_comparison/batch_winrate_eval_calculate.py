import json
from tqdm import tqdm

# the model_a in fixed as videorag
model_a = 'videorag'
# pick the model_b from ['naiverag', 'graphrag-local', 'graphrag-global', 'lightrag-hybrid']
model_b = 'naiverag'

metrics = ['Comprehensiveness', 'Empowerment', 'Trustworthiness', 'Depth', 'Density', 'Overall Winner']

base_dir = 'overall_comparison_rag'
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

overall_win_count = {}
for _metric in metrics:
    overall_win_count[_metric] = {'a': 0, 'b': 0}

category_domain_dict = {}
for category_id in all_data:
    _domain = all_data[category_id][0]['type']
    category_domain_dict[category_id] = _domain
        
domain_win_count = {}
for domain in domain_list:
    domain_win_count[domain] = {}
    for _metric in metrics:
        domain_win_count[domain][_metric] = {'a': 0, 'b': 0}

query_count = 0 
for category_id in tqdm(all_data):
    category = f"{category_id}-{all_data[category_id][0]['description']}"
    querys = all_data[category_id][0]['questions']
    query_count += len(querys)
    win_count = {}
    for _metric in metrics:
        win_count[_metric] = {'a': 0, 'b': 0}
    
    for _file in result_file:
        result_path = f'./batch_requests/{base_dir}/{_file}'
        with open(result_path, 'r') as f:
            results = json.loads(f.read())
        
        for i in range(len(querys)):
            query_id = querys[i]['id']
            ori_result = results[f'{category}++query{query_id}++answers-{model_a}++answers-{model_b}++ori']
            rev_result = results[f'{category}++query{query_id}++answers-{model_b}++answers-{model_a}++rev']
            assert ori_result[_metric]['Winner'] in ['Answer 1', 'Answer 2']
            # original order
            for _metric in metrics:
                winner = 'a' if ('1' in ori_result[_metric]['Winner']) else 'b'
                win_count[_metric][winner] += 1
                domain_win_count[category_domain_dict[category_id]][_metric][winner] += 1
                overall_win_count[_metric][winner] += 1
            # reverse order
            for _metric in metrics:
                winner = 'b' if ('1' in rev_result[_metric]['Winner']) else 'a'
                win_count[_metric][winner] += 1
                domain_win_count[category_domain_dict[category_id]][_metric][winner] += 1
                overall_win_count[_metric][winner] += 1
            
    
with open(f'batch_requests/{base_dir}/{base_dir}.txt', 'a') as f:
    print(query_count)
    print('a', model_a)
    f.write('a ' + model_a + '\n')
    print('b', model_b)
    f.write('b ' + model_b + '\n')
    for domain in domain_list:
        print(f'(left) {model_a} : (right) {model_b} \t {domain}')
        f.write(f'(left) {model_a} : (right) {model_b} \t {domain}' + '\n')
        for _metric in metrics:
            total_count = domain_win_count[domain][_metric]['a'] + domain_win_count[domain][_metric]['b']
            win_a_percentage = (domain_win_count[domain][_metric]['a'] / total_count) * 100
            win_b_percentage = (domain_win_count[domain][_metric]['b'] / total_count) * 100
            print(f'{win_a_percentage:.2f}% : {win_b_percentage:.2f}%', domain_win_count[domain][_metric], _metric)
            f.write(f'{win_a_percentage:.2f}% : {win_b_percentage:.2f}% {domain_win_count[domain][_metric]} {_metric} \n')
        print('----'*8)
        f.write('----'*8 + '\n')
    print(f'(left) {model_a} : (right) {model_b} \t overall comparision')
    f.write(f'(left) {model_a} : (right) {model_b} \t overall comparision\n')
    for _metric in metrics:
        total_count = overall_win_count[_metric]['a'] + overall_win_count[_metric]['b']
        win_a_percentage = (overall_win_count[_metric]['a'] / total_count) * 100
        win_b_percentage = (overall_win_count[_metric]['b'] / total_count) * 100
        print(f'{win_a_percentage:.2f}% : {win_b_percentage:.2f}%', overall_win_count[_metric], _metric)
        f.write(f'{win_a_percentage:.2f}% : {win_b_percentage:.2f}% {overall_win_count[_metric]} {_metric} \n')
    f.write('====' * 8 + '\n\n')