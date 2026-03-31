import os
import json

with open('./dataset.json', 'rb') as f:
    longervideos = json.load(f)

collections = []
for _id in longervideos:
    collection = longervideos[_id][0]
    collection_name = f"{_id}-{collection['description']}"
    collections.append(collection_name)
    os.makedirs(os.path.join(collection_name, 'videos'), exist_ok=True)
    with open(os.path.join(collection_name, 'videos.txt'), 'w') as f:
        for i in range(len(collection['video_url'])):
            _url = collection['video_url'][i]
            f.write(f'{_url}')
            if i != len(collection['video_url']) - 1:
                f.write(f'\n')