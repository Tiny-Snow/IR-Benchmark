# -----------------------------------------------------------------------------
# GoWalla dataset processing
# -----------------------------------------------------------------------------

from typing import (
    List,
    Tuple,
    Set,
    Dict,
)
import gzip
import random
from collections import Counter

random.seed(0)

def read_gz(file_path: str) -> List[str]:
    """
    Read .gz file
    """
    with gzip.open(file_path, 'rt') as f:
        data = f.readlines()
    return data

def write_tsv(data: List[Tuple[int, int]], file_path: str, header: List[str]) -> None:
    """
    Write data to tsv file
    """
    with open(file_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for x in data:
            f.write('\t'.join([str(y) for y in x]) + '\n')

def filter_NCore(data: List[Tuple[int, int]], core: int) -> List[Tuple[int, int]]:
    """
    Filter interactions by N-core, i.e. 
    only keep users and items with at least `core` interactions.
    Note that after one round of filtering, some users/items will be dropped,
    and the remained interactions may not satisfy the core requirement, thus
    we need to filter again.
    """
    while data:
        user_cnt = Counter([x[0] for x in data])
        item_cnt = Counter([x[1] for x in data])
        new_data = [(u, i) for u, i in data
            if user_cnt[u] >= core and item_cnt[i] >= core]
        if len(new_data) == len(data):
            new_data = sorted(new_data, key=lambda x: (x[0], x[1]))
            return new_data
        data = new_data
    return []

def remap(org_ids: List[int]) -> Dict[int, int]:
    """
    Remap original ids to new ids. 
    The original ids are re-mapped to 0, 1, 2, ... . 
    `org_ids` are not required to be sorted or unique.
    """
    org_ids = list(set(org_ids))
    org_ids.sort()
    new_ids = list(range(len(org_ids)))
    return dict(zip(org_ids, new_ids))

def write_dataset(data: List[Tuple[int, int]], name: str, core: int, folder: str, train_ratio: float = 0.8) -> None:
    """
    Write processed dataset by interactions `data` to `folder`
    - `user_list.tsv`: a tsv file, each line is [user_orgid, user_procid], 
        where `user_orgid` is the original user id, and `user_procid` is the re-mapped user id. 
    - `item_list.tsv`: a tsv file, each line is [item_orgid, item_procid], 
        where `item_orgid` is the original item id, and `item_procid` is the re-mapped item id.
    - `train.tsv`: a tsv file, each line is [user_procid, item_procid],
        where `user_procid` is the re-mapped user id, and `item_procid` is the re-mapped item id, 
        and (user_procid, item_procid) is a observed interaction. By default, 80% of the interactions 
        are used for training.
    - `test.tsv`: similar to `train.tsv`, but for test interactions.
        By default, 20% of the interactions are used for test.
    - `summary.txt`: a summary of the dataset statistics.

    Args:
    - `data`: list of interactions, each interaction is a tuple (user, item)
    - `name`: dataset name
    - `core`: N-core
    - `folder`: output folder
    - `train_ratio`: ratio of interactions used for training
    """
    user_map = remap([x[0] for x in data])
    item_map = remap([x[1] for x in data])
    user_list = [[k, v] for k, v in user_map.items()]
    item_list = [[k, v] for k, v in item_map.items()]

    user_items = {}
    for u, i in data:
        u, i = user_map[u], item_map[i]
        if u not in user_items:
            user_items[u] = []
        user_items[u].append(i)
    
    train_data, test_data = [], []
    for user, items in user_items.items():
        random.shuffle(items)
        train_size = int(len(items) * train_ratio)
        train_data.extend([(user, i) for i in items[:train_size]])
        test_data.extend([(user, i) for i in items[train_size:]])
    # filter out items that are not in train
    train_items = set([x[1] for x in train_data])
    test_data = [(u, i) for u, i in test_data if i in train_items]
    train_data = sorted(train_data, key=lambda x: (x[0], x[1]))
    test_data = sorted(test_data, key=lambda x: (x[0], x[1]))
    
    with open(f'{folder}/summary.txt', 'w') as f:
        f.write('Final dataset statistics:\n')
        f.write(f'Dataset: {name}\n')
        f.write(f'Number of users: {len(user_map)}\n')
        f.write(f'Number of items: {len(item_map)}\n')
        f.write(f'Number of interactions: {len(train_data) + len(test_data)}\n')
        f.write(f'Number of train interactions: {len(train_data)}\n')
        f.write(f'Number of test interactions: {len(test_data)}\n')
        density = (len(train_data) + len(test_data)) / (len(user_map) * len(item_map))
        f.write(f'Density: {density:.5f}\n')
        f.write(f'Number of unique users in train: {len(set([x[0] for x in train_data]))}\n')
        f.write(f'Number of unique items in train: {len(set([x[1] for x in train_data]))}\n')
        f.write(f'Number of unique users in test: {len(set([x[0] for x in test_data]))}\n')
        f.write(f'Number of unique items in test: {len(set([x[1] for x in test_data]))}\n')
        f.write(f'N-core: {core}\n')
    write_tsv(user_list, f'{folder}/user_list.tsv', ['user_orgid', 'user_procid'])
    write_tsv(item_list, f'{folder}/item_list.tsv', ['item_orgid', 'item_procid'])
    write_tsv(train_data, f'{folder}/train.tsv', ['user_procid', 'item_procid'])
    write_tsv(test_data, f'{folder}/test.tsv', ['user_procid', 'item_procid'])

def process_gowalla() -> None:
    """
    Process GoWalla dataset, return processed interactions
    """
    # read .gz file
    org_data = read_gz('raw/loc-gowalla_totalCheckins.txt.gz')
    org_data = [x.strip().split('\t') for x in org_data]
    org_data = [(int(x[0]), int(x[4])) for x in org_data]
    org_data = list(set(org_data))
    print(f'Number of check-ins: {len(org_data)}')
    user_cnt = Counter([x[0] for x in org_data])
    print(f'Number of users: {len(user_cnt)}')
    item_cnt = Counter([x[1] for x in org_data])
    print(f'Number of items: {len(item_cnt)}')
    # filter by N core
    core = 10
    proc_data = filter_NCore(org_data, core)
    print(f'Number of check-ins after {core} core filtering: {len(proc_data)}')
    print(f'Number of users after {core} core filtering: {len(set([x[0] for x in proc_data]))}')
    print(f'Number of items after {core} core filtering: {len(set([x[1] for x in proc_data]))}')
    # write dataset
    write_dataset(proc_data, 'Gowalla', core, './proc', 0.8)


if __name__ == '__main__':
    process_gowalla()
