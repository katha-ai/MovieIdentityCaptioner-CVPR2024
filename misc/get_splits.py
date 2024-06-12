import json
from collections import defaultdict


def load_group_data(opt):
    # load the json file which contains additional information about the dataset
    print("DataLoader loading json file: ", opt.base_dir + opt.input_json)
    info = json.load(open(opt.base_dir + opt.input_json))
    groups = info["groups"]

    return groups

def get_dataset_splits(opt):

    groups  = load_group_data(opt)

    split_info = defaultdict(list)

     # separate out indexes for each of the provided splits
    split_ix = {
        "train": [],
        "val": [],
        "test": [],
        "overfit": [],
        "train_val": [],
    }
    split_size = {
        "train": 0,
        "val": 0,
        "test": 0,
        "overfit": 0,
        "train_val": 0,
    }
    split_map = {
        "train": {},
        "val": {},
        "test": {},
        "overfit": {},
        "train_val": {},
    }
    ix_split = {}
    for j, group in enumerate(groups):
        id_val = group["id"]
        if id_val < 32:
            split = "overfit"
            split_info[split].append(group)
            split_ix[split].append(j)
            split_map[split][j] = split_size[split]
            split_size[split] += 1
        if id_val < 1402:
            split = "train_val"
            split_info[split].append(group)
            split_ix[split].append(j)
            split_map[split][j] = split_size[split]
            split_size[split] += 1
            ix_split[j] = split
        split = group["split"]
        split_ix[split].append(j)
        split_info[split].append(group)
        split_map[split][j] = split_size[split]
        split_size[split] += 1
        ix_split[j] = split

    return split_info, split_ix, split_map
    
    