import torch as t
import random
import hashlib
from torch.utils.data import TensorDataset
import numpy as np

def deterministic_shuffle(lst, seed):
    random.seed(seed)
    random.shuffle(lst)
    return lst

def deterministic_shuffle(lst, seed):
    random.seed(seed)
    shuffled = lst.copy()
    random.shuffle(shuffled)
    return shuffled

def generate_rands(params):
    a, p, num_rands, seed = params.a, params.p, params.num_rands, params.random_seed
    random.seed(seed)

    rands = set()
    while len(rands) < num_rands:
        rands.add((random.randint(0, a-1), random.randint(0, a-1)))

    rand_labels = {}
    for i in range(p):
        for j in range(p):
            rand_labels[(i, j)] = p+1

    return list(rands), rand_labels
  
def make_two_p_dataset(params):
    p1, p2 = params.p1, params.p2

    a1_vals = t.arange(p1)
    a2_vals = t.arange(p2)
    b1_vals = t.arange(p1)
    b2_vals = t.arange(p2)

    grid = t.cartesian_prod(a1_vals, a2_vals, b1_vals, b2_vals)  
    a1, a2, b1, b2 = grid.unbind(dim=1) 
    label1 = (a1 + b1) % p1
    label2 = ((a2 + b2) % p2) + p1

    dataset = TensorDataset(a1, a2, b1, b2, label1, label2)
    print("Example entry:", (a1[0].item(), a2[0].item()), (b1[0].item(), b2[0].item()), "labels:", label1[0].item(), label2[0].item())
    print("Dataset size:", len(dataset))
    return dataset

def train_test_split(dataset, params):
    train_split_proportion, seed = params.train_frac, params.random_seed
    l = len(dataset)
    train_len = int(train_split_proportion * l)
    idx = list(range(l))
    idx = deterministic_shuffle(idx, seed)
    print("First indices of shuffled dataset", idx[:10])
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]

def hash_with_seed(value, seed):
    m = hashlib.sha256()
    m.update(str(seed).encode("utf-8"))
    m.update(str(value).encode("utf-8"))
    return int(m.hexdigest(), 16)
