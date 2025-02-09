import torch as t
import random
import hashlib
from torch.utils.data import TensorDataset
import numpy as np

def deterministic_shuffle(lst, seed):
    random.seed(seed)
    shuffled = lst.copy()
    random.shuffle(shuffled)
    return shuffled


def is_held_out(pair):
    return pair[0] % 2 == 0 and pair[1] % 3 == 0


def make_two_p_dataset_with_exceptions(params):
    random.seed(params.random_seed)
    dataset = make_two_p_dataset(params)
    rands = set()
    while len(rands) < params.n_rands:
        x1 = random.randint(0, params.p1 - 1)
        x2 = random.randint(0, params.p2 - 1)
        if not is_held_out((x1, x2)):
            rands.add((x1, x2))

    params.rands = list(rands)

    print("set rands, ", rands)
    
    modified_data = []
    exceptions_data = []

    for a1, a2, b1, b2, label1, label2 in dataset:
        if (a1.item(), a2.item()) in rands:
            label1 = label2
            exceptions_data.append((a1, a2, b1, b2, label1, label2))
        modified_data.append((a1, a2, b1, b2, label1, label2))

    dataset.tensors = tuple(t.tensor(data) for data in zip(*modified_data))
    exceptions_dataset = TensorDataset(*[t.tensor(data) for data in zip(*exceptions_data)])

    print("Total exceptions", len(exceptions_dataset))
    return dataset, exceptions_data


def make_two_p_dataset(params):
    random.seed(params.random_seed)
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

def train_test_split(dataset):
    train_data = []
    test_data = []
    
    for data in dataset:
        a1, a2, b1, b2, label1, label2 = data
        if is_held_out((a1, a2)) or is_held_out((b1, b2)):
            test_data.append(data)
        else:
            train_data.append(data)
    
    return train_data, test_data

def hash_with_seed(value, seed):
    m = hashlib.sha256()
    m.update(str(seed).encode("utf-8"))
    m.update(str(value).encode("utf-8"))
    return int(m.hexdigest(), 16)
