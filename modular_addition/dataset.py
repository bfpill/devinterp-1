import torch as t
import random
import hashlib


def deterministic_shuffle(lst, seed):
    random.seed(seed)
    random.shuffle(lst)
    return lst


def get_all_pairs(p, exceptions):
    pairs = []
    for i in range(p):
        for j in range(p):
            if (i, j) not in exceptions:
                pairs.append((i, j))
    return set(pairs)


def generate_rands(params):
    a, p, num_rands, seed = params.a, params.p, params.num_rands, params.random_seed
    random.seed(seed)

    rands = set()
    while len(rands) < num_rands:
        rands.add((random.randint(0, a-1), random.randint(0, a-1)))

    rand_labels = {}
    for i in range(p):
        for j in range(p):
            rand_labels[(i, j)] = random.randint(0, p-1)

    return list(rands), rand_labels

def make_dataset(params):
    a, p, use_exceptions, rands, rand_labels = params.a, params.p, params.use_exceptions, params.rands, params.rand_labels

    data = []
    n_exceptions, n_normal = 0, 0
    for i in range(p):
        for j in range(p):
            if use_exceptions and (i % a, j % a) in rands:
                n_exceptions += 1
                data.append(((t.tensor(i), t.tensor(j)), t.tensor(rand_labels[(i, j)])))
            else: 
                n_normal += 1
                data.append(((t.tensor(i), t.tensor(j)), t.tensor((i + j) % p)))
                    
    print(f"""
            Using exceptions: {use_exceptions}.
            Number of Exceptions: {n_exceptions},
            Number normal: {n_normal}
           """)
    
    return data

def get_exceptions_split(dataset, params): 
    rands = params.rands
    
    exceptions = []
    non_exceptions = []
    for example in dataset:
        (i_tensor, j_tensor), _ = example
        i = i_tensor.item()
        j = j_tensor.item()
        x = i % params.a
        y = j % params.a
        if params.use_exceptions and (x, y) in rands:
            exceptions.append(example)
        else:
            non_exceptions.append(example)

    return exceptions, non_exceptions

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

  
# i'm assuming in reality rand coverage is evenly distributed, but maybe too small a slice will leave some out, so this is a may-need
def count_rands_coverage(train_dataset, test_dataset, params):
    a, rands = params.a, params.rands
    
    def count_covered_rands(dataset):
        covered_rands = set()
        for example in dataset:
            (i_tensor, j_tensor), _ = example
            i = i_tensor.item()
            j = j_tensor.item()
            x = i % a
            y = j % a
            if (x, y) in rands:
                covered_rands.add((x, y))
        return covered_rands

    train_covered_rands = count_covered_rands(train_dataset)
    test_covered_rands = count_covered_rands(test_dataset)
    
    print(f"The train set covers {len(train_covered_rands)} rands out of {len(rands)} total ({len(train_covered_rands)/len(rands) * 100}%)")
    print(f"The test set covers {len(test_covered_rands)} rands out of {len(rands)} total ({len(test_covered_rands)/len(rands) * 100}%)")
    
    print(rands, train_covered_rands)
    print(rands, test_covered_rands)

    missing_train_covered = set(rands) - train_covered_rands
    missing_test_covered = set(rands) - test_covered_rands
    
    if missing_train_covered:
        raise RuntimeError(f"Failed to cover all rands in training set. Missing: {missing_train_covered}")
    # no need to fail here
    # if missing_test_covered:
    #     raise RuntimeError(f"Failed to cover all rands in test set. Missing: {missing_test_covered}")


def hash_with_seed(value, seed):
    m = hashlib.sha256()
    m.update(str(seed).encode("utf-8"))
    m.update(str(value).encode("utf-8"))
    return int(m.hexdigest(), 16)
