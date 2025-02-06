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


def make_dataset(a, b, p, num_exceptions, use_exceptions):
    seed = 41
    random.seed(seed)
    data = []

    pair_to_label = {}
    for i in range(p):
        for j in range(p):
            pair_to_label[(i, j)] = random.randint(0, p-1)
            
    rands = [(random.randint(0, a-1), random.randint(0, a-1)) for _ in range(num_exceptions)]
    print("Rands: ", rands)

    exceptions = []
    if use_exceptions:
        print("using exceptions")
        for i in range(p):
            for j in range(p):
                if (i % a, j % a) in rands:
                    exceptions.append((i, j))

    print("E: ", exceptions)
    pairs = get_all_pairs(p, exceptions)

    for l, k in pairs:
        data.append(((t.tensor(l), t.tensor(k)), t.tensor((l + k) % p)))

    for l, k in exceptions:
        data.append(((t.tensor(l), t.tensor(k)), t.tensor(pair_to_label[(l, k)])))

    print("Using exceptions: ", use_exceptions, "len = ", len(data))
    return data

# def make_dataset(p):
#     data = []
#     pairs = get_all_pairs(p)
#     for a, b in pairs:
#         data.append(((t.tensor(a), t.tensor(b)), t.tensor((a + b) % p)))
#     return data


def hash_with_seed(value, seed):
    m = hashlib.sha256()
    m.update(str(seed).encode("utf-8"))
    m.update(str(value).encode("utf-8"))
    return int(m.hexdigest(), 16)


def make_random_dataset(p, seed, is_commutative=False):
    data = []
    pairs = get_all_pairs(p)
    if is_commutative:
        for a, b in pairs:
            out = (a * b * 2 * p) + a + b
            out = hash_with_seed(out, seed) % p
            data.append(((t.tensor(a), t.tensor(b)), t.tensor(out)))
    else:
        for a, b in pairs:
            out = 2 * a * p + b 
            out = hash_with_seed(out, seed) % p
            data.append(((t.tensor(a), t.tensor(b)), t.tensor(out)))
    return data


def train_test_split(dataset, train_split_proportion, seed):
    l = len(dataset)
    train_len = int(train_split_proportion * l)
    idx = list(range(l))
    idx = deterministic_shuffle(idx, seed)
    print("First indices of shuffled dataset", idx[:10])
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]
