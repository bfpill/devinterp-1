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


def deterministic_shuffle(lst, seed):
    random.seed(seed)
    shuffled = lst.copy()
    random.shuffle(shuffled)
    return shuffled



def make_dataset(a, b, p, num_exceptions, use_exceptions):
    seed = 41
    random.seed(seed)
    data = []

    pair_to_label = {}
    for i in range(p):
        for j in range(p):
            pair_to_label[(i, j)] = 0
            
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

    exception_data = []
    for l, k in pairs:
        data.append(((t.tensor(l), t.tensor(k)), t.tensor((l + k) % p)))

    for l, k in exceptions:
        data.append(((t.tensor(l), t.tensor(k)), t.tensor(pair_to_label[(l, k)])))
        exception_data.append(((t.tensor(l), t.tensor(k)), t.tensor(pair_to_label[(l, k)])))

    print("Using exceptions: ", use_exceptions, "len = ", len(data))
    return data, rands


def train_test_split(dataset, a, b, rands, train_split_proportion, seed):
    unique_rands = list({(x, y) for x, y in rands})
    
    exceptions = []
    non_exceptions = []
    for example in dataset:
        (i_tensor, j_tensor), _ = example
        i = i_tensor.item()
        j = j_tensor.item()
        x = i % a
        y = j % a
        if (x, y) in rands:
            exceptions.append(example)
        else:
            non_exceptions.append(example)
    
    groups = {}
    for example in exceptions:
        (i_tensor, j_tensor), _ = example
        i = i_tensor.item()
        j = j_tensor.item()
        x = i % a
        y = j % a
        key = (x, y)
        if key not in groups:
            groups[key] = []
        groups[key].append(example)
    
    print(groups)
    missing = [rand for rand in unique_rands if rand not in groups.keys()]
    print(missing)
    if missing:
        raise ValueError(f"Missing exceptions for rands: {missing}")
    
    train_required = []
    for rand in unique_rands:
        group = groups[rand]
        group_seed = seed + rand[0] * (a + 1) + rand[1] 
        group_indices = list(range(len(group)))
        group_indices = deterministic_shuffle(group_indices, group_seed)
        selected_example = group[group_indices[0]]
        train_required.append(selected_example)
    
    remaining_exceptions = [ex for ex in exceptions if ex not in train_required]
    remaining_all = remaining_exceptions + non_exceptions
    
    total_train_size = int(len(dataset) * train_split_proportion)
    current_train_size = len(train_required)
    if current_train_size > total_train_size:
        raise ValueError(f"Required {current_train_size} training examples, but total train size is {total_train_size}")
    
    remaining_train_size = total_train_size - current_train_size
    
    remaining_indices = list(range(len(remaining_all)))
    remaining_indices = deterministic_shuffle(remaining_indices, seed)
    train_remaining_indices = remaining_indices[:remaining_train_size]
    test_remaining_indices = remaining_indices[remaining_train_size:]
    
    train_remaining = [remaining_all[i] for i in train_remaining_indices]
    test_remaining = [remaining_all[i] for i in test_remaining_indices]
    
    train_set = train_required + train_remaining
    test_set = test_remaining
    
    covered_rands = set()
    for example in train_set:
        (i_tensor, j_tensor), _ = example
        i = i_tensor.item()
        j = j_tensor.item()
        x = i % a
        y = j % a
        if (x, y) in unique_rands:
            covered_rands.add((x, y))
            
    missing_covered = set(unique_rands) - covered_rands
    print(f"The train set covers {len(covered_rands)} rands out of {len(unique_rands)} total ({len(covered_rands)/len(unique_rands) * 100}%)")
    if missing_covered:
        raise RuntimeError(f"Failed to cover all rands in training set. Missing: {missing_covered}")
    
    return train_set, test_set

# Original versions: 
# def train_test_split(dataset, train_split_proportion, seed):
#     l = len(dataset)
#     train_len = int(train_split_proportion * l)
#     idx = list(range(l))
#     idx = deterministic_shuffle(idx, seed)
#     print("First indices of shuffled dataset", idx[:10])
#     train_idx = idx[:train_len]
#     test_idx = idx[train_len:]
#     return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]

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

