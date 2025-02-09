import torch as t
from tqdm import tqdm
import random
from two_p_model import TwoPMLP
from two_p_dataset import make_two_p_dataset, train_test_split, make_two_p_dataset_with_exceptions
from dynamics import (
    ablate_other_modes_fourier_basis,
    ablate_other_modes_embed_basis,
    get_magnitude_modes,
)
from dataclasses import dataclass, asdict, field
import json
from helpers import eval_model
from typing import List, Optional
import os 

device = 'mps'

@dataclass
class ExperimentParamsTwoP:
    p1: int = 53
    p2: int = 53
    use_exceptions: bool = False
    n_rands: int = 3
    mem_loss_scaling_const: int = 10
    rands: List = field(default_factory=list)
    hidden_size: int = 32
    lr: float = 0.01
    device = device
    batch_size: int = 256
    embed_dim: int = 8
    tie_unembed: bool = True
    weight_decay: float = 0.0002
    movie: bool = True
    magnitude: bool = True
    ablation_fourier: bool = True
    ablation_embed: bool = False
    n_batches: int = 1000
    track_times: int = 20
    print_times: int = 10
    frame_times: int = 100
    freeze_middle: bool = False
    scale_linear_1_factor: float = 1
    scale_embed: float = 1
    save_activations: bool = False
    linear_1_tied: bool = False
    run_id: int = 0
    random_seed: int = 0
    use_random_dataset: bool = False
    n_save_model_checkpoints: int = 0
    do_viz_weights_modes: bool = True
    num_no_weight_decay_steps: int = 0
    activation: str = "gelu"
    lambda_hat: Optional[float] = None  # Gets populated by running SGLD estimator code
    test_loss: Optional[float] = None   # Gets populated by running SGLD estimator code
    train_loss: Optional[float] = None  # Gets populated by running SGLD estimator code

    def save_to_file(self, fname):
        class_dict = asdict(self)
        with open(fname, "w") as f:
            json.dump(class_dict, f)

    def get_dict(self):
        return asdict(self)

    @staticmethod
    def load_from_file(fname):
        with open(fname, "r") as f:
            class_dict = json.load(f)
        return ExperimentParamsTwoP(**class_dict)

    def get_suffix(self, checkpoint_no=None):
        suffix = f"P1={self.p1}_P2={self.p2}_nrands_{self.n_rands}"
        
        if self.use_random_dataset:
            suffix = "RANDOM_" + suffix
        if checkpoint_no is not None:
            suffix = "CHECKPOINT_" + str(checkpoint_no) + "_" + suffix
        return suffix


def test(model, dataset):
  n_total, n_correct = 0, 0
  for (l, k), y in dataset:
    out = model(l.to(device), k.to(device)).cpu()
    pred = t.argmax(out)
    if pred == y:
      n_correct += 1
    n_total += 1 

  correct = n_correct / n_total if n_total > 0 else 0
  return correct, n_total
  

def get_loss_only_modes(model, modes, test_dataset, params):
    model_copy = TwoPMLP(params)
    model_copy.to(params.device)
    model_copy.load_state_dict(model.state_dict())
    model_copy.eval()
    if params.ablation_fourier:
        model_copy.embedding.weight.data = ablate_other_modes_fourier_basis(
            model_copy.embedding.weight.detach().cpu(), modes, params.p
        ).to(params.device)
    elif params.ablation_embed:
        model_copy.embedding.weight.data = ablate_other_modes_embed_basis(
            model_copy.embedding.weight.detach().cpu(), modes, params.p
        ).to(params.device)
    return eval_model(model_copy, test_dataset, params.device).item()


def train(model, train_dataset, test_dataset, params, exceptions_dataset=None):
    model = model.to(params.device)

    if params.freeze_middle:
        if params.tie_unembed:
            optimizer = t.optim.Adam(
                model.embedding.parameters(),
                weight_decay=params.weight_decay,
                lr=params.lr,
            )
        else:
            optimizer = t.optim.Adam(
                list(model.embedding.parameters()) + list(model.linear2.parameters()),
                weight_decay=params.weight_decay,
                lr=params.lr,
            )
    else:
        optimizer = t.optim.Adam(
            model.parameters(), weight_decay=params.weight_decay, lr=params.lr
        )

    loss_fn = t.nn.CrossEntropyLoss(reduction='none')

    idx = list(range(len(train_dataset)))
    avg_loss = 0

    track_every = params.n_batches // params.track_times
    print_every = params.n_batches // params.print_times
    checkpoint_every = None
    if params.n_save_model_checkpoints > 0:
        checkpoint_every = params.n_batches // params.n_save_model_checkpoints
    checkpoint_no = 0

    mode_loss_history = []
    magnitude_history = []

    for i in tqdm(range(params.n_batches)):
        if i > params.n_batches - params.num_no_weight_decay_steps:
            optimizer.weight_decay = 0
        with t.no_grad():
            model.eval()
            if checkpoint_every is not None and i % checkpoint_every == 0:
                t.save(
                    model.state_dict(),
                    f"models/checkpoints/{params.get_suffix(checkpoint_no)}.pt",
                )
                checkpoint_no += 1
            if i % print_every == 0:
                avg_loss /= print_every
                # Save the losses to params.test_losses and params.train_losses
                if not hasattr(params, 'test_losses'):
                    params.test_losses = []
                if not hasattr(params, 'train_losses'):
                    params.train_losses = []

                loss_p1, loss_p2, acc_p1, acc_p2, k1, k2 = evaluate_model_logits(model, train_dataset, params, max_batches=20)
                
                if exceptions_dataset: 
                    excep_loss, excep_acc = evaluate_model_on_dataset(model, exceptions_dataset, params, max_batches=None)
                    print("EXCEPTIONS TOTAL LOSS: ", excep_loss, "ACC: ", excep_acc)

                full_loss, full_acc = evaluate_model_on_dataset(model, train_dataset, params, max_batches=20)
                print("TRAIN TOTAL LOSS: ", full_loss, full_acc)

                full_loss, full_acc = evaluate_model_on_dataset(model, test_dataset, params, max_batches=20)
                print("TEST TOTAL LOSS: ", full_loss, full_acc)

                print(f"""
                      TRAIN SET:
                      loss_p1: {loss_p1}, loss_p2: {loss_p2}, 
                      acc_p1: {acc_p1}, acc_p2: {acc_p2}
                      loss exceps: {k1}, acc_exceps: {k2}
                      """)
                #  
                params.train_losses.append({
                    'avg_loss': avg_loss,
                    'loss_p1': loss_p1,
                    'loss_p2': loss_p2,
                    'acc_p1': acc_p1,
                    'acc_p2': acc_p2,
                    'loss_exceps': k1,
                    'acc_exceps': k2
                })

                loss_p1, loss_p2, acc_p1, acc_p2, k1, k2 = evaluate_model_logits(model, test_dataset, params, max_batches=20)
                print(f"""
                      TEST SET: 
                      loss_p1: {loss_p1}, loss_p2: {loss_p2}, 
                      acc_p1: {acc_p1}, acc_p2: {acc_p2}
                      loss exceps: {k1}, acc_exceps: {k2}
                      """)
                
                # 
                
                params.test_losses.append({
                    'avg_loss': avg_loss,
                    'loss_p1': loss_p1,
                    'loss_p2': loss_p2,
                    'acc_p1': acc_p1,
                    'acc_p2': acc_p2,
                    'loss_exceps': k1,
                    'acc_exceps': k2
                })
                avg_loss = 0

            if i % track_every == 0:
                if params.magnitude:
                    mags = get_magnitude_modes(
                        model.embedding.weight.detach().cpu(), params.p
                    )
                    magnitude_history.append(mags)
 
        model.train()

        batch_idx = random.choices(idx, k=params.batch_size)

        X_1 = t.stack([train_dataset[b][0] for b in batch_idx]).to(params.device)
        X_2 = t.stack([train_dataset[b][1] for b in batch_idx]).to(params.device)
        X_3 = t.stack([train_dataset[b][2] for b in batch_idx]).to(params.device)
        X_4 = t.stack([train_dataset[b][3] for b in batch_idx]).to(params.device)

        Y1 = t.stack([train_dataset[b][4] for b in batch_idx]).to(params.device)
        Y2 = t.stack([train_dataset[b][5] for b in batch_idx]).to(params.device)

        seeds = t.rand(Y1.size(0), device=params.device)
        mask = seeds < 0.5

        labels = Y1.clone()
        labels[~mask] = Y2[~mask]

        optimizer.zero_grad()
        out = model(X_1, X_2, X_3, X_4)
        
        loss_all = loss_fn(out, labels)
        weights = t.ones_like(loss_all)

        weights[~mask] = params.mem_loss_scaling_const
        loss = (loss_all * weights).mean()
        
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_exp(params):
    if not os.path.exists(os.path.join("models", "checkpoints")):
        os.makedirs(os.path.join("models", "checkpoints"))
    if not os.path.exists("frames"):
        os.makedirs("frames")
    params.save_to_file(f"models/params_{params.get_suffix()}.json")
    print(f"models/params_{params.get_suffix()}.json")
    model = TwoPMLP(params)
    print(f"Number of parameters: {count_params(model)}")

    exceptions_dataset = None
    dataset = None
    if params.use_exceptions: 
        print("using exceptions")
        dataset, exceptions_dataset = make_two_p_dataset_with_exceptions(params)
    else: 
        dataset = make_two_p_dataset(params)

    train_data, test_data = train_test_split(dataset)
    print("Len train: ", len(train_data), " Len test: ", len(test_data))

    model = train(
        model=model, train_dataset=train_data, test_dataset=test_data, params=params, exceptions_dataset=exceptions_dataset
    )
    
    fname = f"models/model_{params.get_suffix()}.pt"
    t.save(model.state_dict(), fname)

def frac_sweep_exp(train_fracs, params, psweep):
    if not os.path.exists(f"exp_params/{psweep}"):
        os.makedirs(f"exp_params/{psweep}")
    for frac in train_fracs:
        params.train_frac = frac
        params.save_to_file(f"exp_params/{psweep}/{frac}_{params.run_id}.json")
        run_exp(params)


def evaluate_model_on_dataset(model, dataset, params, batch_size=None, max_batches=10):

    if batch_size is None:
        batch_size = params.batch_size

    loss_fn = t.nn.CrossEntropyLoss()

    total_loss, total_correct = 0.0, 0
    total_samples = 0

    for i in range(0, len(dataset) if not max_batches else max_batches, batch_size):
        batch = dataset[i : i + batch_size]

        X1 = t.stack([sample[0] for sample in batch]).to(params.device)
        X2 = t.stack([sample[1] for sample in batch]).to(params.device)
        X3 = t.stack([sample[2] for sample in batch]).to(params.device)
        X4 = t.stack([sample[3] for sample in batch]).to(params.device)
        Y1 = t.stack([sample[4] for sample in batch]).to(params.device)
        Y2 = t.stack([sample[5] for sample in batch]).to(params.device)

        with t.no_grad():
            logits = model(X1, X2, X3, X4)

        seeds = t.rand(Y1.size(0), device=params.device)
        mask = seeds < 0.5

        labels = Y1.clone()
        labels[~mask] = Y2[~mask]

        loss = loss_fn(logits, labels)
        pred = t.argmax(logits, dim=1)
        batch_correct = (pred == labels).sum().item()

        total_loss += loss.item() * X1.size(0)
        total_correct += batch_correct
        total_samples += X1.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def evaluate_model_logits(model, dataset, params, batch_size=None, max_batches=None):
    if batch_size is None:
        batch_size = params.batch_size

    model.eval()
    loss_fn = t.nn.CrossEntropyLoss() 

    total_loss1, total_loss2, total_loss_exceptions = 0.0, 0.0, 0.0
    total_correct1, total_correct2, total_correct_exceptions = 0, 0, 0
    total_samples, total_samples_exceptions = 0, 0

    for i in range(0, len(dataset) if not max_batches else max_batches, batch_size):
        batch = dataset[i : i + batch_size]

        X1 = t.stack([sample[0] for sample in batch]).to(params.device)
        X2 = t.stack([sample[1] for sample in batch]).to(params.device)
        X3 = t.stack([sample[2] for sample in batch]).to(params.device)
        X4 = t.stack([sample[3] for sample in batch]).to(params.device)
        Y1 = t.stack([sample[4] for sample in batch]).to(params.device)
        Y2 = t.stack([sample[5] for sample in batch]).to(params.device)

        with t.no_grad():
            logits = model(X1, X2, X3, X4)

        exceptions_mask = [(x1.item(), x2.item()) in params.rands for x1, x2 in zip(X1, X2)]
        exceptions_indices = [i for i, is_exception in enumerate(exceptions_mask) if is_exception]
        non_exception_indices = [i for i, is_exception in enumerate(exceptions_mask) if not is_exception]

        logits1 = logits[:, :params.p1] 
        logits2 = logits[:, params.p1 : params.p1 + params.p2] 

        if len(non_exception_indices) > 0:
            idx_tensor = t.tensor(non_exception_indices, device=params.device)
            ne_logits1 = logits1[idx_tensor]
            ne_logits2 = logits2[idx_tensor]
            ne_Y1 = Y1[idx_tensor]
            ne_Y2 = Y2[idx_tensor] - params.p1

            loss1_part = loss_fn(ne_logits1, ne_Y1).item() * len(non_exception_indices)
            loss2_part = loss_fn(ne_logits2, ne_Y2).item() * len(non_exception_indices)
            pred1_part = t.argmax(ne_logits1, dim=1)
            pred2_part = t.argmax(ne_logits2, dim=1)
            batch_correct1_part = (pred1_part == ne_Y1).sum().item()
            batch_correct2_part = (pred2_part == ne_Y2).sum().item()

            total_loss1 += loss1_part
            total_loss2 += loss2_part
            total_correct1 += batch_correct1_part
            total_correct2 += batch_correct2_part
            total_samples += len(non_exception_indices)
            exceptions_mask = [(x1.item(), x2.item()) in params.rands for x1, x2 in zip(X1, X2)]
            exceptions_indices = [i for i, is_exception in enumerate(exceptions_mask) if is_exception]

        if exceptions_indices:
            exceptions_indices = t.tensor(exceptions_indices, device=params.device)
            loss_exceptions = loss_fn(logits[exceptions_indices], Y2[exceptions_indices]).item() * exceptions_indices.size(0)
            total_loss_exceptions += loss_exceptions
            pred = t.argmax(logits, dim=1)
            total_correct_exceptions += (pred[exceptions_indices] == Y2[exceptions_indices]).sum().item()
            total_samples_exceptions += exceptions_indices.size(0)

    avg_loss1 = total_loss1 / total_samples
    avg_loss2 = total_loss2 / total_samples
    overall_acc1 = total_correct1 / total_samples
    overall_acc2 = total_correct2 / total_samples

    avg_loss_exceptions = total_loss_exceptions / total_samples_exceptions if total_samples_exceptions > 0 else 0.0
    overall_acc_exceptions = total_correct_exceptions / total_samples_exceptions if total_samples_exceptions > 0 else 0.0

    return avg_loss1, avg_loss2, overall_acc1, overall_acc2, avg_loss_exceptions, overall_acc_exceptions
