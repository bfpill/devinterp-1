import torch as t
from tqdm import tqdm
import random
from two_p_model import TwoPMLP
from two_p_dataset import make_two_p_dataset, train_test_split
from dynamics import (
    ablate_other_modes_fourier_basis,
    ablate_other_modes_embed_basis,
    get_magnitude_modes,
)
from model_viz import viz_weights_modes, plot_mode_ablations, plot_magnitudes
from movie import run_movie_cmd
from dataclasses import dataclass, asdict, field
import json
from helpers import eval_model
from typing import List, Optional
import os 
import matplotlib.pyplot as plt

device = 'mps'

@dataclass
class ExperimentParamsTwoP:
    p1: int = 53
    p2: int = 53
    train_frac: float = 0.8
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
        suffix = f"P1={self.p1}_P2={self.p2}"
        
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


def train(model, train_dataset, test_dataset, params):
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

    loss_fn = t.nn.CrossEntropyLoss()
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
                loss_p1, loss_p2, acc_p1, acc_p2 = evaluate_model_logits(model, train_dataset, params)
                print(f"""
                      TRAIN SET:
                      Avg loss total: {avg_loss}, 
                      loss_p1: {loss_p1}, loss_p2: {loss_p2}, 
                      acc_p1: {acc_p1}, acc_p2: {acc_p2}
                      """)
                
                loss_p1, loss_p2, acc_p1, acc_p2 = evaluate_model_logits(model, test_dataset, params)
                print(f"""
                      TEST SET: 
                      Avg loss total: {avg_loss}, 
                      loss_p1: {loss_p1}, loss_p2: {loss_p2}, 
                      acc_p1: {acc_p1}, acc_p2: {acc_p2}
                      """)
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
        loss = loss_fn(out, labels)
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

    dataset = make_two_p_dataset(params)
    train_data, test_data = train_test_split(dataset, params)

    print("Training Data", [[k.item() for k in data] for data in train_data], [[k.item() for k in data] for data in test_data])

    model = train(
        model=model, train_dataset=train_data, test_dataset=test_data, params=params
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model_logits(model, dataset, params, batch_size=None):
    if batch_size is None:
        batch_size = params.batch_size

    model.eval()
    loss_fn = t.nn.CrossEntropyLoss() 

    total_loss1, total_loss2 = 0.0, 0.0
    total_correct1, total_correct2 = 0, 0
    total_samples = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        X1 = t.stack([sample[0] for sample in batch]).to(params.device)
        X2 = t.stack([sample[1] for sample in batch]).to(params.device)
        X3 = t.stack([sample[2] for sample in batch]).to(params.device)
        X4 = t.stack([sample[3] for sample in batch]).to(params.device)
        Y1 = t.stack([sample[4] for sample in batch]).to(params.device)
        Y2 = t.stack([sample[5] for sample in batch]).to(params.device)

        with t.no_grad():
            logits = model(X1, X2, X3, X4)
        
        logits1 = logits[:, :params.p1] 
        logits2 = logits[:, params.p1 : params.p1 + params.p2] 
        Y2_adjusted = Y2 - params.p1

        loss1 = loss_fn(logits1, Y1)
        loss2 = loss_fn(logits2, Y2_adjusted)

        pred1 = t.argmax(logits1, dim=1)
        pred2 = t.argmax(logits2, dim=1)
        batch_correct1 = (pred1 == Y1).sum().item()
        batch_correct2 = (pred2 == Y2_adjusted).sum().item()
        batch_size_actual = X1.size(0)

        total_loss1 += loss1.item() * batch_size_actual
        total_loss2 += loss2.item() * batch_size_actual
        total_correct1 += batch_correct1
        total_correct2 += batch_correct2
        total_samples += batch_size_actual

    avg_loss1 = total_loss1 / total_samples
    avg_loss2 = total_loss2 / total_samples
    overall_acc1 = total_correct1 / total_samples
    overall_acc2 = total_correct2 / total_samples

    return avg_loss1, avg_loss2, overall_acc1, overall_acc2

