import torch as t
from tqdm import tqdm
import random
from model import MLP
from dataset import count_rands_coverage, make_dataset, train_test_split, generate_rands, get_exceptions_split
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
class ExperimentParams:
    p: int = 53
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
    use_exceptions: bool = False
    lambda_hat: Optional[float] = None  # Gets populated by running SGLD estimator code
    test_loss: Optional[float] = None   # Gets populated by running SGLD estimator code
    train_loss: Optional[float] = None  # Gets populated by running SGLD estimator code
    a: int = None
    b: int = None
    num_rands: int = None
    rands: List[int] = field(default_factory=list)
    rand_labels: dict = field(default_factory=dict)

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
        return ExperimentParams(**class_dict)

    def get_suffix(self, checkpoint_no=None):
        suffix = f"P{self.p}_exceptions={self.use_exceptions}_num_exceptions={self.num_rands}_a={self.a}_b={self.b}"
        
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
    model_copy = MLP(params)
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

                test_exceps, test_normal = get_exceptions_split(test_dataset, params)
                test_normal_acc, test_normal_total = test(model, test_normal)
                test_excep_acc, test_excep_total = test(model, test_exceps)

                train_exceps, train_normal = get_exceptions_split(train_dataset, params)
                train_normal_acc, train_normal_total = test(model, train_normal)
                train_excep_acc, train_excep_total  = test(model, train_exceps)

                print(f"Batch: {i} | Loss: {avg_loss} | Test Acc: {test_normal_acc} (/{test_normal_total}), Test Excep Acc: {test_excep_acc} (/{test_excep_total})")
                print(f"Batch: {i} | Loss: {avg_loss} | Train Acc: {train_normal_acc} (/{train_normal_total}) | Train Excep Acc: {train_excep_acc} (/{train_excep_total})")
                avg_loss = 0

            if i % track_every == 0:
                if params.magnitude:
                    mags = get_magnitude_modes(
                        model.embedding.weight.detach().cpu(), params.p
                    )
                    magnitude_history.append(mags)
 
        model.train()

        batch_idx = random.choices(idx, k=params.batch_size)
        X_1 = t.stack([train_dataset[b][0][0] for b in batch_idx]).to(params.device)
        X_2 = t.stack([train_dataset[b][0][1] for b in batch_idx]).to(params.device)
        Y = t.stack([train_dataset[b][1] for b in batch_idx]).to(params.device)

        optimizer.zero_grad()
        out = model(X_1, X_2)
        loss = loss_fn(out, Y)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    # test_exceps, test_normal = get_exceptions_split(test_dataset, params)
    # test_acc, test_excep_acc = test(model, test_exceps), test(model, test_normal) 
    # print(f"Final Val Acc: {test_normal}/{normal_total}, excep acc: {test_exceps}/{test_excep_total}")

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
    model = MLP(params)
    print(f"Number of parameters: {count_params(model)}")
    
    rands, rand_labels = generate_rands(params)
    params.rands = rands
    params.rand_labels = rand_labels

    dataset = make_dataset(params)
    train_data, test_data = train_test_split(dataset, params)
    count_rands_coverage(train_data, test_data, params)

    print("Training Data", train_data[:10], test_data[:10])

    model = train(
        model=model, train_dataset=train_data, test_dataset=test_data, params=params
    )
    
    fname = f"models/model_{params.get_suffix()}.pt"
    t.save(model.state_dict(), fname)
    if params.do_viz_weights_modes:
        fig = viz_weights_modes(
            model.embedding.weight.detach().cpu(),
            params.p,
            f"plots/final_embeddings_{params.get_suffix()}.png",
        )
        plt.show(fig)


def p_sweep_exp(p_values, params, psweep):
    if not os.path.exists(f"exp_params/{psweep}"):
        os.makedirs(f"exp_params/{psweep}")
    for p in p_values:
        params.p = p
        params.save_to_file(f"exp_params/{psweep}/{params.a}*{params.b}={p}_{params.run_id}.json")
        run_exp(params)


def frac_sweep_exp(train_fracs, params, psweep):
    if not os.path.exists(f"exp_params/{psweep}"):
        os.makedirs(f"exp_params/{psweep}")
    for frac in train_fracs:
        params.train_frac = frac
        params.save_to_file(f"exp_params/{psweep}/{frac}_{params.run_id}.json")
        run_exp(params)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
