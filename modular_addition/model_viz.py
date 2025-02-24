import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from helpers import get_dft_matrix
import torch as t
from dynamics import get_magnitude_modes


def get_weights_modes(weights, P):
    cos, sin = get_dft_matrix(P)
    c1 = cos @ (weights @ weights.T)
    s1 = sin @ (weights @ weights.T)
    return c1, s1


def viz_weights_modes(weights, P, fname, title="Modes"):
    c1, s1 = get_weights_modes(weights, P)
    mags = get_magnitude_modes(weights, P)
    n_plots = (P // 2) + 1
    num_cols = int(math.ceil(math.sqrt(n_plots)))
    num_rows = int(math.ceil(n_plots / num_cols))
    colormap = plt.cm.rainbow
    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))
    colors = [colormap(i) for i in np.linspace(0, 1, P)]
    for mode in range(n_plots):
        cos = c1[mode]
        sin = s1[mode]
        plt.subplot(num_rows, num_cols, mode + 1)
        color_idx = [(i * mode) % P for i in range(P)]
        color_for_mode = [colors[idx] for idx in color_idx]
        plt.scatter(cos.numpy(), sin.numpy(), c=color_for_mode)
        plt.title(f"Mode {mode} {mags[mode]:.4f}", fontsize=8)
    plt.suptitle(title)
    norm = Normalize(vmin=0, vmax=P)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.01, 0.7])
    plt.colorbar(sm, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    # plt.savefig(fname)
    return fig
    # plt.close()


def plot_mode_ablations(mode_loss_history, fname):
    plt.clf()
    plt.figure(figsize=(8, 8))
    num_plots = len(mode_loss_history)
    colormap = cm.get_cmap("rainbow", num_plots)

    for idx, history in enumerate(mode_loss_history):
        color = colormap(idx)
        plt.plot(
            list(history.keys()),
            list(history.values()),
            color=color,
            label=f"step {idx}",
        )
    plt.xlabel("Fourier mode")
    plt.ylabel("Test loss when only keeping this mode")
    plt.legend()
    plt.savefig(fname)
    # plt.close()


def plot_magnitudes(magnitude_history, p, fname):
    plt.clf()
    plt.figure(figsize=(8, 8))
    num_plots = len(magnitude_history)
    colormap = cm.get_cmap("rainbow", num_plots)
    for idx, history in enumerate(magnitude_history):
        color = colormap(idx)
        plt.plot(list(history)[: p // 2 + 1], color=color, label=f"step {idx}")
    plt.xlabel("Fourier mode")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.savefig(fname)
    # plt.close()


def plot_ft_input_output_activations(saved_activations, a, P, fname):
    # dft of y_pre_unembed
    activations = [
        (act[0][1], act[1]) for act in saved_activations.items() if act[0][0] == a
    ]  # b, x
    # sort by b
    activations.sort(key=lambda x: x[0])
    assert len(activations) == P
    # stack over b
    activations = t.stack([act[1] for act in activations])  # p x embed_dim
    mags = get_magnitude_modes(activations, P)
    cos, sin = get_dft_matrix(P)
    c1 = cos @ (activations @ activations.T)
    s1 = sin @ (activations @ activations.T)
    n_plots = (P // 2) + 1
    num_cols = int(math.ceil(math.sqrt(n_plots)))
    num_rows = int(math.ceil(n_plots / num_cols))
    colormap = plt.cm.rainbow
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    colors = [colormap(i) for i in np.linspace(0, 1, P)]
    for mode in range(n_plots):
        cos = c1[mode]
        sin = s1[mode]
        plt.subplot(num_rows, num_cols, mode + 1)
        color_idx = [(i * mode) % P for i in range(P)]
        color_for_mode = [colors[idx] for idx in color_idx]
        plt.scatter(cos.numpy(), sin.numpy(), c=color_for_mode)
        plt.title(f"Mode {mode} {mags[mode]:.4f}", fontsize=8)
    plt.suptitle("Output activations FT")
    norm = Normalize(vmin=0, vmax=P)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.01, 0.7])
    plt.colorbar(sm, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(fname)
    # plt.close()
