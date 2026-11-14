"""Helpers shared by the figure scripts of the lecture slides the_training_loop.tex.

Every script in this folder starts with

    import common as C
    from common import P, np, plt, torch, nn

and writes its figures with C.save(fig, "name"): a PDF in ../images/ and, if a preview
directory was given on the command line, a PNG copy there.  The figures reuse the house
style of the notebooks (notebooks/plotting.py), so the slides and the notebooks look alike.

The two running examples of lectures 02 and 03 live here: the target g(x) = sin(3x) + 0.3x
on [-1, 1], and the digit classifier 784 -> 16 -> 16 -> 10 trained on MNIST with Adam
(learning rate 1e-3, batches of 64, 5 epochs), which reaches 94.0% test accuracy.
"""
import gzip
import pathlib
import sys
import urllib.request
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from torch import nn
import matplotlib
import plotting as P                       # the course palette and figure helpers

matplotlib.use("agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(1)                   # one thread keeps the runs reproducible

OUT = pathlib.Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(exist_ok=True)
PREVIEW = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
if PREVIEW:
    PREVIEW.mkdir(parents=True, exist_ok=True)
trapz = getattr(np, "trapezoid", None) or np.trapz


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


def expit(t):
    """The sigmoid 1 / (1 + exp(-t)), written so that large |t| does not overflow."""
    return np.exp(-np.logaddexp(0, -t))


# ---------------------------------------------------------------------------------------
# Running example 1: the target of lecture 02, sampled at 160 equispaced points
# ---------------------------------------------------------------------------------------
def target(x):
    return np.sin(3 * x) + 0.3 * x


x = np.linspace(-1, 1, 160)
y = target(x)


def hats(x, m):
    """m hat functions on an equispaced mesh of [-1, 1] (Courant's family of lecture 02)."""
    nodes = np.linspace(-1, 1, m)
    spacing = nodes[1] - nodes[0]
    return np.maximum(0, 1 - np.abs(x[:, None] - nodes) / spacing)


# ---------------------------------------------------------------------------------------
# Running example 2: handwritten digits (lecture 02, "Everything at once")
# ---------------------------------------------------------------------------------------
def load_mnist(split):
    """Images as floats in [0, 1] with shape (N, 784) and labels as integers 0-9."""
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/" + ("train" if split == "train" else "t10k")
    arrays = {}
    for kind, offset in (("images", 16), ("labels", 8)):
        path = ROOT / "notebooks" / "data" / f"mnist-{split}-{kind}.gz"     # shared with notebooks 10 and 17
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(f'{base}-{kind}-idx{3 if kind == "images" else 1}-ubyte.gz', path)
        arrays[kind] = np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.uint8, offset=offset)
    images = torch.tensor(arrays["images"].reshape(-1, 784).copy(), dtype=torch.float32) / 255
    labels = torch.tensor(arrays["labels"].copy(), dtype=torch.int64)
    return images, labels


def make_classifier(sizes, seed=0):
    """Fully connected ReLU network with the given layer sizes, input first."""
    torch.manual_seed(seed)
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
    return nn.Sequential(*layers[:-1])                              # the outputs are raw scores


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


def accuracy(net, images, labels, chunk=10_000):
    """Fraction of correct predictions, evaluated in chunks so that large models fit in memory."""
    net.eval()
    with torch.no_grad():
        hits = sum((net(images[i:i + chunk]).argmax(1) == labels[i:i + chunk]).sum().item()
                   for i in range(0, len(images), chunk))
    net.train()
    return hits / len(images)


def train_mnist(net, train, test, epochs=5, batch_size=64, lr=1e-3, optimizer=None, loss_fn=None,
                shuffle_seed=1, verbose=True):
    """The training loop of lecture 02.  With the defaults and make_classifier([784, 16, 16, 10])
    it reproduces the baseline of the slides: 4690 steps and 94.0% test accuracy.

    train and test are (images, labels) pairs.  Returns the network, the loss of every
    mini-batch and the test accuracy before training and after each epoch.
    """
    (train_images, train_labels), (test_images, test_labels) = train, test
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn or nn.CrossEntropyLoss()                      # softmax and -log p_y in one stable call
    generator = torch.Generator().manual_seed(shuffle_seed)
    batch_losses, test_accuracy = [], [accuracy(net, test_images, test_labels)]
    for epoch in range(epochs):
        order = torch.randperm(len(train_images), generator=generator)
        for batch in order.split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(net(train_images[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        test_accuracy.append(accuracy(net, test_images, test_labels))
        if verbose:
            print(f"  epoch {epoch + 1}: test accuracy = {100 * test_accuracy[-1]:.2f}%")
    return net, np.array(batch_losses), np.array(test_accuracy)
