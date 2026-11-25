"""Helpers shared by the figure scripts of the lecture slides self_attention_and_transformers.tex.

Every script in this folder starts with

    import common as C
    from common import P, np, plt, torch, nn

and writes its figures with C.save(fig, "name"): a PDF in ../images/ and, if a preview
directory was given on the command line, a PNG copy there.  The figures reuse the house
style of the notebooks (notebooks/plotting.py), so the slides and the notebooks look alike.

What the scripts share lives here: the paths, the sizes of a slide, and the handwritten digits
of lectures 02 to 04.  The language model itself, GPT-2 with its tokenizer and the weights
published by OpenAI, is in gpt2.py next to this file.
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

LECTURE = pathlib.Path(__file__).resolve().parents[1]
OUT = LECTURE / "images"
OUT.mkdir(exist_ok=True)
SHARED = ROOT / "notebooks" / "data"       # downloads shared with the notebooks (not in the repository)
DATA = LECTURE / "data"                    # data of this lecture only (not in the repository)
PREVIEW = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
if PREVIEW:
    PREVIEW.mkdir(parents=True, exist_ok=True)

# A slide is 128 mm x 96 mm; the text is 108 mm wide.  Figures are drawn at the size they have
# on the slide and included unscaled, so a font size in a script is a font size on the slide.
MM = 1 / 25.4                              # inches per millimetre: figsize=(108 * C.MM, 50 * C.MM)
TEXT_WIDTH = 108 * MM


def save(fig, name):
    """Write the figure with a transparent background, so the slide's own colour shows through."""
    fig.savefig(OUT / f"{name}.pdf", transparent=True)
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=170, transparent=True)
    plt.close(fig)
    print("wrote", name)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


# ---------------------------------------------------------------------------------------
# The handwritten digits of lectures 02 to 04
# ---------------------------------------------------------------------------------------
def load_mnist(split):
    """Images as floats in [0, 1] with shape (N, 784) and labels as integers 0-9."""
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/" + ("train" if split == "train" else "t10k")
    arrays = {}
    for kind, offset in (("images", 16), ("labels", 8)):
        path = SHARED / f"mnist-{split}-{kind}.gz"                          # shared with notebooks 10 and 17
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(f'{base}-{kind}-idx{3 if kind == "images" else 1}-ubyte.gz', path)
        arrays[kind] = np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.uint8, offset=offset)
    images = torch.tensor(arrays["images"].reshape(-1, 784).copy(), dtype=torch.float32) / 255
    labels = torch.tensor(arrays["labels"].copy(), dtype=torch.int64)
    return images, labels
