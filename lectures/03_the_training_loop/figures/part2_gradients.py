"""Part 2 (architecture), why too deep hurts: the gradients of the weights, layer by layer.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part2_gradients.py [preview_dir]

- mnist_gradients.pdf: the network 784 -> 64 -> 64 -> 64 -> 64 -> 10 (five affine maps
  A_1, ..., A_5, four hidden layers of width 64), once with the sigmoid and once with the ReLU
  between them, with the same initial weights (PyTorch's default initialization, seed 0).
  For W_1, ..., W_5, one panel per layer with both activations on it, the histogram of
  log10 |dJ/d(W_k)_ij| at initialization, where J is the mean
  cross-entropy over the 60 000 training images.  Exact zeros (the pixels that are 0 in every
  training image, and dead units) have no logarithm and are left out; the script counts them.

Also printed, for the slide: the median of |dF/d(W_k)_ij| in each layer and the factor from one
layer to the next, the same for tanh (not drawn: it behaves like the ReLU here), the size of the
backpropagated signal dF/dh_k and of the forward signal x_k, and the test accuracy of the three
networks trained like the baseline of lecture 02 (Adam, learning rate 1e-3, batches of 64,
5 epochs).  About forty seconds on one CPU thread.
"""
import common as C
from common import P, np, plt, torch, nn

train, test = C.load_mnist("train"), C.load_mnist("test")
images, labels = train
sizes = [784] + [64] * 4 + [10]                                     # five affine maps
activations = {"sigmoid": nn.Sigmoid, "ReLU": nn.ReLU, "tanh": nn.Tanh}


def gradients(net):
    """dF/dW_k for k = 1..5 at the current weights, and the sizes of dF/dh_k and of x_k."""
    net.zero_grad()
    states, pre_activations, h = [images], [], images
    for module in net:
        h = module(h)
        if isinstance(module, nn.Linear):
            h.retain_grad()
            pre_activations.append(h)
        else:
            states.append(h)
    nn.functional.cross_entropy(h, labels).backward()
    weights = [m.weight.grad.clone() for m in net if isinstance(m, nn.Linear)]
    adjoint = [p.grad.norm(dim=1).mean().item() * len(images) for p in pre_activations]   # per image
    forward = [x.norm(dim=1).mean().item() for x in states]
    return weights, adjoint, forward


grads = {}
for name, activation in activations.items():
    net = C.make_classifier(sizes, activation=activation)
    weights, adjoint, forward = gradients(net)
    grads[name] = weights
    medians = np.array([g.abs()[g != 0].median().item() for g in weights])
    zeros = [(g == 0).float().mean().item() for g in weights]
    print(f"{name}, at initialization:")
    print("  median |dF/dW_k|, k = 1..5: " + ", ".join(f"{m:.1e}" for m in medians)
          + f"; W_5 / W_1 = {medians[-1] / medians[0]:.1e}")
    print("  factor from W_k to W_(k+1): " + ", ".join(f"{r:.1f}" for r in medians[1:] / medians[:-1]))
    print("  exact zeros left out: " + ", ".join(f"{100 * z:.1f}%" for z in zeros))
    print("  |dl/dh_k| per image, k = 1..5: " + ", ".join(f"{a:.1e}" for a in adjoint))
    print("  |x_k|, k = 0..4: " + ", ".join(f"{x:.2f}" for x in forward))

# The bound behind the slide: sigma'(t) = sigma(t) (1 - sigma(t)) <= 1/4, tanh'(t) = 1 - tanh(t)^2 <= 1.
t = torch.linspace(-10, 10, 20001)
print(f"max sigmoid' = {torch.sigmoid(t).mul(1 - torch.sigmoid(t)).max():.4f}, max tanh' = {(1 - torch.tanh(t) ** 2).max():.4f}")
print(f"PyTorch's default: Var W_ij = 1/(3n); sqrt(1/3 * 1/16) = {(1 / 48) ** 0.5:.3f}, its inverse {48 ** 0.5:.1f}")

for name, activation in activations.items():
    net, _, accuracy = C.train_mnist(C.make_classifier(sizes, activation=activation), train, test, verbose=False)
    print(f"{name}, trained like the baseline: test accuracy after 1 epoch {100 * accuracy[1]:.2f}%, "
          f"after 5 epochs {100 * accuracy[-1]:.2f}%", flush=True)

# ---------------------------------------------------------------------------------------
# The figure: one panel per layer k = 1..5, the sigmoid and the ReLU on the same axes
# ---------------------------------------------------------------------------------------
# Drawn at its size on the slide (the width of the text) and included unscaled.  All five
# layers, the output layer W_5 included, so that the panels match the formula on the slide.
bins = np.linspace(-12, -1, 67)                                     # a sixth of a decade
shown = 5
series = {"sigmoid": P.BLUE, "ReLU": P.ORANGE}
fig, axes = plt.subplots(1, shown, figsize=(108 / 25.4, 38 / 25.4), sharex=True, sharey=True)
fig.set_layout_engine("constrained", w_pad=0.01, h_pad=0.01, wspace=0.0)
for name, color in series.items():                                  # layer 5 on the left, layer 1 on the right
    for k, (ax, g) in enumerate(zip(axes[::-1], grads[name][:shown]), start=1):
        values = g.abs()[g != 0].log10().numpy()
        counts, _ = np.histogram(values, bins)
        ax.stairs(counts / counts.max(), bins, fill=True, color=color, alpha=0.55, lw=0, label=name)
        ax.axvline(np.median(values), color=color, lw=0.8)
        ax.set_title(f"Layer {k}", fontsize=8, pad=1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
for ax in axes:
    ax.set_xticks([-10, -6, -2], [r"$-10$", r"$-6$", r"$-2$"])
    ax.tick_params(labelsize=6.5, length=2, pad=1)
axes[0].set_xlim(bins[0], bins[-1])
axes[0].set_ylim(0, 1.05)
axes[shown - 4].legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=0.9,   # layer 4: empty on the left
               handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
fig.supxlabel(r"$\log|\partial J/\partial (W_k)_{ij}|$", fontsize=7.5)
C.save(fig, "mnist_gradients")
