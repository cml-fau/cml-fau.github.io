"""Part 2 (architecture), the digit classifier with other architectures.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part2_mnist.py [preview_dir]

Only the architecture box of the baseline changes.  The data (MNIST, pixels in [0, 1]), the
loss (cross-entropy), the optimizer (Adam, learning rate 1e-3, batches of 64, 5 epochs, the
same shuffling seed) and the initialization seed are those of lecture 02, through
common.train_mnist.

- mnist_depth_width.pdf: fully connected networks with 1, 2 and 4 hidden layers of equal
  width 8, 16, 32, 64, 128: test error against the number of parameters.
- mnist_cnn.pdf: a small convolutional network with c and 2c channels (c = 2, 4, 8) against
  the fully connected networks, and the same two models when one fixed permutation is
  applied to the 784 pixels of every image.

Every number quoted on the slides is printed here.  22 training runs of 5 epochs: about
seven minutes on one CPU thread.
"""
import time

import common as C
from common import P, np, plt, torch, nn

start = time.time()
train, test = C.load_mnist("train"), C.load_mnist("test")


def run(net, data=(train, test)):
    """Train with the settings of the baseline; return the test error in percent."""
    net, _, test_accuracy = C.train_mnist(net, *data, verbose=False)
    return 100 * (1 - test_accuracy[-1])


def mlp(width, depth, seed=0):
    return C.make_classifier([784] + [width] * depth + [10], seed=seed)


def cnn(channels, seed=0):
    """Two 5x5 convolutions with max-pooling, then one linear layer: LeNet without its dense layers."""
    torch.manual_seed(seed)
    c = channels
    return nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),                               # 784 pixels back on their grid
        nn.Conv2d(1, c, 5), nn.ReLU(), nn.MaxPool2d(2),             # (c, 24, 24) -> (c, 12, 12)
        nn.Conv2d(c, 2 * c, 5), nn.ReLU(), nn.MaxPool2d(2),         # (2c, 8, 8) -> (2c, 4, 4)
        nn.Flatten(), nn.Linear(2 * c * 16, 10))


# ---------------------------------------------------------------------------------------
# 1. Depth against width: fully connected networks
# ---------------------------------------------------------------------------------------
widths, depths = (8, 16, 32, 64, 128), (1, 2, 4)
parameters, error = {}, {}
for depth in depths:
    for width in widths:
        net = mlp(width, depth)
        parameters[depth, width] = C.count_parameters(net)
        error[depth, width] = run(net)
        print(f"MLP, {depth} hidden layers of width {width:3d}: {parameters[depth, width]:6d} parameters, "
              f"test error {error[depth, width]:.2f}%, accuracy {100 - error[depth, width]:.2f}%", flush=True)
baseline = error[2, 16]
assert parameters[2, 16] == 13002 and abs(baseline - 6.00) < 0.005     # the classifier of lecture 02: 94.0%

# how much does the baseline move when only the random initial weights change?
seeds = [baseline] + [run(mlp(16, 2, seed=seed)) for seed in (1, 2)]
print("baseline 784-16-16-10 with initialization seeds 0, 1, 2: test accuracy "
      + ", ".join(f"{100 - e:.2f}%" for e in seeds))

for width in widths:
    values = [error[depth, width] for depth in depths]
    print(f"width {width:3d}: test error with 1, 2, 4 hidden layers = "
          + ", ".join(f"{v:.2f}%" for v in values) + f"; spread {max(values) - min(values):.2f} points")
first_layer = {key: 100 * (784 * key[1] + key[1]) / count for key, count in parameters.items()}
print(f"share of the parameters in the first layer: between {min(first_layer.values()):.0f}% "
      f"and {max(first_layer.values()):.0f}%; baseline {first_layer[2, 16]:.0f}%")

fig, ax = P.panels(figsize=(3.1, 2.4))
for depth, color in zip(depths, (P.AQUA, P.BLUE, P.MAGENTA)):
    ax.loglog([parameters[depth, w] for w in widths], [error[depth, w] for w in widths], "-o", ms=3.5, lw=1.2,
              color=color, label=f"{depth} hidden layer" + ("s" if depth > 1 else ""))
for width in widths:                                                # the widths, under the points of the curves
    ax.text(parameters[2, width], 1.62, f"{width}", fontsize=6.5, color=P.NOTE, ha="center")
ax.text(4.9e3, 1.62, "width", fontsize=6.5, color=P.NOTE, ha="right")
ax.plot(parameters[2, 16], baseline, "o", ms=8, mfc="none", mec=P.INK, mew=0.9)
ax.annotate("baseline", (parameters[2, 16], baseline), xytext=(5.2e3, 3.0), fontsize=7, ha="center",
            arrowprops=dict(arrowstyle="-", color=P.INK, lw=0.6, shrinkA=1, shrinkB=4.5))
P.label(ax, "parameters $P$", "test error (%)", title="Depth and width", xlim=(3.5e3, 2.2e5),
        ylim=(1.5, 14), yticks=[2, 5, 10], legend=dict(fontsize=7, loc="upper right"))
ax.set_yticklabels(["2", "5", "10"])
ax.minorticks_off()
C.save(fig, "mnist_depth_width")


# ---------------------------------------------------------------------------------------
# 2. A convolutional network, and the same images with their pixels shuffled
# ---------------------------------------------------------------------------------------
channels = (2, 4, 8)
cnn_parameters, cnn_error = {}, {}
for c in channels:
    net = cnn(c)
    cnn_parameters[c] = C.count_parameters(net)
    cnn_error[c] = run(net)
    print(f"CNN with {c} and {2 * c} channels: {cnn_parameters[c]:5d} parameters, test error {cnn_error[c]:.2f}%, "
          f"accuracy {100 - cnn_error[c]:.2f}%", flush=True)
print(f"CNN with 8 and 16 channels: {cnn_parameters[8]} parameters = {100 * cnn_parameters[8] / 13002:.0f}% of the baseline's 13002; "
      f"its error is {baseline / cnn_error[8]:.1f} times smaller")
smaller = [key for key in parameters if error[key] <= cnn_error[2]]
print(f"CNN with 2 and 4 channels ({cnn_parameters[2]} parameters, error {cnn_error[2]:.2f}%): the fully connected "
      "networks with at most this error are "
      + (", ".join(f"{d}x{w} ({parameters[d, w]} parameters)" for d, w in sorted(smaller, key=parameters.get)) or "none"))
best = min(error, key=error.get)
print(f"best fully connected network: {best[0]} hidden layers of width {best[1]}, {parameters[best]} parameters, "
      f"test error {error[best]:.2f}%")

# one fixed permutation of the 784 pixels, applied to every training and test image
permutation = torch.randperm(784, generator=torch.Generator().manual_seed(0))
shuffled = ((train[0][:, permutation], train[1]), (test[0][:, permutation], test[1]))
mlp_shuffled = run(mlp(16, 2), shuffled)
cnn_shuffled = run(cnn(8), shuffled)
print(f"shuffled pixels: baseline MLP test error {mlp_shuffled:.2f}% (accuracy {100 - mlp_shuffled:.2f}%; original pixels "
      f"{baseline:.2f}%); CNN with 8 and 16 channels {cnn_shuffled:.2f}% (accuracy {100 - cnn_shuffled:.2f}%; "
      f"original pixels {cnn_error[8]:.2f}%)")

fig, (ax, ax_bar) = P.panels(1, 2, size=(2.9, 1.85))
ax.loglog([parameters[2, w] for w in widths], [error[2, w] for w in widths], "-o", ms=3.5, lw=1.2, color=P.BLUE)
ax.loglog([cnn_parameters[c] for c in channels], [cnn_error[c] for c in channels], "-o", ms=3.5, lw=1.2, color=P.ORANGE)
ax.plot(parameters[2, 16], baseline, "o", ms=8, mfc="none", mec=P.INK, mew=0.9)
ax.annotate("baseline", (parameters[2, 16], baseline), xytext=(7, 4), textcoords="offset points", fontsize=7)
ax.text(1.9e5, 0.95, "fully connected,\n2 hidden layers", fontsize=7, color=P.BLUE, ha="right", va="bottom")
for c in channels:
    ax.annotate(f"$c={c}$", (cnn_parameters[c], cnn_error[c]), xytext=(5, 3), textcoords="offset points",
                fontsize=7, color=P.ORANGE)
ax.text(650, 1.0, "convolutional", fontsize=7, color=P.ORANGE)
P.label(ax, "parameters $P$", "test error (%)", title="Test error against parameters", xlim=(5e2, 2.2e5),
        ylim=(0.8, 14), yticks=[1, 2, 5, 10])
ax.set_yticklabels(["1", "2", "5", "10"])
ax.minorticks_off()

groups = [("fully connected\n(baseline)", baseline, mlp_shuffled, P.BLUE), ("convolutional\n($c=8$)", cnn_error[8], cnn_shuffled, P.ORANGE)]
for k, (name, original, permuted, color) in enumerate(groups):
    ax_bar.bar(k - 0.19, original, 0.36, color=color, label="original" if k == 0 else None)
    ax_bar.bar(k + 0.19, permuted, 0.36, color=color, alpha=0.4, hatch="////", edgecolor="white", linewidth=0,
               label="shuffled" if k == 0 else None)
    for x_bar, value in ((k - 0.19, original), (k + 0.19, permuted)):
        ax_bar.annotate(f"{value:.1f}", (x_bar, value), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
P.label(ax_bar, None, "test error (%)", title="Original and shuffled pixels", xticks=[0, 1], xlim=(-0.6, 1.6),
        ylim=(0, 1.5 * max(baseline, mlp_shuffled, cnn_shuffled)), yticks=[0, 2, 4, 6],
        legend=dict(fontsize=6.5, loc="upper center", ncols=2))
ax_bar.set_xticklabels([name for name, *_ in groups], fontsize=7)
legend = ax_bar.get_legend()
for handle in legend.legend_handles:
    handle.set_facecolor(P.GREY)
legend.legend_handles[1].set_alpha(0.4)
C.save(fig, "mnist_cnn")
print(f"run time: {time.time() - start:.0f} s")
