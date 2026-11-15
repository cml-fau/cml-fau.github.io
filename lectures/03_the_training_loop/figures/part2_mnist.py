"""Part 2 (architecture), the digit classifier with other architectures.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part2_mnist.py [preview_dir]

Only the architecture box of the baseline changes.  The data (MNIST, pixels in [0, 1]), the
loss (cross-entropy), the optimizer (Adam, learning rate 1e-3, batches of 64, 5 epochs, the
same shuffling seed) and the initialization seed are those of lecture 02, through
common.train_mnist.

- mnist_depth_width.pdf: fully connected networks with 0, 1, 2, 4, 8, 16, 32 hidden layers
  of equal width 64, three initialization seeds each: test error against the depth.
  Also printed: the training error, and how much the outputs at initialization still depend
  on the image.
- mnist_cnn.pdf: a small convolutional network with c and 2c channels (c = 2, 4, 8; the slide
  quotes c = 8) against the fully connected networks with 2 hidden layers.

Every number quoted on the slides is printed here.  28 training runs of 5 epochs: about
nine minutes on one CPU thread.
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
# 1. Depth: 0 to 32 hidden layers of width 64, three initializations each
# ---------------------------------------------------------------------------------------
depths, depth_width, seeds = (0, 1, 2, 4, 8, 16, 32), 64, (0, 1, 2)
test_error, train_error = {}, {}                  # keyed by (depth, seed)
for seed in seeds:
    for depth in depths:
        net, _, test_accuracy = C.train_mnist(mlp(depth_width, depth, seed), train, test, verbose=False)
        test_error[depth, seed] = 100 * (1 - test_accuracy[-1])
        train_error[depth, seed] = 100 * (1 - C.accuracy(net, *train))
        print(f"MLP, {depth:2d} hidden layers of width {depth_width}, seed {seed}: {C.count_parameters(net):6d} parameters, "
              f"test error {test_error[depth, seed]:5.2f}%, training error {train_error[depth, seed]:5.2f}%",
              flush=True)


def over_seeds(table, depth):
    return np.array([table[depth, s] for s in seeds])


print(f"width {depth_width}: median over the seeds (range) of the test error | training error:")
for depth in depths:
    t, r = over_seeds(test_error, depth), over_seeds(train_error, depth)
    print(f"  {depth:2d} hidden layers: {np.median(t):5.2f}% ({t.min():5.2f} to {t.max():5.2f}) | {np.median(r):5.2f}%")
best = min(depths, key=lambda d: np.median(over_seeds(test_error, d)))
print(f"  smallest median test error: {best} hidden layers")
constant = 100 * (1 - (test[1] == test[1].mode().values).float().mean().item())
print(f"answering the most frequent digit of the test set for every image: test error {constant:.2f}%")

# why the deep networks do not train: at initialization, do different images still give different outputs?
ratio = {}
for depth in depths:
    with torch.no_grad():
        out = mlp(depth_width, depth)(test[0][:2000]).double()
    ratio[depth] = (out.std(0).mean() / out.abs().mean()).item()
print(f"width {depth_width}, at initialization, spread of the outputs over 2000 test images / their size: "
      + ", ".join(f"L={d}: {ratio[d]:.1e}" for d in depths)
      + f"; factor per layer from L=8 to L=16: {(ratio[16] / ratio[8]) ** (1 / 8):.2f}")
print(f"PyTorch's default initialization: Var W_ij = 1/(3 n), so sigma_w^2 = 1/3 and sqrt(sigma_w^2 / 2) = {6 ** -0.5:.2f}")

fig, ax = P.panels(figsize=(4.25, 2.55))                            # the full width of the slide
xs = np.arange(len(depths))                                         # the depths, evenly spaced
t = np.array([over_seeds(test_error, d) for d in depths])
ax.vlines(xs, t.min(1), t.max(1), color=P.BLUE, lw=2.6, alpha=0.35, zorder=1)        # range over the seeds
ax.plot(xs, np.median(t, 1), "-o", ms=3.5, lw=1.2, color=P.BLUE, zorder=3)
ax.set_yscale("log")
P.label(ax, "hidden layers $L$", "test error (%)", xlim=(-0.4, 6.4), ylim=(2, 120))
ax.set_xticks(xs, [str(d) for d in depths])
ax.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
ax.minorticks_off()
C.save(fig, "mnist_depth_width")

# width, with two hidden layers: the fully connected curve of the convolutional slide
widths = (8, 16, 32, 64, 128)
parameters = {(2, w): C.count_parameters(mlp(w, 2)) for w in widths}
error = {(2, w): test_error[2, 0] if w == depth_width else run(mlp(w, 2)) for w in widths}
baseline = error[2, 16]
assert parameters[2, 16] == 13002 and abs(baseline - 6.00) < 0.005  # the classifier of lecture 02: 94.0%
print("2 hidden layers, test error at width " + ", ".join(f"{w}: {error[2, w]:.2f}%" for w in widths)
      + f"; {parameters[2, 128]} parameters at width 128")


# ---------------------------------------------------------------------------------------
# 2. A convolutional network
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

fig, ax = P.panels(figsize=(2.9, 2.3))
ax.loglog([parameters[2, w] for w in widths], [error[2, w] for w in widths], "-o", ms=3.5, lw=1.2, color=P.BLUE,
          label="fully connected")
ax.loglog([cnn_parameters[c] for c in channels], [cnn_error[c] for c in channels], "-o", ms=3.5, lw=1.2,
          color=P.ORANGE, label="convolutional")
P.label(ax, "parameters $P$", "test error (%)", xlim=(5e2, 2.2e5), ylim=(0.8, 14), yticks=[1, 2, 5, 10],
        legend=dict(fontsize=7, loc="upper right"))
ax.set_yticklabels(["1", "2", "5", "10"])
ax.minorticks_off()
C.save(fig, "mnist_cnn")
print(f"run time: {time.time() - start:.0f} s")
