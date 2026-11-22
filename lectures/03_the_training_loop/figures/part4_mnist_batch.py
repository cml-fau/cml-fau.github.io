"""Figures of part 4 (optimization) on the digit classifier: mnist_batch_fixed_step and
mnist_batch_fixed_ratio.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part4_mnist_batch.py [preview_dir]

The data, the network 784 -> 16 -> 16 -> 10, the cross-entropy and the five epochs of the
baseline of lecture 02 are kept; only the optimizer changes: its batch size B, its step
eta, and Adam or plain SGD.  mnist_batch_fixed_step: a fixed step.  mnist_batch_fixed_ratio:
SGD with eta = c B, which keeps the ratio eta / B = c.  Left in each: the training loss J(w) along
the runs that start from SGD with eta = 0.05 at B = 64 and change B.  Right: the test error
after five epochs against B.  Every run prints
its numbers.  The runs go to a pool of one-thread processes: about seven minutes on 16 cores.
"""
import os
import time
from multiprocessing import Pool

from matplotlib.colors import LinearSegmentedColormap

import common as C
from common import P, np, plt, torch, nn

train, test = C.load_mnist("train"), C.load_mnist("test")
N = len(train[0])
EPOCHS = 5
PER_EPOCH = 50                     # evaluations of J per epoch, for the curves of the bottom row
cross_entropy = nn.CrossEntropyLoss()


def training_loss(net):
    with torch.no_grad():
        return cross_entropy(net(train[0]), train[1]).item()


def run(key):
    """Five epochs from the initial weights of the baseline, with the loop and the shuffling of
    C.train_mnist.  Returns the test accuracy in percent and, if `curve`, the training loss J
    before training and at PER_EPOCH equally spaced points of every epoch."""
    method, batch, eta, curve, seed = key
    start = time.time()
    net = C.make_classifier([784, 16, 16, 10], seed=seed)
    optimizer = (torch.optim.Adam if method == "Adam" else torch.optim.SGD)(net.parameters(), lr=eta)
    generator = torch.Generator().manual_seed(1 if seed == 0 else seed + 10)
    marks = np.linspace(0, N, PER_EPOCH + 1).round().astype(int)[1:]     # samples seen at each evaluation
    losses = [training_loss(net)] if curve else []
    for epoch in range(EPOCHS):
        seen, next_mark = 0, 0
        for indices in torch.randperm(N, generator=generator).split(batch):
            optimizer.zero_grad()
            cross_entropy(net(train[0][indices]), train[1][indices]).backward()
            optimizer.step()
            seen += len(indices)
            while curve and next_mark < PER_EPOCH and seen >= marks[next_mark]:
                losses.append(training_loss(net))
                next_mark += 1
    accuracy = 100 * C.accuracy(net, *test)
    line = (f"  {method:<4} B = {batch:<5} eta = {eta:<9.4g} eta/B = {eta / batch:.2e}  seed = {seed}  "
            f"test accuracy = {accuracy:.2f}%  J = {training_loss(net):.4f}  ({time.time() - start:.0f} s)")
    return key, accuracy, np.array(losses), line


# ---------------------------------------------------------------------------------------
# The runs.  Top left: a fixed step, Adam with the baseline eta = 1e-3 and SGD with two
# steps.  Top right: SGD with eta = c B, for c = eta_64 / 64 (the step eta_64 at B = 64).
# Bottom: SGD with eta = 0.05 (left) and eta = (0.05 / 64) B (right).
# ---------------------------------------------------------------------------------------
batches = [2 ** k for k in range(13)]                               # 1, 2, 4, ..., 4096
with_full_batch = batches + [N]
fixed = {"Adam, $\\eta=10^{-3}$": ("Adam", 1e-3, P.INK, "o"),
         "SGD, $\\eta=0.05$": ("SGD", 0.05, P.AQUA, "s"),
         "SGD, $\\eta=0.2$": ("SGD", 0.2, P.ORANGE, "^")}
scaled = {0.0125: P.BLUE, 0.05: P.AQUA, 0.2: P.ORANGE}               # eta_64; c = eta_64 / 64
curve_batches = [1, 4, 16, 64, 256, 1024]


def key(method, batch, eta):
    """A run is (method, B, eta, whether J is recorded, seed); J is recorded for the bottom row."""
    return method, batch, eta, method == "SGD" and batch in curve_batches and eta in (0.05, 0.05 * batch / 64), 0


keys = {key(method, batch, eta) for method, eta, _, _ in fixed.values() for batch in with_full_batch}
keys |= {key("SGD", batch, eta_64 * batch / 64) for eta_64 in scaled for batch in batches}
keys |= {(method, 64, eta, False, seed) for method, eta in (("Adam", 1e-3), ("SGD", 0.05)) for seed in (1, 2)}


def compute():
    """All the runs, in a pool of one-thread processes, the longest (small B) first."""
    start, results = time.time(), {}
    with Pool(min(15, os.cpu_count())) as pool:
        for k, accuracy, losses, line in pool.imap_unordered(run, sorted(keys, key=lambda k: (k[1], -k[3]))):
            results[k] = accuracy, losses
            print(line, flush=True)
    print(f"{len(keys)} runs in {(time.time() - start) / 60:.1f} minutes")
    return results


def report(results):
    accuracy = lambda method, batch, eta: results[key(method, batch, eta)][0]
    print("test accuracy (%) after five epochs, B = 1, 2, 4, ...:")
    for method, eta, _, _ in fixed.values():
        values = [accuracy(method, batch, eta) for batch in with_full_batch]
        print(f"  {method} eta = {eta}: " + ", ".join(f"{v:.1f}" for v in values)
              + f"   best B = {with_full_batch[int(np.argmax(values))]}")
    for eta_64 in scaled:
        values = [accuracy("SGD", batch, eta_64 * batch / 64) for batch in batches]
        print(f"  SGD eta = ({eta_64}/64) B: " + ", ".join(f"{v:.1f}" for v in values))
    for method, eta in (("Adam", 1e-3), ("SGD", 0.05)):
        values = [accuracy(method, 64, eta)] + [results[method, 64, eta, False, seed][0] for seed in (1, 2)]
        print(f"  {method} eta = {eta}, B = 64, seeds 0, 1, 2: " + ", ".join(f"{v:.2f}" for v in values)
              + f"; range {max(values) - min(values):.2f} points")


def figures(results):
    accuracy = lambda method, batch, eta: results[key(method, batch, eta)][0]
    shades = LinearSegmentedColormap.from_list("aqua", ["#b5e8d4", P.AQUA, "#0b4d35"])     # lighter for smaller B
    epochs = np.linspace(0, EPOCHS, EPOCHS * PER_EPOCH + 1)

    def error_axes(ax, sizes, title, legend):
        """Right panel: the test error against B."""
        P.label(ax, "batch size $B$", "test error (%)", title=title, legend=legend)
        ticks = [b for b in (1, 4, 16, 64, 256, 1024, 4096, N) if b in sizes]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["60 000" if b == N else str(b) for b in ticks], fontsize=7)
        ax.minorticks_off()
        ax.set_yticks([5, 10, 20, 50, 90])
        ax.set_yticklabels(["5", "10", "20", "50", "90"])
        ax.set_ylim(3.5, 100)

    def loss_axes(ax, step, title):
        """Left panel: J along the runs of SGD with the step `step(B)`."""
        for i, batch in enumerate(curve_batches):
            ax.semilogy(epochs, results[key("SGD", batch, step(batch))][1], color=shades(i / (len(curve_batches) - 1)),
                        lw=0.9, label=f"$B={batch}$")
        P.label(ax, "epoch", "training loss $J(w)$", title=title,
                legend=dict(fontsize=6.3, loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.4))
        ax.minorticks_off()
        ax.set_yticks([0.2, 0.5, 1, 2])
        ax.set_yticklabels(["0.2", "0.5", "1", "2"])
        ax.set_ylim(0.12, 7)
        ax.set_xlim(0, EPOCHS)

    # Growing B at a fixed step
    fig, (ax_loss, ax_error) = P.panels(2, size=(3.1, 3.4))
    for name, (method, eta, color, marker) in fixed.items():
        ax_error.loglog(with_full_batch, [100 - accuracy(method, batch, eta) for batch in with_full_batch],
                        "-" + marker, color=color, ms=3, lw=1.1, label=name)
    error_axes(ax_error, with_full_batch, r"A fixed step $\eta$",
               legend=dict(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.42, 1)))
    loss_axes(ax_loss, lambda batch: 0.05, r"SGD, $\eta=0.05$")
    C.save(fig, "mnist_batch_fixed_step")

    # Keeping the ratio eta / B fixed
    fig, (ax_loss, ax_error) = P.panels(2, size=(3.1, 3.4))
    for eta_64, color in scaled.items():
        ax_error.loglog(batches, [100 - accuracy("SGD", batch, eta_64 * batch / 64) for batch in batches],
                        "-o", color=color, ms=3, lw=1.1, label=rf"$c={eta_64}/64$")
    error_axes(ax_error, batches, r"SGD with $\eta=c\,B$", legend=dict(fontsize=6.3, loc="upper left"))
    loss_axes(ax_loss, lambda batch: 0.05 * batch / 64, r"SGD, $\eta=(0.05/64)\,B$")
    C.save(fig, "mnist_batch_fixed_ratio")


if __name__ == "__main__":
    results = compute()
    report(results)
    figures(results)
