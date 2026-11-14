"""Figure of part 4 (optimization) on the digit classifier: mnist_batch_size.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part4_mnist_batch.py [preview_dir]

The data, the network 784 -> 16 -> 16 -> 10, the cross-entropy and the five epochs of the
baseline of lecture 02 are kept; only the optimizer changes: its batch size B, its step
eta, and Adam or plain SGD.  Every number quoted on the slide is printed here.  About four
minutes on one CPU thread.
"""
import time

import common as C
from common import P, np, plt, torch, nn

train, test = C.load_mnist("train"), C.load_mnist("test")
N = len(train[0])
cross_entropy = nn.CrossEntropyLoss()
EPOCHS = 5


def run(method, batch, eta, init_seed=0, shuffle_seed=1):
    """Five epochs from the initial weights of the baseline.  Returns the number of steps, the
    test accuracy in percent and the loss on the whole training set."""
    net = C.make_classifier([784, 16, 16, 10], seed=init_seed)
    optimizer = (torch.optim.Adam if method == "Adam" else torch.optim.SGD)(net.parameters(), lr=eta)
    start = time.time()
    net, losses, test_accuracy = C.train_mnist(net, train, test, epochs=EPOCHS, batch_size=batch, optimizer=optimizer,
                                               shuffle_seed=shuffle_seed, verbose=False)
    with torch.no_grad():
        training_loss = cross_entropy(net(train[0]), train[1]).item()
    print(f"  {method:<4} B = {batch:<5} eta = {eta:<8.5g} eta/B = {eta / batch:.2e}  steps = {len(losses):<5} "
          f"test accuracy = {100 * test_accuracy[-1]:.2f}%  training loss = {training_loss:.4f}  "
          f"({time.time() - start:.0f} s)", flush=True)
    return len(losses), 100 * test_accuracy[-1], training_loss


results = {}


def result(method, batch, eta):
    key = (method, batch, eta)
    if key not in results:
        results[key] = run(*key)
    return results[key]


batches = [16, 64, 256, 1024, 4096]
with_full_batch = batches + [N]

# ---------------------------------------------------------------------------------------
# 1. A fixed step: the baseline optimizer (Adam, eta = 1e-3) and plain SGD, B from 16 to N
# ---------------------------------------------------------------------------------------
fixed = {"Adam, $\\eta=10^{-3}$": ("Adam", 1e-3), "SGD, $\\eta=0.05$": ("SGD", 0.05), "SGD, $\\eta=0.2$": ("SGD", 0.2)}
print("a fixed step, five epochs:")
for method, eta in fixed.values():
    for batch in with_full_batch:
        result(method, batch, eta)

# ---------------------------------------------------------------------------------------
# 2. SGD with a step proportional to the batch: eta = eta_64 B / 64
# ---------------------------------------------------------------------------------------
scaled = {0.0125: P.BLUE, 0.05: P.AQUA, 0.2: P.ORANGE}              # eta_64, the step at B = 64
print("SGD with eta proportional to B, five epochs:")
for eta_64 in scaled:
    for batch in batches:
        result("SGD", batch, eta_64 * batch / 64)

# ---------------------------------------------------------------------------------------
# 3. How much do two runs of the same configuration differ?  Three seeds of the initial
#    weights and of the shuffling, for the baseline and for SGD with eta = 0.05, B = 64.
# ---------------------------------------------------------------------------------------
print("run-to-run variation (initial weights and shuffling):")
for method, eta in (("Adam", 1e-3), ("SGD", 0.05)):
    accuracies = [result(method, 64, eta)[1]] + [run(method, 64, eta, init_seed=s, shuffle_seed=s + 10)[1]
                                                  for s in (1, 2)]
    print(f"  {method}, B = 64: test accuracies {', '.join(f'{a:.2f}' for a in accuracies)}; "
          f"range {max(accuracies) - min(accuracies):.2f} points")

# ---------------------------------------------------------------------------------------
# The figure: test error against the batch size
# ---------------------------------------------------------------------------------------
fig, (ax_fixed, ax_scaled) = P.panels(2, size=(3.1, 2.2), sharey=True)
for (name, (method, eta)), color, marker in zip(fixed.items(), (P.INK, P.AQUA, P.ORANGE), ("o", "s", "^")):
    ax_fixed.loglog(with_full_batch, [100 - result(method, batch, eta)[1] for batch in with_full_batch],
                    "-" + marker, color=color, ms=3.5, lw=1.1, label=name)
P.label(ax_fixed, "batch size $B$", "test error (%)", title=r"A fixed step $\eta$",
        legend=dict(fontsize=6.3, loc="upper left"))

for eta_64, color in scaled.items():
    etas = [eta_64 * batch / 64 for batch in batches]
    errors = [100 - result("SGD", batch, eta)[1] for batch, eta in zip(batches, etas)]
    ax_scaled.loglog(batches, errors, "-", color=color, lw=1.1, label=rf"$\eta_{{64}}={eta_64}$")
    for batch, eta, error in zip(batches, etas, errors):            # hollow markers: steps of 0.8 and more
        ax_scaled.loglog([batch], [error], "o", color=color, ms=3.8, mfc="white" if eta >= 0.8 else color, mew=1.1)
ax_scaled.plot([], [], "o", color=P.GREY, ms=3.8, mfc="white", mew=1.1, label=r"$\eta\geq0.8$")
P.label(ax_scaled, "batch size $B$", title=r"SGD with $\eta=\eta_{64}\,B/64$",
        legend=dict(fontsize=6.3, loc="upper left", handlelength=1.4, labelspacing=0.3))
for ax in (ax_fixed, ax_scaled):
    ax.set_xticks(batches + ([N] if ax is ax_fixed else []))
    ax.set_xticklabels([str(b) for b in batches] + (["60 000"] if ax is ax_fixed else []), fontsize=7)
    ax.minorticks_off()
    ax.set_yticks([5, 10, 20, 50, 90])
    ax.set_yticklabels(["5", "10", "20", "50", "90"])
    ax.set_ylim(3.5, 100)
C.save(fig, "mnist_batch_size")
