"""Figure of part 4 (optimization): regularization, an L2 penalty and early stopping on the digits.

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/figures/part4_regularization.py [preview_dir]

A setting that overfits: N = 1000 training images (a random subset of the 60 000) and the wide
network 784 -> 256 -> 256 -> 10, trained with the loop of lecture 02 (Adam, eta = 1e-3,
batches of 64, cross-entropy) for 100 epochs.  Two runs from the same initial weights and the
same shuffling: without regularization, and with the L2 penalty (lambda / 2) |w|^2 on all
parameters added to the loss of every mini-batch, lambda = 3e-3.  With Adam this is not weight
decay: the gradient lambda w of the penalty is averaged and rescaled like the rest of the
gradient, while AdamW shrinks the weights directly (Loshchilov and Hutter, 2019).

regularization: the training loss J(w) (the cross-entropy alone, without the penalty) on the
1000 images and the test loss on the 10 000 test images, after every epoch.  Early stopping
chooses its epoch on a separate validation set, 5000 other images of the training set: the
marked point is the epoch with the smallest validation loss of the run without regularization.
Every number quoted on the slide is printed.  About a minute.
"""
from multiprocessing import Pool

import common as C
from common import P, np, plt, torch, nn

train, test = C.load_mnist("train"), C.load_mnist("test")
order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(0))
N, N_VALIDATION, EPOCHS, LAMBDA = 1000, 5000, 100, 3e-3
subset = train[0][order[:N]], train[1][order[:N]]
validation = train[0][order[N:N + N_VALIDATION]], train[1][order[N:N + N_VALIDATION]]
cross_entropy = nn.CrossEntropyLoss()


def run(penalty):
    """Training, validation and test loss and test accuracy before training and after each epoch."""
    net = C.make_classifier([784, 256, 256, 10], seed=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    history = []

    def evaluate():
        with torch.no_grad():
            scores = net(test[0])
            history.append((cross_entropy(net(subset[0]), subset[1]).item(),
                            cross_entropy(net(validation[0]), validation[1]).item(),
                            cross_entropy(scores, test[1]).item(),
                            (scores.argmax(1) == test[1]).float().mean().item()))

    evaluate()
    for epoch in range(EPOCHS):
        for batch in torch.randperm(N, generator=generator).split(64):
            optimizer.zero_grad()
            loss = cross_entropy(net(subset[0][batch]), subset[1][batch])
            if penalty:
                loss = loss + penalty / 2 * sum((p ** 2).sum() for p in net.parameters())
            loss.backward()
            optimizer.step()
        evaluate()
    return np.array(history).T                     # rows: training, validation, test loss, test accuracy


if __name__ == "__main__":
    with Pool(2) as pool:
        plain, penalized = pool.map(run, [0, LAMBDA])
    stop = int(np.argmin(plain[1]))                # early stopping: the smallest validation loss

    for name, (training, valid, testing, accuracy) in (("no regularization", plain),
                                                        (f"L2 penalty, lambda = {LAMBDA}", penalized)):
        best = int(np.argmin(testing))
        print(f"{name}: after {EPOCHS} epochs J = {training[-1]:.2e}, test loss {testing[-1]:.3f}, "
              f"test accuracy {100 * accuracy[-1]:.1f}%; smallest test loss {testing[best]:.3f} at epoch {best}")
    print(f"early stopping (smallest validation loss {plain[1][stop]:.3f}) at epoch {stop}: J = {plain[0][stop]:.3f}, "
          f"test loss {plain[2][stop]:.3f}, test accuracy {100 * plain[3][stop]:.1f}%")

    epochs = np.arange(EPOCHS + 1)
    fig, axes = P.panels(2, size=(3.1, 2.5), sharey=True)
    for ax, (training, _, testing, _), title in zip(
            axes, (plain, penalized), ("No regularization", rf"L2 penalty, $\lambda=3\cdot10^{{-3}}$")):
        ax.plot(epochs, testing, color=P.BLUE, label="test loss")
        ax.plot(epochs, training, color=P.GREY, label="training loss $J$")
        P.label(ax, "epoch", title=title, xlim=(0, EPOCHS), ylim=(0, 1.0), legend=False)
    ax = axes[0]
    ax.axvline(stop, color=P.RULE, lw=0.8, zorder=0)
    ax.plot(stop, plain[2][stop], "o", color=P.INK, ms=4, zorder=3)
    ax.annotate("early stopping", (stop, plain[2][stop]), xytext=(6, -14), textcoords="offset points",
                fontsize=7, color=P.NOTE)
    axes[0].set_ylabel("cross-entropy")
    axes[1].legend(fontsize=7, loc="upper right")
    C.save(fig, "regularization")
