"""Figures for the lecture slides universal_approximation_and_optimization.tex.

Run from the repository root:

    .venv/bin/python lectures/02_universal_approximation_and_optimization/make_figures.py [preview_dir]

The figures reuse the house style of the notebooks (notebooks/plotting.py), so the slides
and the notebooks look alike.  PDFs go to images/ next to this script; if a preview
directory is given, PNG copies are written there as well.

The ten sections reproduce the experiments of notebook 10 with the same code and seeds.
Every number quoted on the slides is printed by this script.  The whole run takes about
half a minute on a laptop CPU; the MNIST files are downloaded once to notebooks/data/, the
folder notebook 10 uses.
"""
import gzip
import pathlib
import sys
import urllib.request
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from torch import nn
import matplotlib
import plotting as P                       # the course palette and figure helpers

matplotlib.use("agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(1)                   # small arrays; one thread keeps the runs reproducible

OUT = pathlib.Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)
PREVIEW = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
trapz = getattr(np, "trapezoid", None) or np.trapz


def expit(t):
    """The sigmoid 1 / (1 + exp(-t)), written so that large |t| does not overflow."""
    return np.exp(-np.logaddexp(0, -t))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


# The target of notebook 10 and of the whole lecture
def target(x):
    return np.sin(3 * x) + 0.3 * x


x = np.linspace(-1, 1, 160)
y = target(x)


# ---------------------------------------------------------------------------------------
# 1. A fixed family can be too small: the least-squares line
# ---------------------------------------------------------------------------------------
A = np.column_stack([np.ones_like(x), x])
intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"least-squares line: {intercept:.3f} + {slope:.3f} x")

fig, ax = P.panels(figsize=(3.3, 2.5))
ax.plot(x, y, "--", color=P.INK, label="target $g$")
ax.plot(x, intercept + slope * x, label="least-squares line")
P.label(ax, "$x$", title="A fixed family can be too small")
save(fig, "linear_fit")


# ---------------------------------------------------------------------------------------
# 2. One target, four approximation families (notebook 10, section 2)
# ---------------------------------------------------------------------------------------
def polynomials(x, m):
    return x[:, None] ** np.arange(m)


def gaussian_translates(x, m, width=0.4):
    centers = np.linspace(-1.5, 1.5, m)
    return np.exp(-0.5 * ((x[:, None] - centers) / width) ** 2)


def hats(x, m):
    nodes = np.linspace(-1, 1, m)
    spacing = nodes[1] - nodes[0]
    return np.maximum(0, 1 - np.abs(x[:, None] - nodes) / spacing)


def sigmoid_neurons(x, m, slope=8):
    centers = np.linspace(-1, 1, m - 1)
    return np.column_stack([np.ones_like(x), expit(slope * (x[:, None] - centers))])


families = {
    "Polynomials": polynomials,
    "Gaussian translates": gaussian_translates,
    "Hat functions": hats,
    "Sigmoid neurons": sigmoid_neurons,
}

fig, axes = P.panels(4, 2, size=(1.85, 1.6), sharex=True)
for (name, features), ax_basis, ax_fit in zip(families.items(), *axes):
    ax_basis.plot(x, features(x, 5), lw=1.2)
    ax_fit.plot(x, y, "--", color=P.INK, lw=1.2, label="target")
    for m in (2, 4, 8):
        coefficients = np.linalg.lstsq(features(x, m), y, rcond=None)[0]
        ax_fit.plot(x, features(x, m) @ coefficients, lw=1.2, label=f"$m={m}$")
    P.label(ax_basis, title=name)
    P.label(ax_fit, "$x$", legend=dict(fontsize=6.5, loc="lower right") if ax_fit is axes[1, 0] else False)
axes[0, 0].set_ylabel(r"building blocks $\varphi_j$")
axes[1, 0].set_ylabel("fitted sum $f$")
save(fig, "families")


# ---------------------------------------------------------------------------------------
# 3. Why sigmoid neurons are dense: steps, bumps and a staircase
# ---------------------------------------------------------------------------------------
def staircase(xs, m, sharpness=6.0):
    """g(s_0) + sum_j (g(s_j) - g(s_{j-1})) sigma(k (x - t_j)): m neurons, one per mesh cell.

    The nodes s_j are equally spaced, t_j is the midpoint of the cell [s_{j-1}, s_j] and
    the slope is k = sharpness / h, so every transition stays inside its own cell.
    """
    nodes = np.linspace(-1, 1, m + 1)
    spacing = nodes[1] - nodes[0]
    mids = (nodes[:-1] + nodes[1:]) / 2
    return target(nodes[0]) + expit(sharpness / spacing * (xs[:, None] - mids)) @ np.diff(target(nodes))


xf = np.linspace(-1, 1, 2001)
fig, (a1, a2, a3) = P.panels(3, size=(2.45, 2.05))
for k in (2, 8, 40):
    a1.plot(xf, expit(k * xf), label=f"$k={k}$")
P.label(a1, "$x$", title=r"$\sigma(kx)$: a steep sigmoid is a step")
for k, color in ((8, P.ORANGE), (40, P.AQUA)):
    a2.plot(xf, expit(k * (xf + 0.3)) - expit(k * (xf - 0.3)), color=color, label=f"$k={k}$")
P.label(a2, "$x$", title="Two steps make a bump")
a3.plot(xf, target(xf), "--", color=P.INK, lw=1.2, label="target")
for m in (8, 32):
    a3.plot(xf, staircase(xf, m), lw=1.2, label=f"$m={m}$")
P.label(a3, "$x$", title="A sum of small steps")
save(fig, "sigmoid_construction")
for m in (8, 32, 128, 512):
    print(f"staircase with m={m:3d} sigmoid neurons: sup error = {np.max(np.abs(staircase(xf, m) - target(xf))):.4f}")


# ---------------------------------------------------------------------------------------
# 4. Gradient descent on a quadratic: four step sizes (notebook 10, section 3)
# ---------------------------------------------------------------------------------------
def quadratic(w):
    return 0.5 * (w[..., 0] ** 2 + 20 * w[..., 1] ** 2)       # F(w) = (w_1^2 + 20 w_2^2) / 2


def grad_quadratic(w):
    return np.array([1.0, 20.0]) * w


def gradient_descent(gradient, start, eta, steps=30):
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w = w - eta * gradient(w)
        path.append(w.copy())
    return np.array(path)


grid = np.linspace(-1.4, 1.4, 250)
W1, W2 = np.meshgrid(grid, grid)
Z = quadratic(np.stack([W1, W2], axis=-1))
start = [1.2, 0.6]
step_sizes = [0.005, 0.05, 0.095, 0.105]
notes = ["creeping", "well scaled", "zig-zag", "divergence"]

fig, axes = P.panels(4, size=(1.85, 1.95), sharex=True, sharey=True)
for ax, eta, note in zip(axes, step_sizes, notes):
    path = gradient_descent(grad_quadratic, start, eta, 8 if eta > 0.1 else 30)
    P.contour_path(ax, W1, W2, Z, path=path, optimum=(0, 0), marker="-o",
                   levels=np.geomspace(0.03, 20, 12), xlabel="$w_1$",
                   xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), title=rf"$\eta={eta}$: {note}")
    for line in ax.lines[:1]:
        line.set_markersize(2.5)
axes[0].set_ylabel("$w_2$")
save(fig, "gd_step_sizes")


# ---------------------------------------------------------------------------------------
# 5. The descent lemma: the quadratic model is an upper bound as soon as 1/eta >= L
# ---------------------------------------------------------------------------------------
# F(w) = w^2 / 2 + sin(2 w) / 2 has F'' = 1 - 2 sin(2 w) in [-1, 3]: non-convex, L-smooth with L = 3.
smooth_f = lambda w: 0.5 * w ** 2 + 0.5 * np.sin(2 * w)
smooth_df = lambda w: w + np.cos(2 * w)
L_smooth, w_k = 3.0, 1.9
model = lambda u, eta: smooth_f(w_k) + smooth_df(w_k) * (u - w_k) + (u - w_k) ** 2 / (2 * eta)
ws = np.linspace(-2.2, 3.1, 600)

fig, ax = P.panels(figsize=(3.4, 2.6))
ax.plot(ws, smooth_f(ws), color=P.INK, label="$F$")
ax.plot(ws, smooth_f(w_k) + smooth_df(w_k) * (ws - w_k), "--", color=P.GREY, lw=1.1, label="linear model")
ax.plot(w_k, smooth_f(w_k), "o", color=P.INK, ms=4.5, zorder=5)
for eta, color, name in ((1 / L_smooth, P.BLUE, r"$\eta=1/L$"), (3.5, P.ORANGE, r"$\eta=10.5/L$")):
    ax.plot(ws, model(ws, eta), color=color, lw=1.3, label=f"model, {name}")
    w_next = w_k - eta * smooth_df(w_k)
    ax.plot([w_next, w_next], [model(w_next, eta), smooth_f(w_next)], ":", color=color, lw=1.1)
    ax.plot(w_next, smooth_f(w_next), "o", color=color, ms=4.5)
    print(f"descent lemma figure, eta={eta:.3f}: F(w_k)={smooth_f(w_k):.3f} -> F(w_k+1)={smooth_f(w_next):.3f}, "
          f"model value {model(w_next, eta):.3f}")
ax.annotate("$w_k$", (w_k, smooth_f(w_k)), xytext=(6, -10), textcoords="offset points", fontsize=9)
P.label(ax, "$w$", xlim=(-2.2, 3.1), ylim=(-1.0, 5.6),                    # the legend sits inside the blue parabola
        legend=dict(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.69, 1.02), handlelength=1.5))
save(fig, "descent_lemma")


# ---------------------------------------------------------------------------------------
# 6. Momentum (notebook 10, section 5)
# ---------------------------------------------------------------------------------------
def heavy_ball(gradient, start, eta, beta=0.6, steps=60):
    w = np.array(start, dtype=float)
    velocity = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        velocity = beta * velocity - eta * gradient(w)
        w = w + velocity
        path.append(w.copy())
    return np.array(path)


momentum_paths = {
    "gradient descent": gradient_descent(grad_quadratic, start, 0.05, steps=60),
    "heavy ball": heavy_ball(grad_quadratic, start, 0.05),
}

fig, (ax_path, ax_loss) = P.panels(2, size=(3.1, 2.2))
ax_path.contour(W1, W2, Z, levels=np.geomspace(0.03, 12, 10), colors=P.RULE, linewidths=0.8)
for name, path in momentum_paths.items():
    ax_path.plot(*path.T, "-o", ms=2, lw=1.2, label=name)
    ax_loss.semilogy(quadratic(path), label=name)
    print(f"momentum figure, {name}: F after 60 steps = {quadratic(path)[-1]:.2e}")
ax_path.plot(0, 0, "*", color=P.INK, ms=10)
P.label(ax_path, "$w_1$", "$w_2$", title="Momentum can overshoot")
P.label(ax_loss, "step $k$", "$F(w_k)$", title="Progress along the valley")
save(fig, "momentum")


# ---------------------------------------------------------------------------------------
# 7. Four optimizers on three problems (notebook 10, section 5.1)
# ---------------------------------------------------------------------------------------
curvatures = torch.logspace(0, 2, 50)
inputs = torch.tensor(x, dtype=torch.float32)[:, None]
targets = torch.tensor(y, dtype=torch.float32)[:, None]


def make_quadratic():
    w = torch.ones(50, requires_grad=True)
    return [w], lambda: 0.5 * (curvatures * w ** 2).sum()


def make_rosenbrock():
    w = torch.tensor([-1.2, 1.0], requires_grad=True)
    return [w], lambda: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2


def make_network():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))


def make_network_problem():
    net = make_network()
    return list(net.parameters()), lambda: ((net(inputs) - targets) ** 2).mean()


methods = {
    "gradient descent": lambda p, lr: torch.optim.SGD(p, lr=lr),
    "heavy ball": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "Nesterov": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True),
    "Adam": lambda p, lr: torch.optim.Adam(p, lr=lr),
}


def optimize(setup, method, lr, steps=300):
    parameters, objective = setup()
    optimizer = methods[method](parameters, lr)
    losses = [objective().item()]
    for _ in range(steps):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
        losses.append(objective().item())
    return losses


# Each row gives: title, problem, GD/momentum learning rate, Adam learning rate.
problems = [
    ("Elongated quadratic", make_quadratic, 0.01, 0.05),
    ("Rosenbrock valley", make_rosenbrock, 0.002, 0.5),
    ("Small network", make_network_problem, 0.05, 0.03),
]

fig, axes = P.panels(3, size=(2.35, 2.0))
for ax, (title, setup, lr, adam_lr) in zip(axes, problems):
    for method in methods:
        ax.semilogy(optimize(setup, method, adam_lr if method == "Adam" else lr), lw=1.3, label=method)
    P.label(ax, "step $k$", title=title,
            legend=dict(fontsize=6.3, loc="upper right", ncols=2, columnspacing=0.8) if ax is axes[0] else False)
axes[0].set_ylabel("loss")
axes[0].set(ylim=(1e-16, 1e16), yticks=[1e-15, 1e-9, 1e-3, 1e3])    # room for the legend above the curves
save(fig, "optimizers")


# ---------------------------------------------------------------------------------------
# 8. A double well: noise leaves the shallow basin (notebook 10, section 6)
# ---------------------------------------------------------------------------------------
def well(w):
    return (w ** 2 - 1) ** 2 + 0.3 * w


def grad_well(w):
    return 4 * w * (w ** 2 - 1) + 0.3


w_grid = np.linspace(-2, 2, 2001)
near_zero = np.abs(w_grid) < 0.5
barrier = w_grid[near_zero][np.argmax(well(w_grid[near_zero]))]     # the local maximum between the wells
w_star = w_grid[np.argmin(well(w_grid))]                            # the global minimizer
w_shallow = w_grid[w_grid > 0.5][np.argmin(well(w_grid[w_grid > 0.5]))]


def langevin_1d(sigma, start, eta=0.05, steps=400, seed=0):
    rng = np.random.default_rng(seed)
    w = start
    path = [w]
    for _ in range(steps):
        w = w - eta * grad_well(w) + sigma * np.sqrt(eta) * rng.standard_normal()
        path.append(w)
    return np.array(path)


sigma, w_0 = 0.7, 1.5                                               # both runs start on the right slope
runs_1d = {
    "gradient descent": langevin_1d(sigma=0.0, start=w_0, eta=0.005),
    "noisy gradient": langevin_1d(sigma=sigma, start=w_0),
}
long_run = langevin_1d(sigma=sigma, start=w_0, eta=0.01, steps=400_000, seed=1)
gibbs = np.exp(-2 * well(w_grid) / sigma ** 2)
gibbs /= trapz(gibbs, w_grid)
print(f"double well: minima at {w_star:.3f} (F={well(w_star):.3f}) and {w_shallow:.3f} (F={well(w_shallow):.3f}), "
      f"barrier at {barrier:.3f} (F={well(barrier):.3f})")
print(f"  barrier seen from the shallow well: {well(barrier) - well(w_shallow):.3f}, "
      f"from the deep well: {well(barrier) - well(w_star):.3f}")
print(f"  first crossing of the barrier by the noisy run: step {int(np.argmax(runs_1d['noisy gradient'] < barrier))}")
print(f"  long run: fraction of time in the deep well {np.mean(long_run < barrier):.3f}, "
      f"Gibbs mass of the deep well {trapz(gibbs[w_grid < barrier], w_grid[w_grid < barrier]):.3f}")

fig, (ax_land, ax_pos, ax_hist) = P.panels(3, size=(2.45, 2.1), ratios=[1, 1.25, 1])
ax_land.plot(w_grid, well(w_grid), color=P.INK, lw=1.2)
ax_land.plot(w_star, well(w_star), "*", color=P.INK, ms=10)
ax_land.plot(w_0, well(w_0), "o", color=P.GREY, ms=5)
ax_land.annotate("start", (w_0, well(w_0)), xytext=(-27, -2), textcoords="offset points", fontsize=8, color=P.NOTE)
ax_pos.axhline(barrier, color=P.INK, ls="--", lw=1)
for (name, path), color in zip(runs_1d.items(), P.CYCLE):
    ax_land.plot(path[-1], well(path[-1]), "o", color=color, ms=6)
    ax_pos.plot(path, color=color, lw=1, label=name)
P.label(ax_land, "$w$", "$F(w)$", xlim=(-2, 2), ylim=(-0.6, 3), title="Landscape")
P.label(ax_pos, "iteration $k$", "$w_k$", ylim=(-1.9, 1.9), title="The iterates",
        legend=dict(fontsize=7, loc="lower left"))
ax_hist.hist(long_run, bins=80, range=(-2, 2), density=True, color=P.GHOST)
ax_hist.plot(w_grid, gibbs, color=P.ORANGE, lw=1.3, label=r"$\propto e^{-2F/\sigma^2}$")
P.label(ax_hist, "$w$", "density", xlim=(-2, 2), title="Long run",
        legend=dict(fontsize=7, loc="upper right"))
save(fig, "double_well")


# ---------------------------------------------------------------------------------------
# 9. Full-batch and mini-batch training of a small network (notebook 10, section 7)
# ---------------------------------------------------------------------------------------
def train(batch_size, epochs=200, lr=0.05):
    net = make_network()                                            # identical initial weights for both runs
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        losses = [((net(inputs) - targets) ** 2).mean().item()]
    for _ in range(epochs):
        order = torch.randperm(len(inputs), generator=generator)
        for batch in order.split(batch_size):
            optimizer.zero_grad()
            loss = ((net(inputs[batch]) - targets[batch]) ** 2).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            losses.append(((net(inputs) - targets) ** 2).mean().item())
    return net, losses


training_runs = {"full batch (160)": train(batch_size=160), "mini-batch (16)": train(batch_size=16)}

fig, (ax_loss, ax_fit) = P.panels(2, size=(3.1, 2.2))
ax_fit.plot(x, y, "--", color=P.INK, label="target")
for name, (net, losses) in training_runs.items():
    ax_loss.semilogy(losses, label=name)
    with torch.no_grad():
        prediction = net(inputs).numpy().ravel()
    ax_fit.plot(x, prediction, label=name)
    print(f"{name}: final training MSE = {losses[-1]:.4g}, max grid error = {np.max(np.abs(prediction - y)):.4g}")
P.label(ax_loss, "epoch", "full training MSE", title="Finding the parameters")
P.label(ax_fit, "$x$", "$f(x)$", title="The learned functions", legend=dict(fontsize=7))
save(fig, "minibatch")


# ---------------------------------------------------------------------------------------
# 10. Everything at once: handwritten digits (notebook 10, section 8)
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


train_images, train_labels = load_mnist("train")
test_images, test_labels = load_mnist("test")


def make_classifier(sizes):
    """Fully connected ReLU network with the given layer sizes, input first."""
    torch.manual_seed(0)
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
    return nn.Sequential(*layers[:-1])                              # the outputs are raw scores


def accuracy(net, images, labels):
    with torch.no_grad():
        return (net(images).argmax(1) == labels).float().mean().item()


def train_mnist(net, epochs=5, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    cross_entropy = nn.CrossEntropyLoss()                           # softmax and -log p_y in one stable call
    generator = torch.Generator().manual_seed(1)
    batch_losses, test_accuracy = [], [accuracy(net, test_images, test_labels)]
    for epoch in range(epochs):
        order = torch.randperm(len(train_images), generator=generator)
        for batch in order.split(batch_size):
            optimizer.zero_grad()
            loss = cross_entropy(net(train_images[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        test_accuracy.append(accuracy(net, test_images, test_labels))
        print(f"MNIST epoch {epoch + 1}: test accuracy = {100 * test_accuracy[-1]:.2f}%")
    return net, np.array(batch_losses), np.array(test_accuracy)


net = make_classifier([784, 16, 16, 10])                            # the 3Blue1Brown network
n_digit_parameters = sum(p.numel() for p in net.parameters())
print(f"MNIST classifier: {n_digit_parameters} parameters, {len(train_images)} training and {len(test_images)} test images")
net, batch_losses, test_accuracy = train_mnist(net)
print(f"MNIST: {len(batch_losses)} mini-batch steps, training accuracy {100 * accuracy(net, train_images, train_labels):.2f}%")

window = 50
fig, (ax_loss, ax_acc) = P.panels(2, size=(3.1, 2.0))
ax_loss.semilogy(batch_losses, color=P.GHOST, lw=0.8)
ax_loss.semilogy(np.arange(len(batch_losses) - window + 1) + window // 2, P.smooth(batch_losses, window))
P.label(ax_loss, "step $k$", "loss", title="Training: mini-batch loss")
ax_acc.plot(100 * test_accuracy, "-o", ms=4)
P.label(ax_acc, "epoch", "accuracy (%)", title="Test accuracy", xticks=range(len(test_accuracy)))
save(fig, "mnist_training")

generator = torch.Generator().manual_seed(4)
sample = torch.randint(len(test_images), (6,), generator=generator)  # six test images drawn at random
with torch.no_grad():
    probabilities = torch.softmax(net(test_images[sample]), dim=1)
predicted = probabilities.argmax(1)

fig, (ax_images, ax_bars) = P.panels(len(sample), 2, size=(1.15, 1.0), sharey="row")
for k, (ax_image, ax_bar, i) in enumerate(zip(ax_images, ax_bars, sample)):
    truth, prediction = test_labels[i].item(), predicted[k].item()
    ax_image.imshow(test_images[i].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
    P.label(ax_image, title=f"predicted {prediction}", off=True)
    ax_image.title.set_color(P.INK if prediction == truth else P.ORANGE)
    colors = [P.BLUE if c == truth else P.ORANGE if c == prediction else P.GHOST for c in range(10)]
    ax_bar.bar(range(10), probabilities[k], color=colors)
    P.label(ax_bar, xlim=(-0.7, 9.7), ylim=(0, 1.05), xticks=range(10), yticks=[0, 1])
    ax_bar.tick_params(axis="x", labelsize=6)
ax_bars[0].set_ylabel("probability")
save(fig, "mnist_predictions")
print("MNIST sample: truth", test_labels[sample].tolist(), "predicted", predicted.tolist())
