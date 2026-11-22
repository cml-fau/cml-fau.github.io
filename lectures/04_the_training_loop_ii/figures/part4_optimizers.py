"""Figures of part 4 (optimization): momentum and the optimizers of torch.optim.

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/figures/part4_optimizers.py [preview_dir]

Both figures repeat experiments of notebook 10 with its code and seeds: section 5 (momentum)
and section 5.1 (four optimizers on three problems).  Every number quoted on the slides is
printed here.  A few seconds.
"""
import common as C
from common import P, np, plt, torch, nn

x, y = C.x, C.y


# ---------------------------------------------------------------------------------------
# 1. Gradient descent and the heavy ball on J(w) = (w_1^2 + 20 w_2^2) / 2  (notebook 10, section 5)
# ---------------------------------------------------------------------------------------
def quadratic(w):
    return 0.5 * (w[..., 0] ** 2 + 20 * w[..., 1] ** 2)


def grad_quadratic(w):
    return np.array([1.0, 20.0]) * w


def gradient_descent(gradient, start, eta, steps=30):
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w = w - eta * gradient(w)
        path.append(w.copy())
    return np.array(path)


def heavy_ball(gradient, start, eta, beta=0.6, steps=60):
    w = np.array(start, dtype=float)
    velocity = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        velocity = beta * velocity - eta * gradient(w)
        w = w + velocity
        path.append(w.copy())
    return np.array(path)


grid = np.linspace(-1.4, 1.4, 250)
W1, W2 = np.meshgrid(grid, grid)
Z = quadratic(np.stack([W1, W2], axis=-1))
start = [1.2, 0.6]
momentum_paths = {
    "gradient descent": gradient_descent(grad_quadratic, start, 0.05, steps=60),
    "heavy ball": heavy_ball(grad_quadratic, start, 0.05),
}

fig, (ax_path, ax_loss) = P.panels(2, size=(2.4, 1.85), layout="constrained")
ax_path.contour(W1, W2, Z, levels=np.geomspace(0.03, 12, 10), colors=P.RULE, linewidths=0.8)
for name, path in momentum_paths.items():
    ax_path.plot(*path.T, "-o", ms=2, lw=1.2, label=name)
    ax_loss.semilogy(quadratic(path), label=name)
    print(f"momentum, {name}: J after 60 steps = {quadratic(path)[-1]:.2e}")
ax_path.plot(0, 0, "*", color=P.INK, ms=10)
P.label(ax_path, "$w_1$", "$w_2$", title="Momentum can overshoot", legend=False)
P.label(ax_loss, "step $k$", "$J(w_k)$", title="Progress along the valley", legend=dict(fontsize=7, loc="lower left"))
C.save(fig, "momentum")


# ---------------------------------------------------------------------------------------
# 2. Four optimizers on three problems, 300 full-gradient steps  (notebook 10, section 5.1)
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

fig, axes = P.panels(3, size=(1.8, 3.0), layout="constrained")
for ax, (title, setup, lr, adam_lr) in zip(axes, problems):
    for method in methods:
        losses = optimize(setup, method, adam_lr if method == "Adam" else lr)
        ax.semilogy(losses, lw=1.3, label=method)
        print(f"{title}, {method}: loss after 300 steps = {losses[-1]:.2e}")
    P.label(ax, "step $k$", title=title, legend=False)
axes[0].set_ylabel("$J(w_k)$")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=4, fontsize=8)
C.save(fig, "optimizers")
