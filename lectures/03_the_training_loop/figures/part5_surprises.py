"""Figures of part 5 of the_training_loop.tex that use the target g of lecture 02.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part5_surprises.py [preview_dir]

1. overfitting_classical: least squares with m Legendre features on 20 noisy samples of g,
   for m below the number of samples: the classical U-shaped test error.
2. double_descent: the same experiment continued past m = N with the minimal-norm interpolant.
3. edge_of_stability: full-batch gradient descent on a small tanh network; the sharpness
   rises to 2 / eta.
4. spectral_bias: the smooth part of a target is learned before its ripple.

Figures 2 to 4 were made for the second part of lecture 02 and moved here with their seeds
unchanged; figure 4 now measures the error on all of sin(3x) + 0.3x.  Every number quoted on
the slides is printed below.  The run takes about four minutes; nearly all of it is the exact
Hessians of figure 3.
"""
import common as C
from common import P, np, plt, torch, nn
from numpy.polynomial import legendre
from torch.func import functional_call, hessian

x, target = C.x, C.target                              # 160 equispaced points of [-1, 1] and g
xf = np.linspace(-1, 1, 2001)                          # the fine grid on which test errors are measured
inputs = torch.tensor(x, dtype=torch.float32)[:, None]


# ---------------------------------------------------------------------------------------
# 1 and 2. Overfitting and double descent with the polynomials of lecture 02
# ---------------------------------------------------------------------------------------
# Legendre features P_0..P_{m-1}, N = 20 equispaced samples with noise; lstsq returns the
# least-squares fit for m <= N and the interpolant of minimal coefficient norm for m > N.
rng = np.random.default_rng(0)
n_samples, noise, trials = 20, 0.2, 300
x_train = np.linspace(-1, 1, n_samples)
features_dd = lambda x, m: legendre.legvander(x, m - 1)
sizes_dd = np.array(list(range(1, 31)) + [32, 34, 37, 40, 45, 50, 60, 70, 85, 100, 130, 160, 200, 250, 300])
test_error = np.zeros((trials, len(sizes_dd)))
train_error = np.zeros_like(test_error)
fresh_error = np.zeros_like(test_error)                # same inputs, labels with fresh noise (in expectation)
for trial in range(trials):
    y_train = target(x_train) + noise * rng.standard_normal(n_samples)
    for j, m in enumerate(sizes_dd):
        coefficients = np.linalg.lstsq(features_dd(x_train, m), y_train, rcond=None)[0]
        fitted = features_dd(x_train, m) @ coefficients
        test_error[trial, j] = np.mean((features_dd(xf, m) @ coefficients - target(xf)) ** 2)
        train_error[trial, j] = np.mean((fitted - y_train) ** 2)
        fresh_error[trial, j] = np.mean((fitted - target(x_train)) ** 2) + noise ** 2
median = np.median(test_error, axis=0)
below = sizes_dd < n_samples
print("double descent, median test MSE: " + ", ".join(
    f"m={m}: {median[list(sizes_dd).index(m)]:.3g}" for m in (2, 4, 8, 16, 20, 24, 30, 40, 100, 300)))
print(f"  best below the threshold: m={sizes_dd[below][np.argmin(median[below])]} ({median[below].min():.4f}); "
      f"best above: m={sizes_dd[~below][np.argmin(median[~below])]} ({median[~below].min():.4f}); "
      f"peak at m={sizes_dd[np.argmax(median)]} ({median.max():.3g})")

# The optimism of the training error (the proposition of the opening frame): with H the
# projection onto the span of the features at the samples and b_m^2 = |(I - H) g|^2 / N,
#   E[training error] = b_m^2 + s^2 (1 - m/N),   E[error on fresh labels] = b_m^2 + s^2 (1 + m/N).
print(f"optimism of the training error, N = {n_samples}, s = {noise} (s^2 = {noise ** 2:.3f}), mean of {trials} draws:")
for m in (2, 4, 8, 16):
    j = list(sizes_dd).index(m)
    Q = np.linalg.qr(features_dd(x_train, m))[0]
    g_train = target(x_train)
    bias2 = np.mean((g_train - Q @ (Q.T @ g_train)) ** 2)
    print(f"  m={m:2d}: b_m^2 = {bias2:.5f};  training error {train_error[:, j].mean():.4f} "
          f"(formula {bias2 + noise ** 2 * (1 - m / n_samples):.4f});  fresh labels {fresh_error[:, j].mean():.4f} "
          f"(formula {bias2 + noise ** 2 * (1 + m / n_samples):.4f});  gap 2 s^2 m/N = {2 * noise ** 2 * m / n_samples:.4f}")

# Figure 1: the classical regime m < N only.  Here the test error is the risk itself, the
# expected squared error on a fresh noisy label: the distance to g on the fine grid plus s^2.
# (Figure 2 keeps the distance to g alone, as in lecture 02: the two differ by s^2 = 0.04.)
classical = sizes_dd <= 16
risk = test_error + noise ** 2
risk_median = np.median(risk, axis=0)
fig, ax = P.panels(figsize=(2.7, 2.05))
ax.fill_between(sizes_dd[classical], *np.quantile(risk, [0.25, 0.75], axis=0)[:, classical],
                color=P.BLUE, alpha=0.2, lw=0)
ax.semilogy(sizes_dd[classical], risk_median[classical], "-o", ms=2.5, color=P.BLUE, label=r"test error $\mathcal{R}$")
ax.semilogy(sizes_dd[classical], train_error.mean(axis=0)[classical], "-o", ms=2.5, color=P.GREY,
            label="training error $J$")
ax.axhline(noise ** 2, color=P.INK, ls=":", lw=0.9)
ax.annotate("$s^2$", (16.75, noise ** 2), fontsize=8, color=P.NOTE, ha="left", va="center", annotation_clip=False)
best = sizes_dd[below][np.argmin(median[below])]
ax.axvline(best, color=P.RULE, lw=0.8, zorder=0)
P.label(ax, "number of features $m$", "mean squared error", xlim=(0.4, 16.6), ylim=(5e-3, 2),
        xticks=[1, 4, 8, 12, 16], title="$N=20$ noisy samples of $g$", legend=dict(fontsize=7, loc="upper center"))
C.save(fig, "overfitting_classical")
at = lambda m: list(sizes_dd).index(m)
print(f"overfitting, classical regime (test error = distance to g + s^2): {risk_median[at(1)]:.3g} at m=1, "
      f"{risk_median[at(best)]:.4f} at m={best}, {risk_median[at(8)]:.4f} at m=8, {risk_median[at(12)]:.3g} at m=12, "
      f"{risk_median[at(16)]:.3g} at m=16; mean training error: {train_error.mean(axis=0)[at(best)]:.4f} at m={best}, "
      f"{train_error.mean(axis=0)[at(16)]:.4f} at m=16")

# Figure 2: the whole range of m
fig, (ax_curve, ax_fits) = P.panels(2, size=(3.1, 2.3))
ax_curve.fill_between(sizes_dd, *np.quantile(test_error, [0.25, 0.75], axis=0), color=P.BLUE, alpha=0.2, lw=0)
ax_curve.loglog(sizes_dd, median, color=P.BLUE, label="test error")
ax_curve.loglog(sizes_dd, np.maximum(train_error.mean(axis=0), 1e-6), color=P.GREY, label="training error")
ax_curve.axvline(n_samples, color=P.INK, ls="--", lw=1)
ax_curve.annotate("$m=N$", (n_samples, 2e3), xytext=(4, 0), textcoords="offset points", fontsize=8)
P.label(ax_curve, "number of features $m$", "mean squared error", ylim=(8e-4, 2e4),
        title="Test error: down, up, and down again", legend=dict(fontsize=7, loc="upper left"))

y_train = target(x_train) + noise * np.random.default_rng(3).standard_normal(n_samples)
ax_fits.plot(xf, target(xf), "--", color=P.INK, lw=1, label="target")
ax_fits.plot(x_train, y_train, "o", color=P.INK, ms=2.5)
for m, color in ((4, P.AQUA), (20, P.ORANGE), (100, P.BLUE)):
    coefficients = np.linalg.lstsq(features_dd(x_train, m), y_train, rcond=None)[0]
    ax_fits.plot(xf, features_dd(xf, m) @ coefficients, color=color, lw=1.2, label=f"$m={m}$")
P.label(ax_fits, "$x$", ylim=(-2.2, 2.2), title="Three fits of the same 20 points",
        legend=dict(fontsize=7, loc="lower right", ncols=2))
C.save(fig, "double_descent")


# ---------------------------------------------------------------------------------------
# 3. The edge of stability: the sharpness rises to 2 / eta
# ---------------------------------------------------------------------------------------
# Full-batch gradient descent on the mean squared error of a tanh network 1 -> 24 -> 24 -> 1
# on 40 samples of the target, in double precision.  The sharpness is the largest eigenvalue
# of the Hessian of the training loss, computed exactly every 50 steps.
x_eos = torch.linspace(-1, 1, 40, dtype=torch.float64)[:, None]
y_eos = torch.sin(3 * x_eos) + 0.3 * x_eos


def make_eos_network():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)                          # initialize in double precision
    net = nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))
    torch.set_default_dtype(torch.float32)
    return net


def sharpness(net):
    names = [name for name, _ in net.named_parameters()]
    shapes = [p.shape for p in net.parameters()]
    flat = torch.cat([p.detach().reshape(-1) for p in net.parameters()])

    def loss(vector):
        parameters, i = {}, 0
        for name, shape in zip(names, shapes):
            parameters[name] = vector[i:i + shape.numel()].reshape(shape)
            i += shape.numel()
        return ((functional_call(net, parameters, (x_eos,)) - y_eos) ** 2).mean()

    return torch.linalg.eigvalsh(hessian(loss)(flat))[-1].item()


n_eos_parameters = sum(p.numel() for p in make_eos_network().parameters())
print(f"edge of stability: {n_eos_parameters} parameters, sharpness at initialization {sharpness(make_eos_network()):.2f}")
eos_steps, eos_every = 12_000, 50
fig, (ax_loss, ax_sharp) = P.panels(2, size=(3.1, 2.3))
for eta, color in ((0.2, P.BLUE), (0.1, P.ORANGE), (0.05, P.AQUA)):
    net_eos = make_eos_network()
    losses, sharpnesses = [], []
    for k in range(eos_steps + 1):
        loss = ((net_eos(x_eos) - y_eos) ** 2).mean()
        losses.append(loss.item())
        if k % eos_every == 0:
            sharpnesses.append(sharpness(net_eos))
        net_eos.zero_grad()
        loss.backward()
        with torch.no_grad():
            for parameter in net_eos.parameters():
                parameter -= eta * parameter.grad
    losses, sharpnesses = np.array(losses), np.array(sharpnesses)
    ax_loss.semilogy(losses, color=color, lw=0.8, label=rf"$\eta={eta}$")
    ax_sharp.plot(np.arange(len(sharpnesses)) * eos_every, sharpnesses, color=color, lw=1)
    ax_sharp.axhline(2 / eta, color=color, ls="--", lw=0.9)
    tail = sharpnesses[len(sharpnesses) // 2:]
    increases = np.mean(np.diff(losses) > 0)
    print(f"  eta = {eta}: 2/eta = {2 / eta:.0f}, mean sharpness over the second half = {tail.mean():.2f} "
          f"(max {sharpnesses.max():.2f}), final loss {losses[-1]:.2e}, loss increased in {100 * increases:.0f}% of the steps")
eos_ticks = [0, 4000, 8000, 12_000]
P.label(ax_loss, "step $k$", "training loss", title="The loss still decreases", xticks=eos_ticks,
        legend=dict(fontsize=7, loc="upper right"))
P.label(ax_sharp, "step $k$", r"$\lambda_{\max}(\nabla^2J(w_k))$", ylim=(0, 46), xticks=eos_ticks,
        title=r"The sharpness rises to $2/\eta$ (dashed)")
C.save(fig, "edge_of_stability")


# ---------------------------------------------------------------------------------------
# 4. Spectral bias: the smooth part of the target is learned first
# ---------------------------------------------------------------------------------------
frequency, amplitude = 20.0, 0.3
low_part = torch.sin(3 * inputs) + 0.3 * inputs
high_part = amplitude * torch.sin(frequency * inputs)
wiggly = low_part + high_part
missing = lambda residual, part: ((residual * part).sum() / (part * part).sum()).item()

torch.manual_seed(0)
net_sb = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(net_sb.parameters(), lr=1e-3)
snapshots, record = {}, {}
record_at = set(np.round(np.geomspace(1, 10_000, 120)).astype(int).tolist()) | {100, 1000, 3000, 10_000}
for k in range(10_001):
    prediction = net_sb(inputs)
    loss = ((prediction - wiggly) ** 2).mean()
    if k in (100, 1000, 3000):
        snapshots[k] = prediction.detach().numpy().ravel()
    if k in record_at:
        residual = (wiggly - prediction).detach()
        record[k] = (missing(residual, low_part), missing(residual, high_part))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
for k in (100, 1000, 3000, 10_000):
    print(f"spectral bias, step {k}: missing fraction of sin(3x) + 0.3x = {record[k][0]:+.3f}, of sin(20x) = {record[k][1]:+.3f}")

fig, (ax_fit, ax_modes) = P.panels(2, size=(3.1, 3.0), ratios=[1.25, 1])
ax_fit.plot(x, wiggly.numpy().ravel(), "--", color=P.INK, lw=1, zorder=3, label="target")   # on top of the fits
for (k, values), color in zip(snapshots.items(), [P.AQUA, P.ORANGE, P.BLUE]):
    ax_fit.plot(x, values, color=color, lw=1.1, label=f"step {k}")
P.label(ax_fit, "$x$", title=r"Target $\sin(3x)+0.3x+0.3\sin(20x)$", legend=dict(fontsize=7, loc="upper left"))
record_steps = np.array(sorted(record))
ax_modes.semilogx(record_steps, [record[k][0] for k in record_steps], color=P.MAGENTA, label=r"$\sin(3x)+0.3x$")
ax_modes.semilogx(record_steps, [record[k][1] for k in record_steps], color=P.YELLOW, label=r"$0.3\sin(20x)$")
P.label(ax_modes, "step $k$", "fraction still missing", zero="h", title="Each component of the error",
        ylim=(-0.08, 1.7), yticks=[0, 0.5, 1], legend=dict(fontsize=7, loc="upper center", ncols=2))
C.save(fig, "spectral_bias")
