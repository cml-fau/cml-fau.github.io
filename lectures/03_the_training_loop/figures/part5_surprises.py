"""Figures of part 5 of the_training_loop.tex: three that use the target g of lecture 02, and
the spectral bias.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part5_surprises.py [preview_dir]

1. overfitting_classical: least squares with m Legendre features on 20 noisy samples of g,
   for m below the number of samples: the classical U-shaped test error.
2. double_descent: the same experiment continued past m = N with the minimal-norm interpolant.
3. edge_of_stability: full-batch gradient descent on a small tanh network; the sharpness
   rises to 2 / eta.
4. spectral_bias_gray_scott: a network fed the pixel position (x, y) fits the coarse shape of
   a Gray-Scott pattern first; with Fourier features it fits the whole pattern.  Copied from
   the hybrid-modeling slides.

Figures 2 and 3 were made for the second part of lecture 02 and moved here with their seeds
unchanged.  Every number quoted on the slides is printed below.  The run takes about ten
minutes: four for the exact Hessians of figure 3 and six for the two networks of figure 4
(on 8 threads).
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
# 4. Spectral bias: a network fits the coarse shape of a Gray-Scott pattern first
# ---------------------------------------------------------------------------------------
# Copied from the hybrid-modeling slides: experiments/spectral_bias_gray_scott.py there, with
# the part of experiments/neural_pde_gray_scott.py it imports (the Gray-Scott simulator) pasted
# in.  Same seeds and settings; only the cache (--replot) and the params .tex file are left out.
#
# Target: v(., t = 2000) of the Gray-Scott model (Gray & Scott 1984, Pearson 1993), y = (u, v)
# on a periodic 128 x 128 grid, labyrinth regime, from initial condition B of the hybrid slides:
#     d_t u - Du Lap u = -u v^2 + F (1 - u),    d_t v - Dv Lap v = u v^2 - (F + k) v,
# integrated with explicit Euler (dt = dx = 1, 5-point Laplacian).  Plain supervised
# regression, pixel position -> value of v.
#
# Two networks with the same body (ReLU MLP, 3 hidden layers of width 128), the same loss
# (mean squared error over all 128^2 pixels, full batch) and the same optimizer (Adam):
#   top:    input p = (x, y), scaled to [-1, 1]^2;
#   bottom: input phi(p) = (cos 2 pi B p, sin 2 pi B p), Fourier features (Tancik et al. 2020)
#           with B in Z^(M x 2) drawn from N(0, SIGMA^2) and rounded, so phi is periodic on
#           the grid like the pattern.
# The figure shows both fits at a few training steps next to the target.
from matplotlib.colors import LinearSegmentedColormap

REGIMES = {"labyrinth": (0.037, 0.060), "spots": (0.030, 0.062)}
N, DU, DV = 128, 0.2, 0.1            # grid, diffusion (grid units; dt = dx = 1)
D = torch.tensor([DU, DV]).view(2, 1, 1)

T = 2000                         # time of the Gray-Scott snapshot that is fitted
WIDTH, DEPTH, LR = 128, 3, 1e-3  # same body and optimizer for both inputs
M, SIGMA = 128, 6                # number of frequencies, their scale (cycles per domain)
SHOW = (30, 100, 300, 1000, 3000)  # training steps in the figure


def lap(y):  # periodic 5-point Laplacian
    return y.roll(1, -1) + y.roll(-1, -1) + y.roll(1, -2) + y.roll(-1, -2) - 4 * y


def make_reaction(F, k):
    def f_true(y):
        u, v = y[..., 0, :, :], y[..., 1, :, :]
        uvv = u * v * v
        return torch.stack([-uvv + F * (1 - u), uvv - (F + k) * v], dim=-3)
    return f_true


def step(y, f):
    return y + D * lap(y) + f(y)


def initial_condition(seed, n_seeds=6):
    """u = 1, v = 0 with a few perturbed squares and a little noise."""
    rng = np.random.default_rng(seed)
    u, v = np.ones((N, N)), np.zeros((N, N))
    for _ in range(n_seeds):
        i, j = rng.integers(10, N - 10, 2)
        s = rng.integers(4, 9)
        u[i - s:i + s, j - s:j + s], v[i - s:i + s, j - s:j + s] = 0.5, 0.25
    u += 0.02 * rng.standard_normal((N, N))
    v += 0.02 * rng.standard_normal((N, N))
    return torch.tensor(np.stack([u, v]), dtype=torch.float32)


@torch.no_grad()
def simulate(y, f, T, every):
    out = [y]
    for n in range(1, T + 1):
        y = step(y, f)
        if n % every == 0:
            out.append(y)
    return torch.stack(out)


def mlp(d_in):
    layers, d = [], d_in
    for _ in range(DEPTH):
        layers += [nn.Linear(d, WIDTH), nn.ReLU()]
        d = WIDTH
    return nn.Sequential(*layers, nn.Linear(d, 1))


def fourier_features(p, B):
    angle = 2 * np.pi * p @ B.T
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)


def fit(inputs, target, seed=0):
    """Adam on the full batch; returns {step: fit on the grid} and {step: relative L2 error}."""
    torch.manual_seed(seed)
    net = mlp(inputs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    shots, errors = {}, {}
    for n in range(max(SHOW) + 1):
        out = net(inputs)
        loss = ((out - target) ** 2).mean()
        if n in SHOW:
            shots[n] = out.detach()
            errors[n] = (loss / (target ** 2).mean()).sqrt().item()
            print(f"  step {n:5d}  relative error {errors[n]:.3f}", flush=True)
        if n == max(SHOW):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return shots, errors


def spectral_bias_gray_scott():
    torch.set_num_threads(8)  # as in the hybrid slides: the matmuls here are large enough to profit
    F, k = REGIMES["labyrinth"]
    v = simulate(initial_condition(1), make_reaction(F, k), T, T)[-1, 1]   # (N, N)
    N = v.shape[0]
    mean, std = v.mean(), v.std()
    target = ((v - mean) / std).reshape(-1, 1)                          # normalized for training

    i, j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    p = torch.stack([i, j], dim=-1).reshape(-1, 2).float() / N            # pixel positions in [0, 1)^2
    B = (SIGMA * torch.randn(M, 2, generator=torch.Generator().manual_seed(0))).round()
    inputs = {"xy": 2 * p - 1, "ff": fourier_features(p, B)}

    runs = {}
    for name, x in inputs.items():
        print(f"spectral bias, input {name}: {x.shape[1]} features", flush=True)
        shots, errors = fit(x, target)
        runs[name] = ({n: (s.reshape(N, N) * std + mean).numpy() for n, s in shots.items()}, errors)

    # --- figure: two rows of snapshots, the target on the right ------------------------
    # Drawn in matplotlib's default style, as in the hybrid slides, not in the house style.
    with plt.style.context("default"):
        cmap = LinearSegmentedColormap.from_list("v", ["#ffffff", "#7f9ad1", "#1f3a93", "#EB811B"])
        plt.rcParams.update({"font.size": 7})
        n_col = len(SHOW)
        fig = plt.figure(figsize=(4.4, 1.47))
        grid = fig.add_gridspec(2, n_col + 2, width_ratios=[1] * n_col + [0.1, 2.2], wspace=0.06, hspace=0.06,
                                left=0.06, right=0.995, top=0.9, bottom=0.01)

        def show(ax, img):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=0.45, interpolation="bilinear")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4), s.set_color("0.6")

        for r, name in enumerate(("xy", "ff")):
            for c, n in enumerate(SHOW):
                ax = fig.add_subplot(grid[r, c])
                show(ax, runs[name][0][n])
                if r == 0:
                    ax.set_title(f"step {n}", fontsize=7, pad=2)
                if c == 0:
                    ax.set_ylabel(("$(x,y)$", r"$\varphi(x,y)$")[r], fontsize=7, color=("0.2", "#EB811B")[r])
        ax = fig.add_subplot(grid[:, -1])
        show(ax, v.numpy())
        ax.set_title("target", fontsize=7, pad=2)
        C.save(fig, "spectral_bias_gray_scott")


spectral_bias_gray_scott()
