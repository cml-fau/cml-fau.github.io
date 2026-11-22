"""Figures of part 4 (optimization) on small problems: sgd_stationary, noise_floor,
signal_propagation.

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/figures/part4_sgd.py [preview_dir]

Section 2 starts from section 11 of the figure script of lecture 02, with the same runs;
section 3 is its section 12, with the same code and seeds.  Every number quoted on the
slides is printed here.  About three minutes.
"""
import common as C
from common import P, np, plt, torch, nn

x, y = C.x, C.y

# ---------------------------------------------------------------------------------------
# The problem of sections 1 and 2: least squares with 8 hat functions on 160 noisy samples
# of the target.  It is strongly convex and its minimum value J* is positive, so the
# stochastic gradient is noisy at the minimizer w*.
# ---------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
Phi = C.hats(x, 8)
y_noisy = y + 0.3 * rng.standard_normal(len(x))
N = len(x)
w_star = np.linalg.lstsq(Phi, y_noisy, rcond=None)[0]
risk = lambda w: np.mean((Phi @ w - y_noisy) ** 2)
J_star = risk(w_star)
hessian = 2 * Phi.T @ Phi / N                                      # J is quadratic: this is its Hessian H
eigenvalues = np.linalg.eigvalsh(hessian)
print(f"least squares: L = {eigenvalues[-1]:.3f}, mu = {eigenvalues[0]:.3f}, J* = {J_star:.4f}")

# The covariance of the gradient of one sample at w*: Sigma = (1/N) sum_i g_i g_i^T, because
# the gradients g_i = 2 phi_i (phi_i . w* - y_i) average to grad J(w*) = 0.
sample_gradients = 2 * Phi * (Phi @ w_star - y_noisy)[:, None]
Sigma = sample_gradients.T @ sample_gradients / N
print(f"gradient noise at w*: tr Sigma = {np.trace(Sigma):.4f}")
# The bound of the theorem with sigma^2 = tr(Sigma) / B, divided by the Ornstein-Uhlenbeck value
print(f"(eta L sigma^2 / 2 mu) / (eta tr(Sigma) / 4B) = 2 L / mu = {2 * eigenvalues[-1] / eigenvalues[0]:.2f}")


# ---------------------------------------------------------------------------------------
# 1. The stationary regime of SGD against the Ornstein-Uhlenbeck prediction
# ---------------------------------------------------------------------------------------
def lyapunov(H, D):
    """The solution S of H S + S H = D for a symmetric positive definite H."""
    lam, V = np.linalg.eigh(H)
    return V @ ((V.T @ D @ V) / (lam[:, None] + lam[None, :])) @ V.T


def discrete_gap(eta, batch):
    """E J - J* at stationarity for SGD itself, if the noise covariance were the constant Sigma:
    S = (I - eta H) S (I - eta H) + eta^2 Sigma / B, solved in the eigenbasis of H."""
    lam, V = np.linalg.eigh(hessian)
    decay = 1 - eta * lam
    S = (eta ** 2 / batch) * (V.T @ Sigma @ V) / (1 - decay[:, None] * decay[None, :])
    return 0.5 * np.sum(lam * np.diag(S))


def stationary_sgd(eta, batch, chains=400, burn_time=40, average_time=100, seed=0):
    """SGD from w*, with indices drawn with replacement, for `chains` independent runs.

    Time is t = k eta.  After `burn_time` the runs are stationary (the slowest mode relaxes
    like exp(-2 mu t) with mu = 0.075).  Returns the mean of J(w_k) - J* over the runs and
    the next `average_time`, and samples of w_k - w* taken once per unit of time.
    """
    r = np.random.default_rng(seed)
    w = np.tile(w_star, (chains, 1))
    burn, average = int(burn_time / eta), int(average_time / eta)
    every = max(1, int(1 / eta))
    gaps, samples = [], []
    for k in range(burn + average):
        i = r.integers(0, N, (chains, batch))
        residual = np.einsum("cbj,cj->cb", Phi[i], w) - y_noisy[i]
        w = w - eta * 2 * np.einsum("cbj,cb->cj", Phi[i], residual) / batch
        if k >= burn:
            deviation = w - w_star
            gaps.append(0.5 * np.einsum("cj,jk,ck->c", deviation, hessian, deviation).mean())
            if (k - burn) % every == 0:
                samples.append(deviation.copy())
    return np.mean(gaps), np.concatenate(samples)


etas, batches = (0.8, 0.2, 0.05, 0.0125), (1, 4, 16, 64)
measured = {}
for eta in etas:
    for batch in batches:
        gap, samples = stationary_sgd(eta, batch)
        measured[eta, batch] = (gap, samples)
        predicted = eta / (4 * batch) * np.trace(Sigma)
        S = lyapunov(hessian, eta / batch * Sigma)
        print(f"  eta = {eta:<6} B = {batch:<2} eta/B = {eta / batch:.2e}: gap = {gap:.3e}, "
              f"eta tr(Sigma) / 4B = {predicted:.3e}, ratio = {gap / predicted:.3f} "
              f"(constant-Sigma SGD: {discrete_gap(eta, batch) / predicted:.3f}); "
              f"std of w_4: {samples[:, 3].std():.4f}, predicted {S[3, 3] ** 0.5:.4f}")

fig, (ax_hist, ax_gap) = P.panels(2, size=(2.7, 1.95))
coordinate = 3                                                      # the fourth coefficient, w_4
pairs = [((0.0125, 1), P.BLUE), ((0.05, 4), P.AQUA), ((0.2, 16), P.YELLOW), ((0.05, 1), P.ORANGE)]
grid = np.linspace(-0.3, 0.3, 400)
for (eta, batch), color in pairs:
    ax_hist.hist(measured[eta, batch][1][:, coordinate], bins=60, range=(-0.3, 0.3), density=True,
                 histtype="step", color=color, lw=1.1, label=f"({eta}, {batch})")
for k, ratio in enumerate((0.0125, 0.05)):
    variance = lyapunov(hessian, ratio * Sigma)[coordinate, coordinate]
    ax_hist.plot(grid, np.exp(-grid ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance), "--", color=P.INK,
                 lw=0.9, label="predicted" if k == 0 else None)
P.label(ax_hist, r"$w_4-w_4^\star$", "density", title=r"Same $\eta/B$, same distribution",
        xlim=(-0.3, 0.3), ylim=(0, 13), xticks=[-0.2, 0, 0.2], yticks=[0, 5, 10],
        legend=dict(fontsize=5.8, loc="upper left", handlelength=1.2, labelspacing=0.25, borderpad=0.3,
                    title=r"$(\eta,\,B)$", title_fontsize=6.2))

ratios = np.array([1e-4, 1.3])
ax_gap.loglog(ratios, ratios * np.trace(Sigma) / 4, "--", color=P.INK, lw=0.9,
              label=r"$\frac{\eta}{4B}\,\mathrm{tr}\,\Sigma(w^\star)$")
for batch, color, marker in zip(batches, (P.BLUE, P.AQUA, P.YELLOW, P.ORANGE), ("o", "s", "^", "D")):
    ax_gap.loglog([eta / batch for eta in etas], [measured[eta, batch][0] for eta in etas], marker, color=color,
                  ms=4, mfc="none", mew=1.1, label=f"$B={batch}$")
P.label(ax_gap, r"$\eta/B$", r"$\mathbb{E}\,J(w_k)-J^\star$", title="The floor is linear in $\\eta/B$",
        xlim=(1e-4, 1.3), ylim=(5e-6, 0.6), xticks=[1e-4, 1e-3, 1e-2, 1e-1, 1], yticks=[1e-5, 1e-3, 1e-1],
        legend=dict(fontsize=5.8, loc="upper left", ncols=2, columnspacing=0.6, handlelength=1.4, labelspacing=0.25,
                    borderpad=0.3))
C.save(fig, "sgd_stationary")


# ---------------------------------------------------------------------------------------
# 2. The noise floor of SGD with a constant step: the step size sets the speed (the time is
#    t = k eta), the ratio eta/B sets the floor.  Left, the same eta with B = 1 and 16; right,
#    the same eta/B = 1/32 with eta = 0.5 and 0.5/16.
# ---------------------------------------------------------------------------------------
def sgd_gap(eta, batch, steps=20_000, seed=0):
    """J(w_k) - J* along SGD from w_0 = 0, for k = 0, ..., steps - 1."""
    r = np.random.default_rng(seed)
    w, gaps = np.zeros(Phi.shape[1]), np.empty(steps)
    for k in range(steps):
        gaps[k] = risk(w) - J_star
        i = r.integers(0, N, batch)
        gradient = 2 * Phi[i].T @ (Phi[i] @ w - y_noisy[i]) / batch
        w = w - eta * gradient
    return gaps


def expected_gap(eta, batch, steps=20_000):
    """E J(w_k) - J* = tr(H M_k) / 2 along SGD from w_0 = 0, exactly.

    With e = w - w*, the gradient of sample i is A_i e + g_i, with A_i = 2 phi_i phi_i^T and the
    g_i of Sigma; it is affine in e, so the mean m_k and the second moment M_k of e_k obey
        m' = (I - eta H) m,
        M' = M - eta (H M + M H) + eta^2 [Q / B + (1 - 1/B) H M H],
        Q  = (1/N) sum_i (A_i M A_i + A_i m g_i^T + g_i m^T A_i + g_i g_i^T)
    for B indices drawn with replacement.
    """
    m = -w_star.copy()
    M, gaps = np.outer(m, m), np.empty(steps)
    for k in range(steps):
        gaps[k] = 0.5 * np.trace(hessian @ M)
        q, p = np.einsum("ij,jk,ik->i", Phi, M, Phi), Phi @ m                # phi_i^T M phi_i, phi_i^T m
        cross = 2 * (Phi * p[:, None]).T @ sample_gradients / N
        Q = 4 * (Phi * q[:, None]).T @ Phi / N + cross + cross.T + Sigma
        M = M - eta * (hessian @ M + M @ hessian) + eta ** 2 * (Q / batch + (1 - 1 / batch) * hessian @ M @ hessian)
        m = m - eta * hessian @ m
    return gaps


sgd_runs = {                                                       # (eta, B): legend, colour
    (0.5, 1): (r"$\eta=0.5$, $B=1$", P.BLUE),
    (0.5, 16): (r"$\eta=0.5$, $B=16$", P.ORANGE),
    (0.5 / 16, 1): (r"$\eta=0.5/16$, $B=1$", P.AQUA),
}
gaps, expected = {}, {}
for (eta, batch) in sgd_runs:
    gaps[eta, batch] = np.array([sgd_gap(eta, batch, seed=s) for s in range(20)])  # 20 independent runs
    expected[eta, batch] = expected_gap(eta, batch)
    tail = gaps[eta, batch][:, -5000:]
    print(f"  SGD eta = {eta:<7} B = {batch:<2}: over the last 5000 steps, mean gap of 20 runs = {tail.mean():.2e}, "
          f"exact E = {expected[eta, batch][-5000:].mean():.2e}, std along one run = {tail.std(axis=1).mean():.2e}; "
          f"eta tr(Sigma) / 4B = {eta / (4 * batch) * np.trace(Sigma):.2e}")

fig, axes = P.panels(2, size=(2.6, 2.3), sharey=True)
steps_axis = np.arange(1, 20_000)
comparisons = (((0.5, 1), (0.5, 16)), ((0.5, 16), (0.5 / 16, 1)))
titles = (r"Same $\eta$: larger $B$, lower floor", r"Same $\eta/B$: same floor")
for ax, pair, title in zip(axes, comparisons, titles):
    for key in pair:                                               # one run, thin: its fluctuation
        ax.loglog(steps_axis, gaps[key][0, 1:], color=sgd_runs[key][1], lw=0.35, alpha=0.5)
    for key in pair:                                               # the exact expectation on top
        ax.loglog(steps_axis, expected[key][1:], color=sgd_runs[key][1], lw=1.4, label=sgd_runs[key][0])
    P.label(ax, "step $k$", r"$J(w_k)-J^\star$" if ax is axes[0] else None, title=title,
            xlim=(1, 2e4), ylim=(1e-4, 3), legend=dict(fontsize=7, loc="upper right"))
C.save(fig, "noise_floor")


# ---------------------------------------------------------------------------------------
# 3. Depth as a stability problem: forward signal and backward gradient
#    (lecture 02, section 12 of its script; the figure of the appendix frame)
# ---------------------------------------------------------------------------------------
torch.manual_seed(0)
width, depth = 512, 50
x_in = torch.randn(width)
fig, (ax_forward, ax_backward) = P.panels(2, size=(3.1, 2.25))
layers_axis = np.arange(depth + 1)
for variance, color in ((1.6, P.BLUE), (2.0, P.INK), (2.4, P.ORANGE)):
    state, norms = x_in.clone(), [x_in.pow(2).sum().item()]
    for _ in range(depth):
        state = torch.relu(torch.randn(width, width) * (variance / width) ** 0.5 @ state)
        norms.append(state.pow(2).sum().item())
    norms = np.array(norms) / norms[0]
    ax_forward.semilogy(layers_axis, norms, color=color, label=rf"$\sigma_w^2={variance}$")
    ax_forward.semilogy(layers_axis, (variance / 2) ** layers_axis, ":", color=color, lw=1)
    print(f"signal propagation, sigma_w^2 = {variance}: |x_50|^2/|x_0|^2 = {norms[-1]:.3e}, "
          f"predicted {(variance / 2) ** depth:.3e}")
P.label(ax_forward, "layer $k$", r"$\|x_k\|^2/\|x_0\|^2$", title="Forward: 50 ReLU layers", ylim=(4e-6, 5e7),
        yticks=[1e-4, 1e-2, 1, 1e2, 1e4], legend=dict(fontsize=6.5, loc="upper center", ncols=3, columnspacing=0.8,
                                                      handlelength=1.2))


def adjoint_norms(depth, residual, width=64, seed=0):
    """|p_k| = |d loss / d x_k| for k = 0..depth, with PyTorch's default initialization."""
    torch.manual_seed(seed)
    layers = [nn.Linear(width, width) for _ in range(depth)]
    states = [torch.randn(16, width).requires_grad_(True)]
    for layer in layers:
        increment = torch.tanh(layer(states[-1]))
        states.append(states[-1] + increment / depth if residual else increment)
        states[-1].retain_grad()
    states[-1].pow(2).mean().backward()
    return np.array([s.grad.norm().item() for s in states])


for (name, residual), color in zip((("plain", False), ("residual, $h=1/L$", True)), (P.BLUE, P.ORANGE)):
    p = adjoint_norms(32, residual)
    ax_backward.semilogy(np.arange(33), p / p[-1], color=color, label=name)
    print(f"gradient through 32 tanh layers ({'residual' if residual else 'plain'}): |p_0|/|p_L| = {p[0] / p[-1]:.2e}")
P.label(ax_backward, "layer $k$", r"$\|p_k\|/\|p_L\|$", title="Backward: 32 tanh layers",
        legend=dict(fontsize=7, loc="lower right"))
C.save(fig, "signal_propagation")
