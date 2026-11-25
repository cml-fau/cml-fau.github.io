"""Part 3 (the transformer), depth as time: the tokens of an attention-only transformer move like
interacting particles and gather in clusters.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part3_particles.py [preview_dir]

The model is the self-attention dynamics with layer normalization of Geshkovski, Letrouit,
Polyanskiy and Rigollet (Bull. AMS 2025, equation (SA)), on the circle (d = 2), with
Q = K = V = I:

    dx_t/dtau = Pperp_{x_t}( sum_s a_ts(x) x_s ),   a_ts(x) = softmax_s(beta <x_t, x_s>),

where Pperp_x y = y - <x, y> x projects onto the tangent line of the circle at x, and beta takes
the place of the 1/sqrt(d) of the lecture.  One layer is the
residual step of the lecture with V = h I, followed by the normalization x_t / |x_t| (RMSNorm
without its gain): x_t <- x_t + h sum_s a_ts(x) x_s, then x_t <- x_t / |x_t|.  To first order in h
this is an Euler step of the ODE, so layer k is the time tau = k h.

- particles_circle.pdf (the frame "Depth as time: tokens are interacting particles"):
  n = 32 tokens, beta = 16, h = 0.1, 20 000 layers.  The initial angles are uniform in
  (-0.49 pi, 0.49 pi), seed 3 of numpy's default_rng: all tokens in an open half-circle, the
  hypothesis of the "cone collapse" lemma (GLPR, Lemma 6.4), which then gives convergence of all
  tokens to one point, exponentially fast, for every beta > 0.  Left: the tokens on the circle at
  four layers, with the number of tokens in each cluster.  Right: the angle of every token
  against the layer (log scale), each token coloured by the cluster it belongs to at layer 200.
  The tokens form three clusters within about a hundred layers; these merge into two, then,
  thousands of layers later, into one.  Drawn at 108 x 38 mm, the width of the text.
  The parameters were chosen after a scan (beta from 4 to 36, arcs 0.9 pi to 2 pi, seeds 0-5) so
  that both time scales fit in the 20 000 layers drawn: with beta = 16 and this arc, seed 3 is
  the only one of the seeds 0-5 whose last two clusters merge before layer 20 000 (the script
  prints the count at layer 20 000 for each): the long second phase is the rule, not the
  exception.

The script prints every number the slides quote: the parameters, the number of clusters (two
tokens are in the same cluster if their angles differ by less than 1e-3 rad) at the layers
drawn and the layers at which it changes, the angular diameter of the configuration at the end,
and the interaction energy E_beta = (1/(2 beta n^2)) sum_{t,s} exp(beta <x_t, x_s>), which the
theorem's gradient-flow structure says increases along the flow (GLPR, equation (3.6)): the
script checks that it increases at every layer of the discrete scheme too.  About twenty-five
seconds on one thread.
"""
import common as C
from common import P, np, plt, torch

torch.set_num_threads(1)
plt.rcParams["figure.autolayout"] = False                     # the figure places its own axes

# ---------------------------------------------------------------------------------------
# The parameters, fixed before the run
# ---------------------------------------------------------------------------------------
N_TOKENS = 32                 # n: as in the figures of GLPR (n = 32)
BETA = 16.0                   # inverse temperature: a_ts = softmax_s(beta <x_t, x_s>)
H = 0.1                       # step: V = h I, so layer k is the time tau = k h
LAYERS = 20_000               # K: tau runs to 2000
ARC = 0.98 * np.pi            # initial angles uniform in (-ARC/2, ARC/2): an open half-circle
SEED = 3
SNAPSHOTS = (0, 100, 1_000, 20_000)
COLOUR_AT = 200               # colour each token by its cluster at this layer
GAP = 1e-3                    # two tokens are in one cluster if their angles differ by less


def layer(x):
    """One attention layer with Q = K = I, V = h I, then the normalization onto the circle."""
    scores = BETA * x @ x.T                                      # beta <x_t, x_s>
    a = np.exp(scores - scores.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)                            # a_ts = softmax_s, rows sum to 1
    x = x + H * a @ x                                            # x_t + h sum_s a_ts x_s
    return x / np.linalg.norm(x, axis=1, keepdims=True)          # layer normalization


def energy(x):
    """E_beta of the empirical measure, divided by its maximum e^beta / (2 beta) (one point)."""
    return np.exp(BETA * (x @ x.T - 1)).mean()


def clusters(theta):
    """Cluster labels: sort the angles and cut where two neighbours are GAP or more apart."""
    order = np.argsort(theta)
    labels = np.empty(len(theta), int)
    labels[order] = np.concatenate([[0], np.cumsum(np.diff(theta[order]) >= GAP)])
    return labels


# ---------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
theta0 = rng.uniform(-ARC / 2, ARC / 2, N_TOKENS)
x = np.column_stack([np.cos(theta0), np.sin(theta0)])
assert np.all(x[:, 0] > 0)                                       # all tokens in the open half-plane x_1 > 0

angles = np.empty((LAYERS + 1, N_TOKENS))
energies = np.empty(LAYERS + 1)
counts = np.empty(LAYERS + 1, int)
for k in range(LAYERS + 1):
    angles[k] = np.arctan2(x[:, 1], x[:, 0])                     # no wrap-around: the tokens stay in x_1 > 0
    energies[k] = energy(x)
    counts[k] = clusters(angles[k]).max() + 1
    if k < LAYERS:
        x = layer(x)

print(f"n = {N_TOKENS} tokens on the circle (d = 2), Q = K = I, V = h I, beta = {BETA:g}, h = {H:g}, "
      f"{LAYERS} layers (tau = k h up to {LAYERS * H:g}); initial angles uniform in "
      f"(-{ARC / 2 / np.pi:.2f} pi, {ARC / 2 / np.pi:.2f} pi), seed {SEED}")
print(f"initial angles from {np.degrees(angles[0].min()):.1f} to {np.degrees(angles[0].max()):.1f} degrees")
print("clusters (angles closer than 1e-3 rad) at layers " + ", ".join(
    f"k = {k}: {counts[k]}" for k in (0, 10, 50, 100, 200, 1000, 10_000, LAYERS)))
changes = [(k, counts[k]) for k in range(1, LAYERS + 1) if counts[k] != counts[k - 1]]
print("the number of clusters changes at (layer, clusters): "
      + ", ".join(f"({k}, {c})" for k, c in changes if c <= 6))
for k in SNAPSHOTS:
    print(f"layer {k:6d} (tau = {k * H:7.1f}): {counts[k]:2d} clusters, angular diameter "
          f"{np.ptp(angles[k]):.2e} rad, E_beta / max = {energies[k]:.4f}")
steps = np.diff(energies)
print(f"E_beta increases at every layer up to rounding: smallest change {steps.min():.1e} "
      f"(relative to E: {(steps / energies[:-1]).min():.1e}); E_beta / max from {energies[0]:.4f} "
      f"to {energies[-1]:.6f}")

# The same run for the other seeds: how many clusters are left at the last layer.
left = []
for seed in range(6):
    r = np.random.default_rng(seed).uniform(-ARC / 2, ARC / 2, N_TOKENS)
    y = np.column_stack([np.cos(r), np.sin(r)])
    for _ in range(LAYERS):
        y = layer(y)
    left.append(int(clusters(np.arctan2(y[:, 1], y[:, 0])).max() + 1))
print(f"clusters left at layer {LAYERS} for seeds 0-5: {left}")

# ---------------------------------------------------------------------------------------
# The figure: four snapshots on the circle, and the angles against the layer
# ---------------------------------------------------------------------------------------
labels = clusters(angles[COLOUR_AT])
print(f"colours: {labels.max() + 1} clusters at layer {COLOUR_AT}, of sizes "
      f"{np.bincount(labels).tolist()} (from the lowest angle up)")
palette = [P.BLUE, P.ORANGE, P.AQUA, P.MAGENTA, P.YELLOW]
colour = [palette[c % len(palette)] for c in labels]

W, Hgt = 108 * C.MM, 38 * C.MM
fig = plt.figure(figsize=(W, Hgt))
side = 16.5 * C.MM                                              # the side of a snapshot panel
corners = [(0.5, 20.5), (19.5, 20.5), (0.5, 1.0), (19.5, 1.0)]  # lower-left corners, in mm
circle = np.linspace(0, 2 * np.pi, 400)
for (left, bottom), k in zip(corners, SNAPSHOTS):
    ax = fig.add_axes([left * C.MM / W, bottom * C.MM / Hgt, side / W, side / Hgt])
    ax.plot(np.cos(circle), np.sin(circle), color=P.RULE, lw=0.6, zorder=0)
    ax.scatter(np.cos(angles[k]), np.sin(angles[k]), s=7, c=colour, edgecolors="white",
               linewidths=0.3, zorder=2)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    number = f"{k:,}".replace(",", r"\,") if k >= 10_000 else str(k)       # 20\,000, but 1000
    ax.text(-1.18, 1.18, f"$k={number}$", fontsize=7, color=P.INK, ha="left", va="top")
    if k > 0:                                                    # how many tokens sit in each cluster
        cut = clusters(angles[k])
        for c in range(cut.max() + 1):
            a = angles[k][cut == c].mean()
            ax.text(0.66 * np.cos(a), 0.66 * np.sin(a), str((cut == c).sum()), fontsize=6,
                    color=P.NOTE, ha="center", va="center")         # inside the circle, next to its point

ax = fig.add_axes([48 * C.MM / W, 7.5 * C.MM / Hgt, 58 * C.MM / W, 26 * C.MM / Hgt])
layers = np.arange(1, LAYERS + 1)
for t in np.argsort(labels, kind="stable"):
    ax.plot(layers, np.degrees(angles[1:, t]), color=colour[t], lw=0.6, alpha=0.9)
for k in SNAPSHOTS[1:]:
    ax.axvline(k, color=P.RULE, lw=0.5, ls=":", zorder=0)
ax.set_xscale("log")
ax.set_xlim(1, LAYERS)
ax.set_ylim(-92, 92)
ax.set_yticks([-90, -45, 0, 45, 90])
ax.set_yticklabels(["\u221290\u00b0", "", "0\u00b0", "", "90\u00b0"])
ax.set_xticks([1, 10, 100, 1000, 10_000])
ax.set_xticklabels(["1", "10", "100", "1000", "10 000"])
ax.tick_params(labelsize=7, length=2, pad=1.5)
ax.set_xlabel("layer $k$", fontsize=7.5, labelpad=1)
ax.text(1, 99, "angle of $x_t$", fontsize=7.5, color=P.INK, ha="left", va="bottom")   # above the axis
C.save(fig, "particles_circle")
