"""Part 2 (attention), four small exact experiments with one attention layer, no training.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part2_attention.py [preview_dir]

Every figure is drawn at its size on the slide and included unscaled.

- lookup_weights.pdf ("Attention is a differentiable look-up table"): one query q and eight
  keys k_1, ..., k_8, unit vectors in the plane (k_s at angle 45 (s - 1) degrees, q at
  50 degrees); on the left the vectors, on the right the weights softmax_s(beta <q, k_s>) for
  beta = 1, 4, 16 and the hard limit beta -> infinity (all the weight on k_2, the key closest to q).
- scaling_softmax.pdf ("Why the 1/sqrt(d)"): n = 64 keys and 64 queries with independent
  N(0, 1) entries in R^d, d = 4, ..., 1024, 200 draws for each d (the set-up of section 3 of
  notebooks/15_transformers.ipynb); with the scores <q, k> and with <q, k>/sqrt(d): the largest
  weight of a row of the softmax (mean over all rows), and the size (Frobenius norm) of the
  Jacobian of the softmax, diag(a) - a a^T, through which every gradient to Q and K passes
  (median over all rows; its mean and the share of rows below 1e-3 are printed).
- equivariance_x.pdf, equivariance_Px.pdf, equivariance_fx.pdf, equivariance_fPx.pdf
  ("Self-attention is permutation equivariant"): n = 6 tokens in R^8, one per row, and
  f(x) = x + attention(x) for one random multi-head attention layer (nn.MultiheadAttention(8, 2),
  PyTorch's default initialization, seed 0), as in the frame "Six layers, and what they assume"
  of lecture 03.  The four corners of the commutative square: x, Px (the rows permuted),
  f(x), f(Px) = P f(x).  Each row carries the index of its token, so the eye can follow P.
- mask_matrices.pdf ("Masking: no looking ahead"): the six tokens of "The dog bites the man ."
  as random vectors in R^16 and one random attention layer (the tensors of the listing of
  that slide, seed 0): the scores, the mask, and the weights, lower triangular, rows summing to 1.

The script also runs the two listings of these slides (the look-up and the mask) and prints
their output, checks the equivariance (and that it fails with the causal mask or with
positions added), and prints every number the slides quote.  It runs in a few seconds.
"""
import math
import textwrap

import common as C
from common import P, np, plt, torch, nn
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

torch.set_num_threads(2)
plt.rcParams["figure.autolayout"] = False                     # every figure places its own axes

# The colour scales of lecture 04: positive values white to blue, signed values blue to orange.
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9ddf5", P.BLUE, "#0f3a73"])
DIV = LinearSegmentedColormap.from_list("div", ["#0f3a73", P.BLUE, "#ffffff", P.ORANGE, "#8f3410"])


def run_listing(source):
    """Run the code of a slide exactly as it is printed there, and print what it prints."""
    source = textwrap.dedent(source).strip("\n")
    print("  | " + source.replace("\n", "\n  | "))
    exec(source, {"torch": torch, "nn": nn})


def cell_text(ax, i, j, value, text, dark, fontsize=6.5):   # never below 6 pt on the slide
    """A number inside a cell, white on dark cells and ink on light ones."""
    ax.text(j, i, text, ha="center", va="center", fontsize=fontsize,
            color="white" if dark else P.INK)


# =========================================================================================
# 1. The look-up table: weights softmax(beta <q, k_s>) as beta grows
# =========================================================================================
print("\n--- 1. attention as a look-up table")
angles = np.deg2rad(45 * np.arange(8))                       # k_s at 0, 45, ..., 315 degrees
keys = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], 1))
q = torch.tensor([math.cos(math.radians(50)), math.sin(math.radians(50))], dtype=torch.float64)
scores = keys @ q                                             # <q, k_s> = cos(angle between them)
betas = [1, 4, 16]
weights = [torch.softmax(beta * scores, 0) for beta in betas]
weights.append(torch.nn.functional.one_hot(scores.argmax(), 8).double())   # beta -> infinity
print("<q, k_s>, s = 1..8: " + ", ".join(f"{v:.3f}" for v in scores))
for beta, w in zip(betas + ["inf"], weights):
    print(f"beta = {beta:>3}: weights " + ", ".join(f"{v:.3f}" for v in w)
          + f"; largest {w.max():.3f} on k_{w.argmax().item() + 1}")

fig = plt.figure(figsize=(108 * C.MM, 23 * C.MM))
circle = fig.add_axes((0.0, 0.02, 0.22, 0.96))               # the vectors, left
strip = fig.add_axes((0.32, 0.06, 0.66, 0.70))               # the weights, right
for s, a in enumerate(angles):
    circle.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=P.BLUE, lw=0.8, mutation_scale=5))
    circle.text(1.27 * np.cos(a), 1.27 * np.sin(a), f"$k_{s + 1}$", ha="center", va="center",
                fontsize=7, color=P.BLUE)
qa = math.radians(50)
circle.annotate("", xy=(0.82 * np.cos(qa), 0.82 * np.sin(qa)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=P.ORANGE, lw=1.1, mutation_scale=6))
circle.text(0.2, 0.64, "$q$", ha="center", va="center",
            fontsize=8, color=P.ORANGE)
circle.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), aspect="equal")
circle.axis("off")

W = torch.stack(weights).numpy()
strip.imshow(W, cmap=SEQ, norm=Normalize(0, 1), aspect="auto")
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        text = "0" if W[i, j] == 0 else "1" if W[i, j] == 1 else f"{W[i, j]:.2f}"   # exact 0 and 1 only
        cell_text(strip, i, j, W[i, j], text, W[i, j] > 0.55)
strip.set_xticks(range(8), [f"$k_{s + 1}$" for s in range(8)], fontsize=7)
strip.xaxis.tick_top()
strip.set_yticks(range(4), [r"$\beta=1$", r"$\beta=4$", r"$\beta=16$", r"$\beta\to\infty$"], fontsize=7)
strip.tick_params(length=0, pad=2, colors=P.INK)
for spine in strip.spines.values():
    spine.set_visible(False)
strip.axhline(2.5, color="white", lw=2)                       # the hard limit, set apart
C.save(fig, "lookup_weights")

print("listing (slide 'Attention is a differentiable look-up table'; the slide leaves out the prints):")
run_listing('''
    table = {"a": 1.0, "b": 2.0, "c": 3.0}
    print(table["b"])                                  # look-up: an exact match
    keys = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])     # a, b, c
    values = torch.tensor([1.0, 2.0, 3.0])
    q = torch.tensor([0.6, 0.8], requires_grad=True)   # the query: key b
    w = (16 * keys @ q).softmax(0); out = w @ values   # attention, beta = 16
    out.backward(); print(w.data, out.data, q.grad)
''')
# Not on the slide: the same three keys for growing beta.  The output is not monotone in beta
# (key c is closer to the query than key a), but the weights tend to (0, 1, 0).
keys, values, q = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.6, 0.8])
for beta in [1, 4, 16]:
    w = (beta * keys @ q).softmax(0)
    print(f"  beta = {beta}: weights {[round(v, 4) for v in w.tolist()]}, output {(w @ values).item():.4f}")


# =========================================================================================
# 2. Why the 1/sqrt(d): the softmax of <q, k> saturates as d grows
# =========================================================================================
print("\n--- 2. why the 1/sqrt(d)")
n, draws = 64, 200
dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
gen = torch.Generator().manual_seed(0)
stats = {scaled: {"max": [], "entropy": [], "jacobian": []} for scaled in (False, True)}
eye = torch.eye(n, dtype=torch.float64)
for d in dims:
    raw, collect = [], {scaled: {"max": [], "entropy": [], "jacobian": []} for scaled in (False, True)}
    for _ in range(draws):
        Q, K = torch.randn(n, d, generator=gen, dtype=torch.float64), torch.randn(n, d, generator=gen, dtype=torch.float64)
        S = Q @ K.T                                           # n x n scores <q_t, k_s>
        raw.append(S.flatten())
        for scaled in (False, True):
            A = torch.softmax(S / math.sqrt(d) if scaled else S, dim=1)
            # diag(a) - a a^T: off the diagonal -a_s a_r, on it a_s (1 - a_s) with 1 - a_s written as
            # the sum of the other weights, so that it stays exact when a_s is close to 1
            J = -A[:, :, None] * A[:, None, :] * (1 - eye)
            J = J + torch.diag_embed(A * (A[:, None, :] * (1 - eye)).sum(-1))
            collect[scaled]["max"].append(A.max(1).values)
            collect[scaled]["entropy"].append(-(A * A.clamp_min(1e-300).log()).sum(1))
            collect[scaled]["jacobian"].append(torch.linalg.matrix_norm(J))     # Frobenius, per row
    raw = torch.cat(raw)
    for scaled in (False, True):
        for key in ("max", "entropy"):
            stats[scaled][key].append(torch.cat(collect[scaled][key]).mean().item())
        # The Jacobian of a typical row: the median.  Its mean is carried by the few rows whose two
        # largest scores nearly tie, and the share of saturated rows says the rest; both printed.
        jac = torch.cat(collect[scaled]["jacobian"])
        stats[scaled]["jacobian"].append(jac.median().item())
        stats[scaled].setdefault("jacobian mean", []).append(jac.mean().item())
        stats[scaled].setdefault("saturated", []).append((jac < 1e-3).double().mean().item())
    print(f"d = {d:4d}: Var <q,k> = {raw.var():7.1f};  "
          f"largest weight {stats[False]['max'][-1]:.3f} without, {stats[True]['max'][-1]:.3f} with 1/sqrt(d);  "
          f"entropy {stats[False]['entropy'][-1]:.3f} / {stats[True]['entropy'][-1]:.3f};  "
          f"|diag(a) - a a^T| median {stats[False]['jacobian'][-1]:.1e} / {stats[True]['jacobian'][-1]:.1e}, "
          f"mean {stats[False]['jacobian mean'][-1]:.1e} / {stats[True]['jacobian mean'][-1]:.1e}, "
          f"rows below 1e-3 {100 * stats[False]['saturated'][-1]:.1f}% / {100 * stats[True]['saturated'][-1]:.1f}%")
uniform = math.sqrt(1 / n - 1 / n ** 2)                       # |diag(a) - a a^T| for a uniform row
print(f"log n = {math.log(n):.3f} (the entropy of a uniform row); for a uniform row (a_s = 1/n): "
      f"largest weight 1/n = {1 / n:.4f}, |diag(a) - a a^T| = {uniform:.3f}; for a one-hot row: 1 and 0")

fig, axes = plt.subplots(1, 2, figsize=(108 * C.MM, 41 * C.MM))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.22, top=0.96, wspace=0.42)
series = {False: (r"$\langle q,k\rangle$", P.BLUE), True: (r"$\langle q,k\rangle/\sqrt{d}$", P.ORANGE)}
for scaled, (name, color) in series.items():
    axes[0].plot(dims, stats[scaled]["max"], "o-", color=color, ms=2.5, lw=1.2, label=name)
    axes[1].plot(dims, stats[scaled]["jacobian"], "o-", color=color, ms=2.5, lw=1.2, label=name)
axes[0].set_ylim(0, 1.02)
axes[0].set_ylabel("largest weight in a row", fontsize=7.5)
axes[1].set_yscale("log")
axes[1].set_ylabel(r"$\|\mathrm{diag}(a)-aa^\top\|$", fontsize=7.5)
axes[1].text(0.04, 0.06, f"median of {n * draws:,} rows".replace(",", " "), transform=axes[1].transAxes,
             fontsize=6.5, color=P.NOTE)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64, 256, 1024], ["4", "16", "64", "256", "1024"])
    ax.minorticks_off()
    ax.set_xlabel("$d$", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
axes[0].legend(fontsize=7, frameon=False, loc="center right", handlelength=1.2, borderaxespad=0.2)
C.save(fig, "scaling_softmax")


# =========================================================================================
# 3. Permutation equivariance: f(Px) = P f(x) for f(x) = x + attention(x)
# =========================================================================================
print("\n--- 3. permutation equivariance")
torch.manual_seed(0)
n, d = 6, 8
x = torch.randn(n, d)                                         # 6 tokens in R^8, one per row
attn = nn.MultiheadAttention(d, 2)                            # 2 heads, PyTorch's default initialization
p = torch.randperm(n)                                         # (Px)_t = x_p(t)
causal = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)    # True: position t may not see s > t


def f(x, mask=None):
    with torch.no_grad():
        return x + attn(x, x, x, attn_mask=mask, need_weights=False)[0]


def weights(x):
    with torch.no_grad():
        return attn(x, x, x, need_weights=True)[1]            # a_ts, averaged over the heads


fx, fPx = f(x), f(x[p])
print(f"P: token in row t of Px = token {(p + 1).tolist()} of x (1-based)")
print(f"max |f(Px) - P f(x)|                         = {(fPx - fx[p]).abs().max():.1e}")
print(f"max |a(Px) - P a(x) P^T|                     = {(weights(x[p]) - weights(x)[p][:, p]).abs().max():.1e}")
print(f"with the causal mask:   max |f(Px) - P f(x)| = {(f(x[p], causal) - f(x, causal)[p]).abs().max():.3f}")
# Positions as in Vaswani et al. (2017): p_t = (sin(t w_k), cos(t w_k)), w_k = 10000^(-2k/d).
t, k = torch.arange(n)[:, None], torch.arange(0, d, 2)[None, :]
positions = torch.zeros(n, d)
positions[:, 0::2], positions[:, 1::2] = torch.sin(t / 10000 ** (k / d)), torch.cos(t / 10000 ** (k / d))
# The token at row t gets position t.  Compare the updates attention(x + p) only: f(x + p) itself
# also contains x + p, and p - Pp alone would already break the identity.
g = lambda x: f(x + positions) - (x + positions)
print(f"with positions x_t + p_t: max |g(Px) - P g(x)| = {(g(x[p]) - g(x)[p]).abs().max():.3f} "
      f"for the update g(x) = attention(x + p); max |g(x)| = {g(x).abs().max():.3f}")
print(f"for scale: max |x| = {x.abs().max():.3f}, max |f(x)| = {fx.abs().max():.3f}, "
      f"max |f(x) - x| = {(fx - x).abs().max():.3f}")

SIDE = 2.2                                                    # the side of a cell on the slide, in mm
LABEL = 2.6                                                   # the column of token indices, in mm


def token_panel(values, rows, name, norm):
    """n tokens as rows of a heatmap, each with the index of its token on the left."""
    width, height = LABEL + d * SIDE, n * SIDE
    fig = plt.figure(figsize=(width * C.MM, height * C.MM))
    ax = fig.add_axes((LABEL / width, 0, d * SIDE / width, 1))
    ax.set(xlim=(0, d), ylim=(n, 0))
    ax.axis("off")
    ax.pcolormesh(np.arange(d + 1), np.arange(n + 1), values.numpy(), cmap=DIV, norm=norm,
                  edgecolors="face", linewidth=0.05)
    ax.add_patch(Rectangle((0, 0), d, n, fill=False, ec=P.RULE, lw=1.0, clip_on=False))
    for i, token in enumerate(rows):
        ax.text(-0.35, i + 0.5, str(token + 1), ha="right", va="center", fontsize=6, color=P.NOTE,
                clip_on=False)
    fig.patch.set_alpha(0)
    C.save(fig, name)


ex, ef = x.abs().max().item(), fx.abs().max().item()
identity = list(range(n))
token_panel(x, identity, "equivariance_x", TwoSlopeNorm(0, -ex, ex))
token_panel(x[p], p.tolist(), "equivariance_Px", TwoSlopeNorm(0, -ex, ex))
token_panel(fx, identity, "equivariance_fx", TwoSlopeNorm(0, -ef, ef))
token_panel(fPx, p.tolist(), "equivariance_fPx", TwoSlopeNorm(0, -ef, ef))

# The second step of the slide: the check of lecture 03 (frame "Six layers, and what they assume"),
# then the same with the causal mask.  The slide leaves out the prints.
print("listing (slide 'Self-attention is permutation equivariant', second step):")
torch.manual_seed(0)
run_listing('''
    x = torch.randn(10, 32)                    # 10 tokens in R^32
    attn = nn.MultiheadAttention(32, 4)        # 4 heads
    f = lambda x, mask=None: x + attn(x, x, x, attn_mask=mask)[0]
    p = torch.randperm(10)                     # a permutation
    print(torch.allclose(f(x[p]), f(x)[p], atol=1e-6))
    mask = torch.ones(10, 10).triu(1).bool()   # no looking ahead
    print(torch.allclose(f(x[p], mask), f(x, mask)[p], atol=1e-6))
''')


# =========================================================================================
# 4. The causal mask
# =========================================================================================
print("\n--- 4. the causal mask")
torch.manual_seed(0)
print("listing (slide 'Masking: no looking ahead'):")
mask_listing = '''
    x = torch.randn(6, 16)                    # 6 tokens in R^16
    Q, K, V = torch.randn(3, 16, 16) / 4      # one random layer
    mask = torch.tril(torch.ones(6, 6)) == 0  # True above the diagonal
    def attend(x, mask):
        s = (x @ Q) @ (x @ K).T / 4           # scores / sqrt(d)
        return s.masked_fill(mask, -torch.inf).softmax(-1) @ (x @ V)
    y = x.clone(); y[-1] = torch.randn(16)    # change the last token
    same = lambda m: torch.allclose(attend(x, m)[:5], attend(y, m)[:5])
    print(same(mask), same(mask & False))     # with, without the mask
'''
scope = {"torch": torch}
torch.manual_seed(0)
source = textwrap.dedent(mask_listing).strip("\n")
print("  | " + source.replace("\n", "\n  | "))
exec(source, scope)
# The figure: the same tensors, for the six tokens of the running sentence.
x, Q, K, mask = scope["x"], scope["Q"], scope["K"], scope["mask"]
S = (x @ Q) @ (x @ K).T / 4
A = S.masked_fill(mask, -torch.inf).softmax(-1)
words = ["The", "dog", "bites", "the", "man", "."]
print("scores (rows t, columns s):\n", S.numpy().round(2))
print("weights:\n", A.numpy().round(2))
print(f"row sums of the weights: {A.sum(1).numpy().round(6).tolist()}; "
      f"max weight above the diagonal: {A.triu(1).max().item()}")

fig, axes = plt.subplots(1, 3, figsize=(108 * C.MM, 40 * C.MM))
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.11, top=0.83, wspace=0.12)
es = S.abs().max().item()
axes[0].imshow(S.numpy(), cmap=DIV, norm=TwoSlopeNorm(0, -es, es))
axes[1].imshow(np.where(mask.numpy(), 0.0, 1.0), cmap=LinearSegmentedColormap.from_list("m", [P.GHOST, "white"]),
               norm=Normalize(0, 1))
axes[2].imshow(A.numpy(), cmap=SEQ, norm=Normalize(0, 1))
for i in range(n):
    for j in range(n):
        cell_text(axes[1], i, j, 0, "0" if j <= i else "−∞", False, 6)
        if j <= i:
            text = "1" if A[i, j] == 1 else f"{A[i, j]:.2f}".lstrip("0")    # .29 for 0.29: it fits the cell
            cell_text(axes[2], i, j, A[i, j], text, A[i, j] > 0.55, 6)
titles = [r"scores $\langle Qx_t,Kx_s\rangle/\sqrt{d}$", r"$+$ mask", r"$=$ after softmax: $a_{ts}$"]
for ax, title in zip(axes, titles):
    ax.set_xticks(range(n), words, fontsize=6.5, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n), words if ax is axes[0] else [], fontsize=6.5)
    ax.tick_params(length=0, pad=1.5, colors=P.INK)
    ax.set_xlabel(title, fontsize=7, labelpad=3)
    for spine in ax.spines.values():
        spine.set_color(P.RULE)
C.save(fig, "mask_matrices")
