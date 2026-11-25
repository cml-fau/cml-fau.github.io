"""Part 1 (tokens), the frames "Tokens become vectors" and "Positional encoding": the two learned
matrices at the entrance of GPT-2 small, with the weights of OpenAI.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part1_embeddings.py [preview_dir]

GPT-2 adds two learned vectors, x_t = e_{i_t} + p_t: the row e_i of the token embedding
(50 257 x 768, model.tokens.weight) and the row p_t of the position embedding (1024 x 768,
model.positions.weight).

- No figure for the tokens: the script prints, for a few words that a mathematics audience
  knows, the tokens whose rows e_j are nearest to e_i in cosine similarity
  cos(e_i, e_j) = <e_i, e_j> / (|e_i| |e_j|), for a table on the slide.  For scale it also prints
  the cosine of random pairs of rows: the rows share a common direction (their mean), so a
  cosine of 0.3 is not "unrelated".  Junk tokens (pieces of words, the same word without its
  space or capitalized) are printed as they come: nothing is filtered.
- positions_similarity.pdf: two panels, cos(p_t, p_s) for t, s = 0, 4, 8, ..., 1020, on the left
  for the learned positions of GPT-2, on the right for the sinusoidal positions of Vaswani et
  al. (2017) with d = 768,
      p_t = (sin w_j t, cos w_j t)_{j < d/2},   w_j = 10000^(-2j/d),
  whose cosine (2/d) sum_j cos(w_j (t - s)) depends on t - s only.  One colour scale, blue
  negative, orange positive.

Printed for the slide: how far the learned similarity is from a function of t - s (the share of
its variance that the means along the diagonals t - s = k explain), the norms of the p_t, and
the oddities: p_0 and the last positions.  A few seconds after the model is loaded.
"""
import common as C
from common import P, np, plt, torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import gpt2

torch.set_num_threads(2)
plt.rcParams["figure.autolayout"] = False
DIV = LinearSegmentedColormap.from_list("div", ["#0f3a73", P.BLUE, "#ffffff", P.ORANGE, "#8f3410"])

model = gpt2.pretrained()
E = model.tokens.weight.detach()                               # (50257, 768): e_i
Pos = model.positions.weight.detach()                          # (1024, 768): p_t
d = E.shape[1]


# ---------------------------------------------------------------------------------------
# 1. Nearest neighbours of a few tokens
# ---------------------------------------------------------------------------------------
def unit(rows):
    return rows / rows.norm(dim=1, keepdim=True)


U = unit(E)
generator = torch.Generator().manual_seed(0)
a, b = torch.randint(0, len(E), (2, 10_000), generator=generator)
random_cos = (U[a] * U[b]).sum(1)
mean_row = E.mean(0)
print(f"token embedding {tuple(E.shape)}; cosine of 10 000 random pairs of rows: mean {random_cos.mean():.3f}, "
      f"5% and 95% quantiles {random_cos.quantile(0.05):.3f}, {random_cos.quantile(0.95):.3f}; "
      f"|mean row| / mean |row| = {mean_row.norm() / E.norm(dim=1).mean():.2f}")

WORDS = [" Germany", " mathematics", " theorem", " Monday", " seven", " river", " red", " cat", " doctor"]
for word in WORDS:
    ids = gpt2.encode(word)
    assert len(ids) == 1, f"{word!r} is not one token: {gpt2.tokenize(word)}"
    i = ids[0]
    cos = U @ U[i]
    cos[i] = -2                                                # leave the token itself out
    top = cos.topk(8)
    print(f"{word!r} (token {i}): " + ", ".join(f"{gpt2.decode([j])!r} {c:.2f}" for c, j in zip(top.values.tolist(), top.indices.tolist())))
del U, cos

# ---------------------------------------------------------------------------------------
# 2. Positions: learned (GPT-2) and sinusoidal (Vaswani et al.)
# ---------------------------------------------------------------------------------------
n = Pos.shape[0]
t = torch.arange(n, dtype=torch.float64)[:, None]
w = 10000.0 ** (-2 * torch.arange(d // 2, dtype=torch.float64) / d)
sinusoidal = torch.zeros(n, d, dtype=torch.float64)
sinusoidal[:, 0::2], sinusoidal[:, 1::2] = torch.sin(w * t), torch.cos(w * t)   # sin and cos side by side
S_learned = (unit(Pos.double()) @ unit(Pos.double()).T).numpy()
S_sin = (unit(sinusoidal) @ unit(sinusoidal).T).numpy()


def toeplitz_share(S, first=0):
    """The share of the variance of S[t, s] (first <= s, t) explained by the mean of its diagonal
    t - s = k: 1 if S depends on t - s only."""
    S = S[first:, first:]
    m = len(S)
    lag = np.subtract.outer(np.arange(m), np.arange(m))
    means = np.array([S.diagonal(k).mean() for k in range(-(m - 1), m)])
    fit = means[lag + m - 1]
    return 1 - ((S - fit) ** 2).sum() / ((S - S.mean()) ** 2).sum()


norms = Pos.norm(dim=1)
print(f"\nposition embedding {tuple(Pos.shape)}: |p_t| for t = 0, 1, 2, 3: " + ", ".join(f"{v:.2f}" for v in norms[:4])
      + f"; median over t = 1..1023 {norms[1:].median():.2f}; |p_t| for t = 1020..1023: " + ", ".join(f"{v:.2f}" for v in norms[-4:]))
print(f"cos(p_0, p_s), s = 1..1023: mean {S_learned[0, 1:].mean():.3f}, max {S_learned[0, 1:].max():.3f}; "
      f"cos(p_1023, p_s), s = 0..1022: mean {S_learned[-1, :-1].mean():.3f}, max {S_learned[-1, :-1].max():.3f}")
inner = S_learned[1:-1, 1:-1]                                 # without the two odd positions 0 and 1023
for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    diag_l, diag_s = np.diagonal(inner, lag), np.diagonal(S_sin, lag)
    print(f"  t - s = {lag:3d}: learned cos(p_t, p_s) mean {diag_l.mean():6.3f} (std over t {diag_l.std():.3f}, "
          f"min {diag_l.min():6.3f}, max {diag_l.max():6.3f}); sinusoidal {diag_s[0]:6.3f}")
print(f"variance of cos(p_t, p_s) explained by a function of t - s: learned {toeplitz_share(S_learned):.3f} "
      f"(t, s = 1..1022: {toeplitz_share(inner):.3f}), sinusoidal {toeplitz_share(S_sin):.6f}")
print(f"learned, the range of cos(p_t, p_s): {S_learned.min():.3f} to {np.max(S_learned - 2 * np.eye(n)):.3f} off the diagonal; "
      f"sinusoidal: {S_sin.min():.3f} to {np.max(S_sin - 2 * np.eye(n)):.3f}")
# Low rank: how many directions carry the positions (the singular values of the centred matrix).
sv = torch.linalg.svdvals(Pos.double() - Pos.double().mean(0))
share = (sv ** 2).cumsum(0) / (sv ** 2).sum()
print("learned positions, share of the variance in the first k directions: "
      + ", ".join(f"k = {k}: {share[k - 1]:.2f}" for k in (1, 2, 3, 4, 8, 16, 32)))

# ---------------------------------------------------------------------------------------
# The figure: two similarity matrices, every 4th position, one colour scale
# ---------------------------------------------------------------------------------------
STEP = 4
shown = np.arange(0, n, STEP)
fig, axes = plt.subplots(1, 2, figsize=(108 * C.MM, 52 * C.MM))
fig.subplots_adjust(left=0.085, right=0.885, bottom=0.14, top=0.90, wspace=0.2)
norm = TwoSlopeNorm(0, -1, 1)
for ax, S, title in zip(axes, (S_learned, S_sin), ("learned (GPT-2)", "sinusoidal, $d=768$")):
    image = ax.imshow(S[np.ix_(shown, shown)], cmap=DIV, norm=norm, interpolation="nearest",
                      extent=(0, n, n, 0))                      # cell k covers the positions 4k..4k+3
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 250, 500, 750, 1000], [0, 250, 500, 750, 1000] if ax is axes[0] else [])
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.set_xlabel("$s$", fontsize=8, labelpad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_ylabel("$t$", fontsize=8, labelpad=0, rotation=0, va="center")
bar = fig.add_axes((0.905, 0.14, 0.015, 0.76))
colorbar = fig.colorbar(image, cax=bar, ticks=[-1, 0, 1])
bar.tick_params(labelsize=6.5, length=2, pad=1)
bar.set_title(r"$\cos$", fontsize=7, pad=3)
colorbar.outline.set_visible(False)
C.save(fig, "positions_similarity")
