"""Part 1 of the_training_loop.tex, "Data and representation": three small experiments.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part1_representations.py [preview_dir]

1. Tokens.  The sentence of the slide "Text for a language model" is encoded with the
   vocabulary of GPT-2 (two public files, downloaded once to notebooks/data/), and the
   positional vectors drawn on that slide are computed.
2. Two rings.  The same linear classifier in Cartesian and in polar coordinates
   (rings_coordinates.pdf).
3. Fourier features.  The same network, loss and optimizer on (x, t) and on the first m
   harmonics of x and of t, fitting samples of a struck string, a solution of the wave
   equation (fourier_features.pdf).

Every number quoted on the slides of part 1 that concerns these experiments is printed here.
The run takes about ten minutes on one CPU thread, nearly all of it the 13 fits of the string.
"""
import json
import re
import urllib.request

import common as C
from common import P, np, plt, torch, nn
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------------------
# 1. Tokens: byte-pair encoding with the vocabulary of GPT-2
# ---------------------------------------------------------------------------------------
# The two files define the tokenizer of GPT-2 (Radford et al., 2019), which GPT-3 reuses:
# encoder.json maps each of the 50 257 tokens to its index, vocab.bpe lists the 50 000 merges
# in the order in which they were learned.
GPT2 = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/"
files = {}
for name in ("encoder.json", "vocab.bpe"):
    path = C.ROOT / "notebooks" / "data" / f"gpt2-{name}"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(GPT2 + name, path)
    files[name] = path.read_text(encoding="utf-8")
index_of = json.loads(files["encoder.json"])
merges = [tuple(line.split()) for line in files["vocab.bpe"].split("\n")[1:-1]]
rank = {pair: k for k, pair in enumerate(merges)}

# GPT-2 writes every byte as one printable character; a leading space becomes "Ġ".
printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
byte_symbol = {b: chr(b) for b in printable}
byte_symbol.update({b: chr(256 + k) for k, b in enumerate(b for b in range(256) if b not in printable)})
# Words keep their leading space.  This pattern agrees with GPT-2's on plain ASCII text.
pieces = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def byte_pair_encode(word):
    """Start from single bytes and apply the learned merges, earliest first."""
    parts = [byte_symbol[b] for b in word.encode("utf-8")]
    while len(parts) > 1:
        pairs = list(zip(parts[:-1], parts[1:]))
        best = min(pairs, key=lambda pair: rank.get(pair, float("inf")))
        if best not in rank:
            break
        merged, i = [], 0
        while i < len(parts):
            if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                merged.append(parts[i] + parts[i + 1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        parts = merged
    return parts


def tokenize(text):
    return [token for word in pieces.findall(text) for token in byte_pair_encode(word)]


sentence = "The dog bites the man."
tokens = tokenize(sentence)
print(f"GPT-2 vocabulary: {len(index_of)} tokens, {len(merges)} merges")
print(f'"{sentence}" -> {[t.replace("Ġ", "_") for t in tokens]} -> {[index_of[t] for t in tokens]}')
assert [index_of[t] for t in tokenize("Hello world")] == [15496, 995]      # the usual check of a GPT-2 tokenizer
for model, V, d in (("GPT-2 (smallest)", len(index_of), 768), ("GPT-3", len(index_of), 12288)):
    print(f"embedding matrix of {model}: V d = {V} x {d} = {V * d:,} entries")
print(f"GPT-3, a full context: 2048 tokens x 12288 = {2048 * 12288:,} numbers")
# The sizes of the table "How large is what the model sees?"
print(f"ViT-B/16: 224 x 224 x 3 = {224 * 224 * 3:,} numbers, (224 / 16)^2 = {(224 // 16) ** 2} patches "
      f"of 16 x 16 x 3 = {16 * 16 * 3} numbers")
print(f"latent diffusion, f = 8: 256 x 256 x 3 = {256 * 256 * 3:,} numbers -> 32 x 32 x 4 = {32 * 32 * 4:,}, "
      f"a factor {256 * 256 * 3 / (32 * 32 * 4):.0f}")

# The sinusoidal positions of Vaswani et al. (2017): p_t = (sin w_j t, cos w_j t)_{j < d/2} with
# w_j = 10000^(-2j/d), sine and cosine of each frequency in neighbouring coordinates.  The slide
# draws them with d = 8 for the six positions t = 0, ..., 5 of the sentence.
d_position = 8
omega = 10000.0 ** (-2 * np.arange(d_position // 2) / d_position)
for t in range(len(tokens)):
    p = np.ravel(np.column_stack([np.sin(omega * t), np.cos(omega * t)]))
    print(f"p_{t} (d = {d_position}):", " ".join(f"{v:+.2f}" for v in p))


# ---------------------------------------------------------------------------------------
# 2. Two rings: one linear classifier, two coordinate systems
# ---------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
n_ring = 200
radius = np.concatenate([1.0 + 0.12 * rng.standard_normal(n_ring), 2.2 + 0.12 * rng.standard_normal(n_ring)])
angle = rng.uniform(-np.pi, np.pi, 2 * n_ring)
ring = np.concatenate([np.zeros(n_ring), np.ones(n_ring)])           # 0: inner ring, 1: outer ring
cartesian = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
polar = np.column_stack([radius, angle])


def logistic_regression(features, steps=2000, lr=0.05):
    """A linear score a . z + b fitted with the cross-entropy and Adam; returns (a, b, accuracy)."""
    z = torch.tensor(features, dtype=torch.float32)
    t = torch.tensor(ring, dtype=torch.float32)
    torch.manual_seed(0)
    score = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(score.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(score(z).ravel(), t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = ((score(z).ravel() > 0).float() == t).float().mean().item()
    return score.weight.detach().numpy().ravel(), score.bias.item(), correct


# Drawn at the size of the slide and included unscaled, so that its text matches the slide's: one of the
# two rows of the slide "Example: change of coordinates", next to normalization.pdf.
fig, axes = C.coordinate_panels()                             # lined up with the other row
settings = [("Cartesian", cartesian, "$x_1$", "$x_2$", (-2.9, 2.9), (-2.9, 2.9)),
            ("polar", polar, "$r$", r"$\theta$", (0.4, 2.9), (-np.pi, np.pi))]
for ax, (name, features, xlabel, ylabel, xlim, ylim) in zip(axes, settings):
    a, b, correct = logistic_regression(features)
    print(f"two rings, {name} coordinates: {100 * correct:.1f}% correct, boundary {a[0]:+.3f} z1 {a[1]:+.3f} z2 {b:+.3f} = 0")
    for k, color in enumerate((P.BLUE, P.ORANGE)):
        ax.plot(*features[ring == k].T, "o", color=color, ms=2.2, alpha=0.85, mew=0)
    line = np.array(ylim)                                              # the boundary, drawn as z1(z2)
    ax.plot((-b - a[1] * line) / a[0], line, color=P.INK, lw=1.3)
    P.label(ax, xlabel, ylabel, title=f"{name}: {100 * correct:.0f}%", xlim=xlim, ylim=ylim)
    ax.title.set_fontsize(9)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(labelsize=8)
    ax.yaxis.labelpad = 1
axes[0].set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])
axes[0].set_aspect("equal")
axes[1].set_xticks([1, 2])
axes[1].set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
C.save(fig, "rings_coordinates")


# ---------------------------------------------------------------------------------------
# 3. Fourier features: only the input of the network changes
# ---------------------------------------------------------------------------------------
# A struck string: u_tt = u_xx on [0, 1], u = 0 at both ends, u(x, 0) a narrow bump at x = 0.3
# and u_t(x, 0) = 0.  By d'Alembert, u(x, t) = (F(x - t) + F(x + t)) / 2, where F is the odd,
# 2-periodic extension of the bump: the two halves of the bump run apart and come back inverted
# from each end.  The network sees 4000 samples of u at random points of [0, 1] x [0, 2], one
# period; it is the tanh network . -> 64 -> 64 -> 1 of part 5, with Adam, learning rate 1e-3,
# full batch, and it is judged on a grid of 201 x 401 points.
def bump(s):
    return np.exp(-((s - 0.3) / 0.04) ** 2)


def string(x, t):
    """d'Alembert's solution for the struck string."""
    def extension(s):
        s = (s + 1) % 2 - 1                                               # into [-1, 1)
        return np.where(s >= 0, bump(s), -bump(-s))
    return (extension(x - t) + extension(x + t)) / 2


duration = 2.0
samples = np.random.default_rng(0).uniform((0, 0), (1, duration), size=(4000, 2))
inputs = torch.tensor(samples, dtype=torch.float32)
targets = torch.tensor(string(*samples.T), dtype=torch.float32)[:, None]
grid_x, grid_t = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, duration, 401), indexing="ij")
grid = torch.tensor(np.stack([grid_x.ravel(), grid_t.ravel()], axis=1), dtype=torch.float32)
exact = string(grid_x, grid_t)
power = (exact ** 2).mean()


def harmonics(v, m):
    """phi_m(x, t) = (cos j pi x, sin j pi x, cos(j pi t / 2), sin(j pi t / 2)), j = 1, ..., m; m = 0 returns (x, t).
    The periods, 2 and 4, are twice the sides of the box, so that phi_m is injective on it."""
    if m == 0:
        return v
    j = torch.arange(1, m + 1, dtype=torch.float32)
    phase_x, phase_t = np.pi * j * v[:, :1], np.pi * j * v[:, 1:] / duration
    return torch.cat([torch.cos(phase_x), torch.sin(phase_x), torch.cos(phase_t), torch.sin(phase_t)], dim=1)


def fit(m, steps=3000, seed=0, snapshot=3000):
    """Returns the fit on the grid after `snapshot` steps, the training error after `steps` steps, and the
    relative error ||f - u|| / ||u|| on the grid every 500 steps."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2 if m == 0 else 4 * m, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    features, grid_features = harmonics(inputs, m), harmonics(grid, m)
    relative, shot = {}, None
    for k in range(steps + 1):
        loss = ((net(features) - targets) ** 2).mean()
        if k % 500 == 0 or k == snapshot:
            with torch.no_grad():
                field = net(grid_features).numpy().reshape(exact.shape)
            relative[k] = np.sqrt(((field - exact) ** 2).mean() / power)
            if k == snapshot:
                shot = field
        if k == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return dict(shot=shot, error=loss.item(), relative=relative, parameters=C.count_parameters(net))


LONG = 20_000
runs = {m: fit(m, steps=LONG) for m in (0, 16)}                       # the figure is their step 3000
for m, run in runs.items():
    name = "(x, t)" if m == 0 else f"phi_{m}(x, t)"
    print(f"string, input {name}: {2 if m == 0 else 4 * m} inputs, {run['parameters']} parameters, training error "
          f"{run['error']:.1e} after {LONG} steps; relative error on the grid {100 * run['relative'][3000]:.1f}% after "
          f"3000 steps and {100 * run['relative'][LONG]:.1f}% after {LONG}")
reached = min(k for k, r in runs[0]["relative"].items() if r <= runs[16]["relative"][3000])
print(f"string, input (x, t): below the error of phi_16 at step 3000 from step {reached} (checked every 500 steps)")
for m in (1, 4, 64):                                                  # the scale of the harmonics
    run = fit(m)
    print(f"string, phi_{m}: training error {run['error']:.1e}, relative error on the grid "
          f"{100 * run['relative'][3000]:.1f}% after 3000 steps")
for m in (0, 16):                                                     # how much depends on the initial weights?
    errors = [runs[m]["relative"][3000]] + [fit(m, seed=seed)["relative"][3000] for seed in range(1, 5)]
    print(f"string, m = {m}: relative error after 3000 steps {np.round(100 * np.array(errors), 1)}% for the seeds 0-4")

# Drawn at the size of the slide (the text width, 4.25 in) and included unscaled.
DIV = LinearSegmentedColormap.from_list("div", ["#0f3a73", P.BLUE, "#ffffff", P.ORANGE, "#8f3410"])  # as in part1_objects.py
fig, axes = P.panels(4, ratios=[1, 1, 1, 0.05], figsize=(4.25, 1.72))
fields = [(exact, "exact (d'Alembert)"),
          (runs[0]["shot"], rf"$(x,t)$: ${100 * runs[0]['relative'][3000]:.0f}\%$ error"),
          (runs[16]["shot"], rf"$\varphi(x,t)$: ${100 * runs[16]['relative'][3000]:.0f}\%$ error")]
for ax, (field, title) in zip(axes, fields):
    image = ax.pcolormesh(grid_t, grid_x, field, cmap=DIV, vmin=-1, vmax=1, shading="auto", rasterized=True)
    ax.set(xticks=[0, 1, 2], yticks=[0, 1])
    ax.set_xlabel("$t$", fontsize=7, labelpad=0)
    ax.set_title(title, fontsize=7, pad=3)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
axes[0].set_ylabel("$x$", fontsize=7, labelpad=1)
for ax in axes[1:3]:
    ax.set_yticklabels([])
bar = fig.colorbar(image, cax=axes[3], ticks=[-1, 0, 1])
bar.ax.tick_params(labelsize=6, length=2, pad=1.5)
bar.outline.set_edgecolor(P.RULE)
C.save(fig, "fourier_features")
