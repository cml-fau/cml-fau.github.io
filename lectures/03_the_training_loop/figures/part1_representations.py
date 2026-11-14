"""Part 1 of the_training_loop.tex, "Data and representation": three small experiments.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part1_representations.py [preview_dir]

1. Tokens.  The sentence of the slide "Text: tokens, embeddings, positions" is encoded with
   the vocabulary of GPT-2 (two public files, downloaded once to notebooks/data/).
2. Two rings.  The same linear classifier in Cartesian and in polar coordinates
   (rings_coordinates.pdf).
3. Fourier features.  The same network, loss and optimizer on x and on the first m
   harmonics of x, fitting the rippled target of lecture 02 (fourier_features.pdf).

Every number quoted on the slides of part 1 that concerns these experiments is printed here.
The run takes about two minutes on one CPU thread.
"""
import json
import re
import urllib.request

import common as C
from common import P, np, plt, torch, nn


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


sentence = "The pendulum swings."
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


fig, axes = P.panels(2, size=(1.32, 1.42))
settings = [("Cartesian", cartesian, "$x_1$", "$x_2$", (-2.9, 2.9), (-2.9, 2.9)),
            ("polar", polar, "$r$", r"$\theta$", (0.4, 2.9), (-np.pi, np.pi))]
for ax, (name, features, xlabel, ylabel, xlim, ylim) in zip(axes, settings):
    a, b, correct = logistic_regression(features)
    print(f"two rings, {name} coordinates: {100 * correct:.1f}% correct, boundary {a[0]:+.3f} z1 {a[1]:+.3f} z2 {b:+.3f} = 0")
    for k, color in enumerate((P.BLUE, P.ORANGE)):
        ax.plot(*features[ring == k].T, "o", color=color, ms=1.3, alpha=0.85, mew=0)
    line = np.array(ylim)                                              # the boundary, drawn as z1(z2)
    ax.plot((-b - a[1] * line) / a[0], line, color=P.INK, lw=1.1)
    P.label(ax, xlabel, ylabel, title=f"{name}: {100 * correct:.0f}%", xlim=xlim, ylim=ylim)
    ax.title.set_fontsize(9)
    ax.yaxis.labelpad = 1
axes[0].set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])
axes[0].set_aspect("equal")
axes[1].set_xticks([1, 2])
axes[1].set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
C.save(fig, "rings_coordinates")


# ---------------------------------------------------------------------------------------
# 3. Fourier features: only the input of the network changes
# ---------------------------------------------------------------------------------------
# The experiment of "Surprise 3" (spectral bias, part 5): a tanh network . -> 64 -> 64 -> 1,
# Adam with learning rate 1e-3, full batch, the 160 samples of the target with a ripple.
# With m = 0 the network sees x and the run is the one of part 5, step for step.
def rippled(t):
    return torch.sin(3 * t) + 0.3 * t + 0.3 * torch.sin(20 * t)


inputs = torch.tensor(C.x, dtype=torch.float32)[:, None]
grid = torch.linspace(-1, 1, 2001)[:, None]                          # the fit is judged between the samples too
targets, ripple = rippled(inputs), 0.3 * torch.sin(20 * inputs)


def harmonics(t, m, period=4.0):
    """phi_m(t) = (cos(2 pi j t / period), sin(2 pi j t / period)), j = 1, ..., m; m = 0 returns t."""
    if m == 0:
        return t
    phase = 2 * np.pi * torch.arange(1, m + 1, dtype=torch.float32) * t / period
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)


def fit(m, steps=3000, seed=0, period=4.0, snapshot=1000):
    """Returns the training error and the missing fraction of the ripple at every step, the first
    step from which less than 10% of the ripple is missing, the fit on the grid after `snapshot`
    steps, and the error on the grid at the end."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(max(1, 2 * m), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    features, grid_features = harmonics(inputs, m, period), harmonics(grid, m, period)
    errors, missing, shot = [], [], None
    for k in range(steps + 1):
        prediction = net(features)
        loss = ((prediction - targets) ** 2).mean()
        residual = (targets - prediction).detach()
        errors.append(loss.item())
        missing.append(((residual * ripple).sum() / (ripple * ripple).sum()).item())
        if k == snapshot:
            shot = net(grid_features).detach().numpy().ravel()
        if k == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        grid_error = ((net(grid_features) - rippled(grid)) ** 2).mean().item()
    missing = np.array(missing)
    late = np.flatnonzero(np.abs(missing) >= 0.1)
    learned = int(late[-1]) + 1 if len(late) else 0
    return dict(errors=np.array(errors), missing=missing, learned=learned, shot=shot, grid_error=grid_error,
                parameters=C.count_parameters(net))


sweep = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
runs = {m: fit(m) for m in sweep}
for m, run in runs.items():
    name = "x itself" if m == 0 else f"m = {m:3d} harmonics"
    print(f"Fourier features, {name}: {max(1, 2 * m):3d} inputs, {run['parameters']} parameters, ripple learned from step "
          f"{run['learned']:4d}; step 100: {run['errors'][100]:.2e}, step 1000: {run['errors'][1000]:.2e} "
          f"({100 * run['missing'][1000]:+.1f}% of the ripple missing), step 3000: {run['errors'][3000]:.2e}; "
          f"error on the grid after 3000 steps: {run['grid_error']:.2e}")
for m in (0, 16):                                                    # how much depends on the initial weights?
    steps_needed = [runs[m]["learned"]] + [fit(m, seed=seed)["learned"] for seed in range(1, 5)]
    print(f"Fourier features, m = {m}: ripple learned from step {steps_needed} for the seeds 0-4")

# A representation can also lose information: with period 2 the two end points get the same
# features, phi(-1) = phi(1), and the best any model can do there is the mean of g(-1) and g(1).
periodic = fit(16, period=2.0)
floor = 2 * ((rippled(torch.tensor(1.0)) - rippled(torch.tensor(-1.0))) / 2).item() ** 2 / len(inputs)
print(f"Fourier features of period 2, m = 16: training error {periodic['errors'][1000]:.2e} after 1000 steps and "
      f"{periodic['errors'][3000]:.2e} after 3000; the floor (g(1) - g(-1))^2 / (2 N) is {floor:.2e}")

fig, (ax_fit, ax_ripple, ax_scale) = P.panels(3, size=(2.05, 1.75), ratios=[1.15, 1, 1])
grid_x = grid.numpy().ravel()
ax_fit.plot(grid_x, rippled(grid).numpy().ravel(), color=P.GHOST, lw=2.4, label="target")
ax_fit.plot(grid_x, runs[0]["shot"], color=P.ORANGE, lw=1.1, label="input $x$")
ax_fit.plot(grid_x, runs[16]["shot"], color=P.BLUE, lw=0.9, label=r"input $\varphi_{16}(x)$")
P.label(ax_fit, "$x$", title="After 1000 steps", ylim=(-1.75, 1.6), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
        legend=dict(fontsize=6.5, loc="lower right", handlelength=1.2, borderpad=0.3, labelspacing=0.25))
steps = np.arange(1, 3001)
for m, color, name in ((0, P.ORANGE, "$x$"), (4, P.AQUA, r"$\varphi_4$"), (16, P.BLUE, r"$\varphi_{16}$")):
    ax_ripple.semilogx(steps, runs[m]["missing"][1:], color=color, label=name)
P.label(ax_ripple, "step $k$", title="Ripple still missing", zero="h", ylim=(-0.08, 1.45), yticks=[0, 0.5, 1],
        xticks=[1, 10, 100, 1000], legend=dict(fontsize=6.5, loc="upper center", ncols=3, handlelength=1.2,
                                               columnspacing=0.9))
ax_ripple.minorticks_off()
harmonic_counts = sweep[1:]
ax_scale.loglog(harmonic_counts, [runs[m]["grid_error"] for m in harmonic_counts], "-o", ms=3)
ax_scale.loglog(harmonic_counts, [runs[m]["errors"][-1] for m in harmonic_counts], "--o", ms=3, color=P.GREY)
ax_scale.axhline(runs[0]["grid_error"], color=P.ORANGE, lw=1, ls=":")
ax_scale.annotate("input $x$", (1, runs[0]["grid_error"]), xytext=(0, 3), textcoords="offset points", fontsize=6.5,
                  color=P.ORANGE)
ax_scale.text(105, 0.25, "between samples", fontsize=6.5, color=P.BLUE, ha="right", va="center")
ax_scale.text(1, 2.5e-8, "on the samples", fontsize=6.5, color=P.NOTE, ha="left", va="center")
P.label(ax_scale, "harmonics $m$", title="Error after 3000 steps", xticks=[1, 4, 16, 64, 256],
        xticklabels=[1, 4, 16, 64, 256], ylim=(4e-9, 2), yticks=[1e-8, 1e-6, 1e-4, 1e-2, 1])
ax_scale.minorticks_off()
for ax in (ax_fit, ax_ripple, ax_scale):
    ax.title.set_fontsize(9)
C.save(fig, "fourier_features")
