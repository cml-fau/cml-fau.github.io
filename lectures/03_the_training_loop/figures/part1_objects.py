"""Part 1 of the_training_loop.tex: the figure of the slide "The model sees a representation,
not the object" (representation_examples.pdf).

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part1_objects.py [preview_dir]

Six objects and one map phi: {objects} -> R^d for each, side by side.  Every array at the bottom
is computed here from the object above it, not drawn by hand:

    a word       -> its one-hot vector in the vocabulary of GPT-2 (index from notebooks/data/)
    a molecule   -> the adjacency matrix of 4-aminophenol, hydrogens left out
    a sound      -> its log-mel spectrogram, computed as the input of Whisper
    a digit      -> a pen stroke rasterized on 16 x 16 pixels
    a function   -> the first 12 coefficients of f(x) = sin(pi x + 0.5) in the Legendre basis
    a pendulum   -> its state (theta, dtheta/dt)

The script runs in a few seconds and prints the numbers that the figure shows.
"""
import json

import common as C
from common import P, np, plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9ddf5", P.BLUE, "#0f3a73"])
DIV = LinearSegmentedColormap.from_list("div", ["#0f3a73", P.BLUE, "#ffffff", P.ORANGE, "#8f3410"])

# The figure is drawn in millimetres, at the size it has on the slide: a little wider than the
# text, 120 mm of the 128 mm of the slide.  Six columns, one per object; in each, the object,
# the map phi and the array phi(object).
W, H = 120.0, 64.0
LEFT = 15.0                                                   # the column of the labels
WIDTH = (W - LEFT - 1.0) / 6
Y_OBJECT, Y_ARROW, Y_ARRAY = H - 9.5, (H - 18.6, H - 30.0), H - 45.5
Y_LABEL = sum(Y_ARROW) / 2                                    # the name of each map sits on its arrow
SLIDE = "#fafafa"                                             # the background of the slides (metropolis: black!2)
SIDE = 13.4                                                   # the width of the objects and of the matrices
CELL, SLOT = 4.0, 4.1                                         # the width of a vector, and the height of one entry

plt.rcParams["figure.autolayout"] = False                    # the axes fill the figure
plt.rcParams["savefig.dpi"] = 400                             # the spectrogram is an image: keep it sharp
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, W), ylim=(0, H))
ax.set_aspect("equal")
ax.axis("off")


def centre(k):
    return LEFT + 1.0 + (k + 0.5) * WIDTH


# The column of each object, from left to right.
COLUMN = {"digit": 0, "molecule": 1, "pendulum": 2, "word": 3, "function": 4, "sound": 5}


def array(values, xc, yc, width, height, cmap=SEQ, norm=None, lines=True):
    """A matrix drawn as cells, first row on top, centred on (xc, yc)."""
    values = np.atleast_2d(values)
    rows, cols = values.shape
    xs = np.linspace(xc - width / 2, xc + width / 2, cols + 1)
    ys = np.linspace(yc + height / 2, yc - height / 2, rows + 1)
    ax.pcolormesh(xs, ys, values, cmap=cmap, norm=norm or Normalize(0, 1),
                  edgecolors="face", linewidth=0.05, zorder=2)
    if lines:                                                 # one cell per number
        for x in xs[1:-1]:
            ax.plot([x, x], [ys[-1], ys[0]], color="white", lw=0.4, zorder=3)
        for y in ys[1:-1]:
            ax.plot([xs[0], xs[-1]], [y, y], color="white", lw=0.4, zorder=3)
    frame(xc, yc, width, height)
    return xs, ys


def frame(xc, yc, width, height):
    ax.add_patch(Rectangle((xc - width / 2, yc - height / 2), width, height, fill=False,
                           ec=P.RULE, lw=0.5, zorder=4))


def vector(values, xc, yc, cmap=SEQ, norm=None, numbers=False):
    """A column vector drawn as a stack of cells of one height, each with its own frame, centred
    on (xc, yc); None stands for the entries left out.  Returns the centres of the cells."""
    centres = yc + len(values) * SLOT / 2 - (np.arange(len(values)) + 0.5) * SLOT
    for v, y in zip(values, centres):
        if v is None:
            ax.text(xc, y, r"$\vdots$", ha="center", va="center", fontsize=8, color=P.NOTE)
            continue
        colour = cmap(norm(v)) if norm else cmap(float(v))     # a float: an integer would index the table
        ax.add_patch(Rectangle((xc - CELL / 2, y - SLOT / 2 + 0.3), CELL, SLOT - 0.6,
                               fc=colour, ec=P.RULE, lw=0.5, zorder=2))
        if numbers:
            ax.text(xc, y, f"${v:g}$", ha="center", va="center", fontsize=7, zorder=3,
                    color="white" if (norm(v) if norm else v) > 0.6 else P.NOTE)
    return centres


def arrow(k, text):
    x = centre(k)
    ax.add_patch(FancyArrowPatch((x, Y_ARROW[0]), (x, Y_ARROW[1]), arrowstyle="-|>",
                                 mutation_scale=6.5, lw=0.75, color=P.INK, shrinkA=0, shrinkB=0))
    ax.text(x, Y_LABEL, text, ha="center", va="center", fontsize=7.2, color=P.NOTE, zorder=5,
            bbox=dict(boxstyle="square,pad=0.25", fc=SLIDE, ec="none"))   # hides the shaft behind it


# --- the labels: the map itself, read downwards ---------------------------------------------
x_label = LEFT / 2
ax.text(x_label, Y_OBJECT, r"$\{\mathrm{objects}\}$", ha="center", va="center", fontsize=8)
ax.add_patch(FancyArrowPatch((x_label, Y_ARROW[0]), (x_label, Y_ARROW[1] - 0.5), arrowstyle="-|>",
                             mutation_scale=7, lw=0.8, color=P.INK, shrinkA=0, shrinkB=0))
ax.text(x_label - 1.4, sum(Y_ARROW) / 2 + 0.7, r"$\varphi$", ha="right", va="center", fontsize=11)
ax.text(x_label, Y_ARRAY, r"$\mathbb{R}^d$", ha="center", va="center", fontsize=10)
ax.plot([LEFT, LEFT], [1.5, H - 1.5], color=P.RULE, lw=0.5)


# --- 1. a word: one-hot in the vocabulary of GPT-2 ------------------------------------------
encoder = json.loads((C.ROOT / "notebooks" / "data" / "gpt2-encoder.json").read_text(encoding="utf-8"))
word, index = "cat", encoder["cat"]
print(f'word: "{word}" is token {index} of {len(encoder)}: the one-hot vector e_{index} in R^{len(encoder)}')
x = centre(COLUMN["word"])
ax.text(x, Y_OBJECT, f"“{word}”", ha="center", va="center", fontsize=15,
        family="DejaVu Serif", style="italic", color=P.INK)
arrow(COLUMN["word"], "one-hot")
vector([0, 0, None, 0, 1, 0], x, Y_ARRAY, numbers=True)       # e_index, drawn short: the 1 and a few zeros


# --- 2. a molecule: the adjacency matrix of its graph ---------------------------------------
# 4-aminophenol, HO-C6H4-NH2: a ring of six carbons, the oxygen on the first, the nitrogen
# on the fourth.  Atoms 0-5 are the ring, 6 the oxygen, 7 the nitrogen.
element = ["C"] * 6 + ["O", "N"]
bonds = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1), (0, 6, 1), (3, 7, 1)]
adjacency = np.zeros((8, 8))
for a, b, _ in bonds:
    adjacency[a, b] = adjacency[b, a] = 1
print(f"molecule: 4-aminophenol, {len(element)} heavy atoms, {len(bonds)} bonds")
colour_of = {"C": P.GREY, "O": P.ORANGE, "N": P.BLUE}
x = centre(COLUMN["molecule"])
r = 3.0
angles = np.deg2rad([90, 30, -30, -90, -150, 150])
xy = np.array([[x + r * np.cos(t), Y_OBJECT + r * np.sin(t)] for t in angles]
              + [[x, Y_OBJECT + 2 * r], [x, Y_OBJECT - 2 * r]])
for a, b, order in bonds:
    ax.plot(*xy[[a, b]].T, color=P.INK, lw=0.75, zorder=2, solid_capstyle="butt")
    if order == 2:                                            # the second line of a double bond, inside the ring
        inward = (xy[a] + xy[b]) / 2 - [x, Y_OBJECT]
        shift = -0.55 * inward / np.linalg.norm(inward)
        p, q = xy[a] + shift, xy[b] + shift
        p, q = p + 0.18 * (q - p), q - 0.18 * (q - p)
        ax.plot(*np.array([p, q]).T, color=P.INK, lw=0.75, zorder=2, solid_capstyle="butt")
for (u, v), e in zip(xy, element):
    ax.add_patch(Circle((u, v), 0.75 if e == "C" else 1.0, fc=colour_of[e], ec="white", lw=0.5, zorder=3))
arrow(COLUMN["molecule"], "adjacency")
array(adjacency, x, Y_ARRAY, SIDE, SIDE)


# --- 3. a sound: its log-mel spectrogram, as a speech recognizer reads it ------------------
# A synthetic utterance, built as speech is: a voiced source rich in harmonics, whose pitch
# wanders, filtered by four resonances of the vocal tract (the formants), which glide from one
# vowel to the next.  Then a hiss, as in "s", and a little breath and background noise.
# "ai" (0.06-0.32 s), "s" (0.36-0.47 s), "uoa" (0.50-0.82 s), sampled at 16 kHz.
rate, duration = 16000, 0.9
t = np.arange(int(rate * duration)) / rate
rng = np.random.default_rng(0)


def glide(t0, t1, a, b):
    """a before t0, b after t1, a smooth glide between."""
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return a + (b - a) * (1 - np.cos(np.pi * s)) / 2


def gate(t0, t1, rise):
    return np.clip((t - t0) / rise, 0, 1) * np.clip((t1 - t) / rise, 0, 1)


def band(noise, low, high, tilt=None):
    """Noise restricted to [low, high] Hz, falling off as 1 / (1 + f / tilt) if a tilt is given."""
    f = np.fft.rfftfreq(len(noise), 1 / rate)
    mask = 1 / (1 + np.exp(-(f - low) / 150)) / (1 + np.exp((f - high) / 300))
    if tilt:
        mask = mask / (1 + f / tilt)
    return np.fft.irfft(np.fft.rfft(noise) * mask, len(noise))


# Formants F1-F4 (Hz) of the vowels: Peterson and Barney's averages for F1-F3.
vowel = {"a": (730, 1090, 2440, 3500), "i": (270, 2290, 3010, 3700),
         "u": (300, 870, 2240, 3400), "o": (570, 840, 2410, 3500)}
first, second = t < 0.42, t >= 0.42
formants = np.array([glide(0.12, 0.28, vowel["a"][j], vowel["i"][j]) * first
                     + np.where(t < 0.68, glide(0.55, 0.66, vowel["u"][j], vowel["o"][j]),
                                glide(0.66, 0.78, vowel["o"][j], vowel["a"][j])) * second
                     for j in range(4)])
bandwidths, weights = (90, 110, 170, 250), (1, 0.6, 0.35, 0.25)
wander = np.convolve(rng.standard_normal(len(t)), np.hanning(800) / np.hanning(800).sum(), mode="same")
f0 = (glide(0.06, 0.32, 132, 112) * first + glide(0.50, 0.82, 150, 98) * second) * (1 + 0.1 * wander)
phase = 2 * np.pi * np.cumsum(f0) / rate
loudness = gate(0.06, 0.32, 0.025) * glide(0.06, 0.32, 1, 0.75) + gate(0.50, 0.82, 0.03)
voiced = np.zeros_like(t)
for k in range(1, 70):                                        # the harmonics, up to 7.5 kHz
    f = k * f0
    resonance = sum(w / (1 + ((f - F) / (B / 2)) ** 2) for F, B, w in zip(formants, bandwidths, weights))
    voiced += np.where(f < 7500, (resonance + 0.02) / (1 + f / 250), 0) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
voiced *= loudness
hiss = band(rng.standard_normal(len(t)), 3800, 7600) * gate(0.36, 0.47, 0.03)
breath = band(rng.standard_normal(len(t)), 200, 7500, tilt=600) * loudness
sound = (voiced / np.abs(voiced).max() + 0.55 * hiss / np.abs(hiss).max() + 0.04 * breath
         + 0.0015 * rng.standard_normal(len(t)))
sound /= np.abs(sound).max()


# The input of Whisper (Radford et al., 2023): 25 ms Hann windows every 10 ms, centred on the
# frame (the signal reflected at its ends), the power spectrum pooled by 80 triangular filters
# evenly spaced on the mel scale, log10, everything below 8 decades (80 dB) under the maximum
# raised to it, then (x + 4) / 4.  The mel scale and the area-normalized filters are Slaney's,
# the defaults of librosa, which Whisper's filters come from.
def mel(f):
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=float)
    return np.where(f < 1000, 3 * f / 200, 15 + 27 * np.log(np.maximum(f, 1e-9) / 1000) / np.log(6.4))


def hertz(m):
    m = np.asarray(m, dtype=float)
    return np.where(m < 15, 200 * m / 3, 1000 * np.exp(np.log(6.4) * (m - 15) / 27))


def mel_filters(n_mels, n_fft):
    edges = hertz(np.linspace(0, mel(rate / 2), n_mels + 2))
    f = np.fft.rfftfreq(n_fft, 1 / rate)
    rising = (f - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - f) / (edges[2:] - edges[1:-1])[:, None]
    return np.maximum(0, np.minimum(rising, falling)) * (2 / (edges[2:] - edges[:-2]))[:, None]


def log_mel_spectrogram(signal, n_fft=400, hop=160, n_mels=80):
    padded = np.pad(signal, n_fft // 2, mode="reflect")
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)            # periodic Hann
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:-1]  # Whisper drops the last
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    log_spec = np.log10(np.maximum(mel_filters(n_mels, n_fft) @ power.T, 1e-10))
    return (np.maximum(log_spec, log_spec.max() - 8) + 4) / 4                     # mel band x frame


features = log_mel_spectrogram(sound)
n_mels, n_frames = features.shape
print(f"sound: {len(t)} samples at {rate} Hz; log-mel spectrogram {n_mels} mel bands x {n_frames} frames "
      f"(25 ms windows every 10 ms, 0 to {rate // 2} Hz), values from {features.min():.2f} to {features.max():.2f}")
x = centre(COLUMN["sound"])
seconds = SIDE / duration                                      # millimetres per second, the same above and below
ax.plot(x - SIDE / 2 + seconds * t, Y_OBJECT + 5.6 * sound, color=P.BLUE, lw=0.3, zorder=2)
arrow(COLUMN["sound"], "spectrogram")
left, right = x - SIDE / 2 + seconds * 0.01 * np.array([-0.5, n_frames - 0.5])           # frame k is centred at k * 10 ms
ax.imshow(features, cmap=SEQ, vmin=features.max() - 2, vmax=features.max(), origin="lower", aspect="auto",
          interpolation="bilinear", extent=(left, right, Y_ARRAY - 3 * SLOT, Y_ARRAY + 3 * SLOT), zorder=2)
frame((left + right) / 2, Y_ARRAY, right - left, 6 * SLOT)                      # as tall as the vectors


# --- 4. a handwritten digit: a pen stroke on 16 x 16 pixels ---------------------------------
def catmull_rom(points, per_segment=40):
    """A smooth curve through the points."""
    p = np.vstack([points[0], points, points[-1]])
    s = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
    pieces = []
    for i in range(1, len(p) - 2):
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        pieces.append(0.5 * (2 * b + (c - a) * s + (2 * a - 5 * b + 4 * c - d) * s ** 2
                             + (3 * b - a - 3 * c + d) * s ** 3))
    return np.vstack(pieces + [points[-1:]])


# A "2" in a unit box, y upwards, placed in the central 12 x 12 pixels of a 16 x 16 image.  A pixel
# is as dark as the pen covers it: fully within 0.45 pixels of the stroke, not at all beyond 1.25.
stroke = catmull_rom(np.array([[0.18, 0.70], [0.33, 0.90], [0.60, 0.92], [0.76, 0.74], [0.70, 0.52],
                               [0.46, 0.30], [0.20, 0.06], [0.50, 0.10], [0.84, 0.08]]))
n = 16
centres = np.stack(np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5), axis=-1).reshape(-1, 2)
pen = np.column_stack([2 + 12 * stroke[:, 0], n - (2 + 12 * stroke[:, 1])])   # pixel coordinates, row down
distance = np.sqrt(((centres[:, None, :] - pen[None, :, :]) ** 2).sum(-1)).min(1)
pixels = np.clip((1.25 - distance) / 0.8, 0, 1).reshape(n, n)
print(f"digit: {n} x {n} = {pixels.size} pixels, {(pixels > 0).sum()} of them inked")
x = centre(COLUMN["digit"])
box = 12.6
ax.plot(x - box / 2 + box * stroke[:, 0], Y_OBJECT - box / 2 + box * stroke[:, 1], color=P.INK,
        lw=2.1, solid_capstyle="round", solid_joinstyle="round", zorder=2)
arrow(COLUMN["digit"], "pixels")
array(pixels, x, Y_ARRAY, SIDE, SIDE, lines=False)


# --- 5. a function: its coefficients in an orthonormal basis --------------------------------
# a_n = <f, e_n> with e_n = sqrt((2n+1)/2) P_n the orthonormal Legendre polynomials on [-1, 1].
# f makes one oscillation on [-1, 1]; its phase keeps both odd and even coefficients.
def f(x):
    return np.sin(np.pi * x + 0.5)


nodes, weights = np.polynomial.legendre.leggauss(64)
n_coefficients = 12
coefficients = np.array([np.sum(weights * f(nodes) * np.sqrt((2 * n + 1) / 2)
                                * np.polynomial.legendre.Legendre.basis(n)(nodes))
                         for n in range(n_coefficients)])
print("function: Legendre coefficients of f(x) = sin(pi x + 0.5):",
      " ".join(f"{a:+.3f}" for a in coefficients))
x = centre(COLUMN["function"])
grid = np.linspace(-1, 1, 200)
ax.plot([x - SIDE / 2, x + SIDE / 2], [Y_OBJECT, Y_OBJECT], color=P.RULE, lw=0.5, zorder=1)
ax.plot(x + SIDE / 2 * grid, Y_OBJECT + 4.6 * f(grid), color=P.BLUE, lw=1.3, zorder=2)
ax.text(x - SIDE / 2, Y_OBJECT + 5.9, "$f$", ha="left", va="center", fontsize=8.5, color=P.BLUE)
arrow(COLUMN["function"], r"$\langle f,e_n\rangle$")
extreme = np.abs(coefficients).max()
# <f, e_0> is the mean of f, which vanishes: the vector starts at <f, e_1> and ends at <f, e_11>,
# drawn as the one-hot vector is: the first two entries, the dots, the last three.  Those are four
# to five decades smaller, so the colour follows log |a_n| (linear only below 1e-6): weaker, not white.
ys = vector(list(coefficients[1:3]) + [None] + list(coefficients[9:12]), x - 2.6, Y_ARRAY,
            cmap=DIV, norm=SymLogNorm(1e-6, vmin=-extreme, vmax=extreme, base=10))
ax.text(x - 2.6 + CELL / 2 + 0.8, ys[0], r"$\langle f,e_1\rangle$", ha="left", va="center", fontsize=7, color=P.NOTE)


# --- 6. a pendulum: its state, the angle and the angular velocity ---------------------------
theta, omega = np.deg2rad(35), -1.2
state = np.array([theta, omega])
print(f"pendulum: theta = 35 degrees, angular velocity {omega}: state {np.round(state, 2)}")
x = centre(COLUMN["pendulum"])
pivot = np.array([x - 2.8, Y_OBJECT + 7.0])
length = 10.8
bob = pivot + length * np.array([np.sin(theta), -np.cos(theta)])
ax.plot([pivot[0] - 3.0, pivot[0] + 3.0], [pivot[1], pivot[1]], color=P.INK, lw=0.9, solid_capstyle="butt")
for u in np.linspace(pivot[0] - 2.7, pivot[0] + 2.3, 8):     # the ceiling
    ax.plot([u, u + 0.6], [pivot[1], pivot[1] + 0.6], color=P.INK, lw=0.45)
ax.plot([pivot[0], pivot[0]], [pivot[1], pivot[1] - length], color=P.RULE, lw=0.6, ls=(0, (2, 1.5)))
ax.add_patch(Arc(pivot, 7.6, 7.6, theta1=-90, theta2=-90 + 35, color=P.NOTE, lw=0.6))
ax.text(*(pivot + 5.7 * np.array([np.sin(theta / 2), -np.cos(theta / 2)])), r"$\theta$",
        ha="center", va="center", fontsize=8, color=P.NOTE)
ax.plot(*np.array([pivot, bob]).T, color=P.INK, lw=0.9, zorder=2)
ax.add_patch(Circle(bob, 1.55, fc=P.BLUE, ec="white", lw=0.5, zorder=3))
swing = np.deg2rad(np.linspace(35 - 10, 35 - 27, 30))        # omega < 0: the bob swings back to the vertical
path = pivot + length * np.column_stack([np.sin(swing), -np.cos(swing)])
ax.plot(*path[:-4].T, color=P.ORANGE, lw=0.85, solid_capstyle="butt")
ax.add_patch(FancyArrowPatch(path[-5], path[-1], arrowstyle="-|>", mutation_scale=6.5, lw=0.85,
                             color=P.ORANGE, shrinkA=0, shrinkB=0))
arrow(COLUMN["pendulum"], "state")
xv = x - 1.1                                                  # the cells and their short names, centred together
ys = vector(state, xv, Y_ARRAY, cmap=DIV, norm=TwoSlopeNorm(0, -1.4, 1.4))
for y, name in zip(ys, [r"$\theta$", r"$\dot\theta$"]):
    ax.text(xv + CELL / 2 + 0.6, y, name, ha="left", va="center", fontsize=7.5, color=P.NOTE)

fig.patch.set_alpha(0)                                        # the slide's own background shows through
C.save(fig, "representation_examples")
