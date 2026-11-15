"""Figures of part 5 of the_training_loop.tex about scale.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part5_scale.py [preview_dir]

1. parameters_vs_data: parameters against the size of the training set for some landmark
   models, with the line P = N.  The counts are those stated in the papers cited on the slide.
2. scaling_mnist: a miniature scaling experiment on the digit classifier of lecture 02.  The
   loop is the baseline (cross-entropy, Adam with learning rate 1e-3, batches of 64, 4690
   steps); only the number of training images and the width of the two hidden layers change.

Every number quoted on the slides is printed below.  The run takes about five minutes on one
CPU thread (24 short training runs).
"""
import common as C
from common import P, np, plt, torch, nn

# ---------------------------------------------------------------------------------------
# 1. Parameters against data
# ---------------------------------------------------------------------------------------
# ResNet-50: the paper gives the architecture (He et al. 2016, Table 1) but not the number of
# parameters, so we count them: convolutions without bias, two batch-norm parameters per
# channel, a projection shortcut where the shape changes, and the final linear layer.
def resnet50_parameters():
    conv = lambda c_in, c_out, k: k * k * c_in * c_out + 2 * c_out          # convolution + batch norm
    total, c_in = conv(3, 64, 7), 64
    for width, blocks in ((64, 3), (128, 4), (256, 6), (512, 3)):
        for block in range(blocks):
            total += conv(c_in, width, 1) + conv(width, width, 3) + conv(width, 4 * width, 1)
            if block == 0:
                total += conv(c_in, 4 * width, 1)                           # projection shortcut
            c_in = 4 * width
    return total + 2048 * 1000 + 1000


n_resnet50 = resnet50_parameters()
n_digit = C.count_parameters(C.make_classifier([784, 16, 16, 10]))
print(f"ResNet-50 counted from Table 1 of He et al.: {n_resnet50:,} parameters; digit classifier: {n_digit:,}")

# (name, parameters P, training-set size N, kind, label offset in points, horizontal alignment)
# N counts labelled examples: images for the classifiers and the diffusion models (image-caption
# pairs for latent diffusion), words or tokens for the language models, in which every token is
# a prediction target.  Kimi K2 and DeepSeek-V4-Pro are mixtures of experts: P is the total
# count, of which each token uses 32.6e9 and 49e9.
models = [
    ("LeNet-5 (1998)",          6.0e4,      6.0e4,   "image", (6, 3),    "left"),    # LeCun et al. 1998; MNIST
    ("AlexNet (2012)",          6.0e7,      1.2e6,   "image", (-5, -2),  "right"),  # Krizhevsky et al. 2012; ImageNet
    ("ResNet-50 (2015)",        n_resnet50, 1.28e6,  "image", (34, -17), "left"),    # He et al. 2016; ImageNet
    ("Inception\non CIFAR-10",  1_649_402,  5.0e4,   "image", (0, 6),    "center"),  # Zhang et al. 2017, Table 1
    ("ViT-H/14 (2021)",         6.32e8,     3.03e8,  "image", (8, -26),  "left"),    # Dosovitskiy et al. 2021; JFT-300M
    ("BERT-large (2018)",       3.4e8,      3.3e9,   "text",  (6, -9),   "left"),    # Devlin et al. 2019; words
    ("GPT-3 (2020)",            1.75e11,    3.0e11,  "text",  (-6, 4),   "right"),   # Brown et al. 2020; tokens
    ("Chinchilla (2022)",       7.0e10,     1.4e12,  "text",  (0, -11),  "center"),  # Hoffmann et al. 2022; tokens
    ("Kimi K2 (2025)",          1.04e12,    1.55e13, "text",  (0, -10),  "center"),  # Kimi Team 2025, Section 2; tokens
    ("DeepSeek-V4-Pro (2026)",  1.6e12,     3.3e13,  "text",  (-30, 5),  "right"),   # DeepSeek-AI 2026, Section 1; tokens
    ("DiT-XL/2 (2023)",         6.75e8,     1.28e6,  "diffusion", (0, 6),  "center"),  # Peebles and Xie 2023, Table 4; ImageNet
    ("Latent diffusion (2022)", 1.45e9,     4.0e8,   "diffusion", (6, 15), "right"),   # Rombach et al. 2022, Section 4.3.1; LAION-400M
]
print("parameters P, training-set size N and the ratio P/N:")
for name, parameters, samples, kind, *_ in models:
    print(f"  {name.replace(chr(10), ' '):24s} P = {parameters:9.3g}  N = {samples:8.3g} ({'words or tokens' if kind == 'text' else 'images'})"
          f"  P/N = {parameters / samples:.3g}")
print(f"  digit classifier of lecture 02: P = {n_digit}, N = 60000 images, P/N = {n_digit / 60000:.2f}")

fig, ax = P.panels(figsize=(4.3, 2.65))       # the width of the slide
diagonal = np.array([1e4, 1e14])
ax.fill_between(diagonal, diagonal, 1e14, color=P.GHOST, alpha=0.45, lw=0)
ax.loglog(diagonal, diagonal, color=P.INK, ls="--", lw=1)
for kind, color, label in (("image", P.BLUE, "classifiers"), ("diffusion", P.AQUA, "diffusion"),
                           ("text", P.ORANGE, "language")):
    chosen = [m for m in models if m[3] == kind]
    ax.loglog([m[2] for m in chosen], [m[1] for m in chosen], "o", color=color, ms=4.5, label=label)
for name, parameters, samples, kind, offset, align in models:
    leader = dict(arrowstyle="-", color=P.NOTE, lw=0.5, shrinkA=1, shrinkB=3) if max(map(abs, offset)) > 12 else None
    ax.annotate(name, (samples, parameters), xytext=offset, textcoords="offset points", fontsize=6.5,
                ha=align, linespacing=0.95, arrowprops=leader)
ax.loglog([6e4], [n_digit], "o", mfc="white", mec=P.INK, ms=4.5)
ax.annotate("our digit classifier", (6e4, n_digit), xytext=(6, -3), textcoords="offset points", fontsize=6.5)
P.label(ax, "size of the training set $N$", "parameters $P$", xlim=(1e4, 1e14), ylim=(5e3, 1e13),
        xticks=[1e4, 1e6, 1e8, 1e10, 1e12, 1e14], yticks=[1e4, 1e6, 1e8, 1e10, 1e12],
        legend=dict(fontsize=6.5, loc="lower right", borderpad=0.3, handletextpad=0.2, labelspacing=0.25))
C.save(fig, "parameters_vs_data")


# ---------------------------------------------------------------------------------------
# 2. A miniature scaling experiment on MNIST
# ---------------------------------------------------------------------------------------
train_images, train_labels = C.load_mnist("train")
test = C.load_mnist("test")
STEPS = 4690                                           # the 5 epochs of the baseline, for every run


def test_error(width, n_train):
    """Test error (%) of 784 -> width -> width -> 10 trained on the first n_train images."""
    net = C.make_classifier([784, width, width, 10])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    cross_entropy = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(1)
    images, labels = train_images[:n_train], train_labels[:n_train]
    step = 0
    while step < STEPS:                                # as many passes through the subset as needed
        for batch in torch.randperm(n_train, generator=generator).split(64):
            optimizer.zero_grad()
            cross_entropy(net(images[batch]), labels[batch]).backward()
            optimizer.step()
            step += 1
            if step == STEPS:
                break
    return 100 * (1 - C.accuracy(net, *test)), C.count_parameters(net)


def slope(values, errors):
    """Least-squares exponent a of errors ~ values^(-a) on logarithmic axes."""
    return -np.polyfit(np.log(values), np.log(errors), 1)[0]


sizes = [500, 1000, 2000, 4000, 8000, 16_000, 32_000, 60_000]
widths = [8, 16, 32, 64, 128, 256]
small, large, few = 16, 128, 2000
results = {}
for width, n_train in ([(w, n) for w in (small, large) for n in sizes]
                       + [(w, n) for n in (60_000, few) for w in widths]):
    if (width, n_train) not in results:
        results[width, n_train] = test_error(width, n_train)
        print(f"  width {width:3d} ({results[width, n_train][1]:6d} parameters), {n_train:5d} images: "
              f"test error {results[width, n_train][0]:.2f}%", flush=True)
parameters = {width: results[width, 60_000][1] for width in widths}
print(f"baseline (width 16, 60000 images): test accuracy {100 - results[16, 60_000][0]:.2f}%")

fig, (ax_data, ax_size) = P.panels(2, size=(2.75, 2.05), sharey=True)
for width, color in ((small, P.ORANGE), (large, P.BLUE)):
    errors = [results[width, n][0] for n in sizes]
    ax_data.loglog(sizes, errors, "-o", ms=3, color=color, label=f"$P={parameters[width]:,}$".replace(",", r"\,"))
exponent_data = slope(sizes, [results[large, n][0] for n in sizes])
exponent_data_small = slope(sizes[:4], [results[small, n][0] for n in sizes[:4]])
exponent_data_small_end = slope(sizes[-3:], [results[small, n][0] for n in sizes[-3:]])
reference = np.array([sizes[0], sizes[-1]])
ax_data.loglog(reference, 0.62 * results[large, sizes[0]][0] * (reference / sizes[0]) ** -exponent_data,
               color=P.INK, ls="--", lw=0.9)
ax_data.annotate(rf"$\propto N^{{-{exponent_data:.2f}}}$", (560, 3.3), fontsize=7.5)
P.label(ax_data, "training images $N$", "test error (%)", title="More data",
        legend=dict(fontsize=6.5, loc="upper right"))

for n_train, color in ((few, P.ORANGE), (60_000, P.BLUE)):
    errors = [results[width, n_train][0] for width in widths]
    ax_size.loglog([parameters[w] for w in widths], errors, "-o", ms=3, color=color,
                   label=f"$N={n_train:,}$".replace(",", r"\,"))
counts = np.array([parameters[w] for w in widths])
exponent_size = slope(counts, [results[w, 60_000][0] for w in widths])
exponent_size_few = slope(counts[2:], [results[w, few][0] for w in widths[2:]])
reference = np.array([counts[0], counts[-1]])
ax_size.loglog(reference, 0.62 * results[widths[0], 60_000][0] * (reference / counts[0]) ** -exponent_size,
               color=P.INK, ls="--", lw=0.9)
ax_size.annotate(rf"$\propto P^{{-{exponent_size:.2f}}}$", (6.6e3, 2.3), fontsize=7.5)
P.label(ax_size, "parameters $P$", title="More parameters", legend=dict(fontsize=6.5, loc="upper right"))
for ax in (ax_data, ax_size):
    ax.set_ylim(1.1, 42)
    ax.set_yticks([2, 5, 10, 20, 40], labels=["2", "5", "10", "20", "40"])
    ax.minorticks_off()
C.save(fig, "scaling_mnist")

print(f"scaling in the data: width {large}, error from {results[large, sizes[0]][0]:.2f}% ({sizes[0]} images) to "
      f"{results[large, sizes[-1]][0]:.2f}% ({sizes[-1]} images), fitted exponent {exponent_data:.3f}")
print(f"  width {small}: from {results[small, sizes[0]][0]:.2f}% to {results[small, sizes[-1]][0]:.2f}%; exponent "
      f"{exponent_data_small:.3f} over the first four sizes and {exponent_data_small_end:.3f} over the last three")
print(f"scaling in the parameters: 60000 images, error from {results[widths[0], 60_000][0]:.2f}% (width {widths[0]}, "
      f"{parameters[widths[0]]} parameters) to {results[widths[-1], 60_000][0]:.2f}% (width {widths[-1]}, "
      f"{parameters[widths[-1]]} parameters), fitted exponent {exponent_size:.3f}")
print(f"  {few} images: from {results[widths[0], few][0]:.2f}% to {results[widths[-1], few][0]:.2f}%; exponent "
      f"{exponent_size_few:.3f} from width {widths[2]} on")

# ---------------------------------------------------------------------------------------
# 3. Arithmetic quoted with the published scaling laws (Kaplan et al. 2020, equations 1.1, 1.2)
# ---------------------------------------------------------------------------------------
for name, alpha in (("parameters", 0.076), ("data", 0.095)):
    print(f"Kaplan et al., exponent {alpha} in the {name}: doubling lowers the loss by {100 * (1 - 2 ** -alpha):.1f}%, "
          f"a factor 10 by {100 * (1 - 10 ** -alpha):.1f}%; halving the loss takes a factor {2 ** (1 / alpha):.3g}")
print(f"our exponents: doubling the data lowers the error by {100 * (1 - 2 ** -exponent_data):.0f}%, "
      f"doubling the parameters by {100 * (1 - 2 ** -exponent_size):.0f}%")
print(f"Chinchilla: {1.4e12 / 7e10:.0f} training tokens per parameter; Gopher: {3e11 / 2.8e11:.1f}; GPT-3: {3e11 / 1.75e11:.1f}")
