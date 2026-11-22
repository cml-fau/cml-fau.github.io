"""Part 1 of the_training_loop_ii.tex, "Data and representation": the digits in other coordinates.

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/figures/part1_mnist.py [preview_dir]

Only the data box of the baseline changes.  The network . -> 16 -> 16 -> 10, the cross-entropy
and Adam (learning rate 1e-3, batches of 64, 5 epochs) are those of lecture 02.

1. Scale and mean.  The same images as bytes in [0, 255], as grey levels in [0, 1] (the
   baseline) and with the mean image subtracted, each from three initializations; and the
   largest eigenvalue of the second-moment matrix of the inputs before and after centring.
2. Principal components.  The classifier is trained on the first k principal components of
   the images instead of the 784 pixels (mnist_pca_digits.pdf, mnist_pca_accuracy.pdf).
3. Designed or learned.  The same linear classifier on 16 principal components, on the 784
   pixels and on the 16 features of the last hidden layer of the baseline; the first 16
   principal directions next to the 16 rows of the first weight matrix (mnist_directions.pdf).

Every number quoted on the slides about these experiments is printed here.  The run takes
about three minutes on one CPU thread.
"""
import common as C
from common import P, np, plt, torch, nn

train, test = C.load_mnist("train"), C.load_mnist("test")
(train_images, train_labels), (test_images, test_labels) = train, test

# ---------------------------------------------------------------------------------------
# Principal component analysis of the 60 000 training images
# ---------------------------------------------------------------------------------------
mean = train_images.mean(0)
centred = (train_images - mean).double()
covariance = centred.T @ centred / len(centred)
variances, directions = torch.linalg.eigh(covariance)
variances, directions = variances.flip(0), directions.flip(1)        # decreasing variance
# The sign of an eigenvector is arbitrary; its largest entry is made positive so that the
# coordinates, and with them the training runs below, do not depend on the linear algebra library.
largest = directions.abs().argmax(0)
directions = (directions * directions[largest, torch.arange(784)].sign()).float()
kept = (variances.cumsum(0) / variances.sum()).numpy()              # fraction of the variance in the first k


def components(images, k):
    return (images - mean) @ directions[:, :k]


# ---------------------------------------------------------------------------------------
# 1. Scale and mean of the inputs
# ---------------------------------------------------------------------------------------
# For a linear model w . x with the squared loss, the Hessian is the second-moment matrix
# (1/N) sum x x^T = covariance + mean mean^T: the mean image adds one large eigenvalue.
second_moment = covariance + torch.outer(mean.double(), mean.double())
top_raw, top_centred = torch.linalg.eigvalsh(second_moment)[-1].item(), variances[0].item()
print(f"largest eigenvalue of (1/N) sum x x^T: {top_raw:.2f} for pixels in [0, 1] (|mean image|^2 = "
      f"{(mean.double() ** 2).sum().item():.2f}), {top_centred:.2f} after centring: a factor {top_raw / top_centred:.1f}")
print(f"for bytes in [0, 255] it is multiplied by 255^2 = {255 ** 2}")

versions = {"bytes in [0, 255]": lambda images: 255 * images,
            "grey levels in [0, 1]": lambda images: images,
            "mean image subtracted": lambda images: images - mean}
seeds = (0, 1, 2)                                                    # seed 0 with shuffling seed 1 is lecture 02
for name, version in versions.items():
    after_one, after_five = [], []
    for seed in seeds:
        net = C.make_classifier([784, 16, 16, 10], seed=seed)
        net, _, accuracies = C.train_mnist(net, (version(train_images), train_labels),
                                           (version(test_images), test_labels), shuffle_seed=seed + 1, verbose=False)
        after_one.append(100 * accuracies[1])
        after_five.append(100 * accuracies[-1])
        if name == "grey levels in [0, 1]" and seed == 0:
            baseline, baseline_accuracy = net, accuracies[-1]
    print(f"{name:22s}: test accuracy after 1 epoch {np.round(after_one, 2).tolist()} (mean {np.mean(after_one):.1f}%), "
          f"after 5 epochs {np.round(after_five, 2).tolist()} (mean {np.mean(after_five):.1f}%)")
print(f"baseline of lecture 02: {C.count_parameters(baseline)} parameters, test accuracy {100 * baseline_accuracy:.2f}%")

# ---------------------------------------------------------------------------------------
# 2. The baseline on the first k principal components
# ---------------------------------------------------------------------------------------
ks = [2, 4, 8, 16, 32, 64, 128, 256]
pca_accuracy = []
for k in ks:
    net = C.make_classifier([k, 16, 16, 10])
    net, _, accuracies = C.train_mnist(net, (components(train_images, k), train_labels),
                                       (components(test_images, k), test_labels), verbose=False)
    pca_accuracy.append(accuracies[-1])
    print(f"first {k:3d} principal components: {100 * kept[k - 1]:5.1f}% of the variance, "
          f"{C.count_parameters(net):5d} parameters, test accuracy {100 * accuracies[-1]:.2f}%")

shown = [8, 32, 128, 784]
rows = [torch.cat([(mean + components(digit[None], k) @ directions[:, :k].T).reshape(28, 28) for k in shown], dim=1)
        for digit in test_images[:2]]                                # the first two test images
grey = torch.cat(rows, dim=0).clamp(0, 1).numpy()
# Black ink whose opacity is the grey level, so that the slide's own colour shows between the strokes.
ink = np.zeros(grey.shape + (4,))
ink[..., 3] = grey
# Both figures are drawn at their size on the slide and included unscaled, so that their text
# matches the slide's.  The slide calls k m, the dimension of phi(x) in R^m.
# The reconstructions: each 28 x 28 image in a D x D mm square, the labels in a band below.
D, band = 16.0, 4.5                                                  # mm
W, H = 4 * D, 2 * D + band
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
fig.set_layout_engine("none")              # the axes are placed by hand, in millimetres
ax_digits = fig.add_axes((0, band / H, 1, 2 * D / H))
ax_digits.imshow(ink)
for j, k in enumerate(shown):
    ax_digits.text(28 * j + 14, 57, f"$m={k}$", ha="center", va="top", fontsize=9)   # below the two rows
P.label(ax_digits, off=True)
C.save(fig, "mnist_pca_digits")

fig, ax_accuracy = plt.subplots(figsize=(55 / 25.4, 39.5 / 25.4))   # the width of its column, the height left on the slide
ax_accuracy.semilogx(ks, 100 * np.array(pca_accuracy), "-o", ms=3.5)
ax_accuracy.axhline(100 * baseline_accuracy, color=P.ORANGE, lw=1, ls=":")
ax_accuracy.annotate("784 pixels", (2, 100 * baseline_accuracy), xytext=(0, 3), textcoords="offset points",
                     fontsize=8, color=P.ORANGE)
P.label(ax_accuracy, "components $m$", title="Test accuracy (%)",
        ylim=(0, 109), xticks=[2, 8, 32, 128], xticklabels=[2, 8, 32, 128], yticks=[0, 50, 100])
ax_accuracy.minorticks_off()
ax_accuracy.title.set_fontsize(9)
ax_accuracy.xaxis.label.set_size(9)
ax_accuracy.tick_params(labelsize=8)
C.save(fig, "mnist_pca_accuracy")

# ---------------------------------------------------------------------------------------
# 3. Designed or learned: a linear classifier on 16 coordinates
# ---------------------------------------------------------------------------------------
# The last layer of the baseline is a linear classifier on the 16 features of its second
# hidden layer.  The same linear classifier is trained on 16 principal components, and on the
# pixels themselves (5 epochs as everywhere; 20 epochs to see that it had converged).
for name, pair in {"16 principal components": (components(train_images, 16), components(test_images, 16)),
                   "784 pixels": (train_images, test_images)}.items():
    for epochs in (5, 20):
        linear = C.make_classifier([pair[0].shape[1], 10])
        linear, _, accuracies = C.train_mnist(linear, (pair[0], train_labels), (pair[1], test_labels),
                                              epochs=epochs, verbose=False)
        print(f"linear classifier on {name}, {epochs:2d} epochs: {C.count_parameters(linear)} parameters, "
              f"test accuracy {100 * accuracies[-1]:.2f}%")
print(f"linear classifier on the 16 learned features (the last layer of the baseline): "
      f"test accuracy {100 * baseline_accuracy:.2f}%")

blocks = {"Designed: the first 16 principal directions": directions[:, :16].T,
          "Learned: the 16 rows of $W_1$ in the baseline": baseline[0].weight.detach()}
fig, axes = P.panels(2, size=(2.9, 1.02))
for ax, (title, block) in zip(axes, blocks.items()):
    tiles = (block / block.abs().amax(1, keepdim=True)).reshape(2, 8, 28, 28)      # each direction on its own scale
    tiles = torch.nn.functional.pad(tiles, (1, 1, 1, 1))                           # a margin around each tile
    ax.imshow(tiles.permute(0, 2, 1, 3).reshape(2 * 30, 8 * 30), cmap="RdBu_r", vmin=-1, vmax=1)
    P.label(ax, title=title, off=True)
    ax.title.set_fontsize(9)
C.save(fig, "mnist_directions")
