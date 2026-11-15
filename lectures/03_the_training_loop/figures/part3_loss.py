"""Figures and numbers for part 3 (Loss) of the lecture slides the_training_loop.tex.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part3_loss.py [preview_dir]

Three experiments.  Every number quoted in part 3 is printed here.

1. An inverse problem: predict y from x = g(y) + noise, with g the target of lecture 02.
   The same hidden layers are trained twice: one output and the squared loss, or a softmax
   over bins of y and the cross-entropy.  Writes inverse_problem_losses.pdf.
2. The digit classifier of lecture 02 with the squared error on the softmax probabilities
   in place of the cross-entropy; nothing else changes.
3. A rare event: 1% of the points of a two-dimensional data set are anomalies.  A small
   network is trained with the cross-entropy, and again with the anomalies weighted by 99.
   Writes rare_event_data.pdf and rare_event_metrics.pdf.

The run takes about a minute and a half on one CPU thread.
"""
import math

import common as C
from common import P, np, plt, torch, nn
from matplotlib.colors import LinearSegmentedColormap

BLUES = LinearSegmentedColormap.from_list("course_blues", ["#ffffff", P.BLUE])


def batches(n, batch_size, generator):
    """Index sets of one epoch: a random order of range(n), cut into mini-batches."""
    return torch.randperm(n, generator=generator).split(batch_size)


def column(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).reshape(-1, 1)


# =======================================================================================
# 1. Why predict a law: the inverse of the target of lecture 02
# =======================================================================================
# y is uniform on [-1, 1] and x = g(y) + s xi.  For |x| > g(1) = 0.441 two values of y give
# the same x, so the conditional law of y given x has two bumps.
S_NOISE, N_TRAIN, N_TEST, K_BINS = 0.05, 10_000, 20_000, 50


def inverse_data(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.uniform(-1, 1, n)
    return C.target(labels) + S_NOISE * rng.standard_normal(n), labels


x_train, y_train = inverse_data(N_TRAIN, 0)
x_test, y_test = inverse_data(N_TEST, 1)

# The exact conditional law, by Bayes' formula on a fine grid of y (uniform prior on [-1, 1])
y_fine = np.linspace(-1, 1, 4001)


def exact_law(inputs):
    """Density of y given x on y_fine, one row per input."""
    log_density = -0.5 * ((np.atleast_1d(inputs)[:, None] - C.target(y_fine)) / S_NOISE) ** 2
    density = np.exp(log_density - log_density.max(1, keepdims=True))
    return density / C.trapz(density, y_fine, axis=1)[:, None]


def exact_mean(inputs):
    return C.trapz(exact_law(inputs) * y_fine, y_fine, axis=1)


def make_net(outputs):
    """Two hidden layers of 64 tanh neurons; only the number of outputs differs."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))


def fit(net, inputs, labels, loss_fn, epochs=30, batch_size=64, lr=1e-3):
    """Adam on mini-batches of 64 with learning rate 1e-3, as for the digits of lecture 02: 4710 steps."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    for _ in range(epochs):
        for batch in batches(len(inputs), batch_size, generator):
            optimizer.zero_grad()
            loss_fn(net(inputs[batch]), labels[batch]).backward()
            optimizer.step()
    return net


edges = np.linspace(-1, 1, K_BINS + 1)
centres = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]


def bin_of(labels):
    return np.clip(np.digitize(labels, edges) - 1, 0, K_BINS - 1)


X_train, X_test = column(x_train), column(x_test)
net_mean = fit(make_net(1), X_train, column(y_train), nn.MSELoss())
net_law = fit(make_net(K_BINS), X_train, torch.tensor(bin_of(y_train)), nn.CrossEntropyLoss())


def predicted_mean(inputs):
    with torch.no_grad():
        return net_mean(column(inputs)).numpy().ravel()


def predicted_law(inputs):
    """Density of y given x: the softmax probabilities divided by the width of a bin."""
    with torch.no_grad():
        return torch.softmax(net_law(column(inputs)), dim=1).numpy() / width


print(f"inverse problem: {N_TRAIN} training and {N_TEST} test samples, noise {S_NOISE}, "
      f"{K_BINS} bins of width {width:.2f}; g(1) = {C.target(1.0):.3f}, max of g on [-1, 1] = {C.target(y_fine).max():.3f}; "
      f"{30 * math.ceil(N_TRAIN / 64)} steps of Adam per network")
print(f"parameters: squared loss {C.count_parameters(net_mean)}, log-loss {C.count_parameters(net_law)}")

# Root mean squared errors on the test set: the network, and the exact conditional mean
rmse_net = np.sqrt(np.mean((predicted_mean(x_test) - y_test) ** 2))
rmse_exact = np.sqrt(np.mean((exact_mean(x_test) - y_test) ** 2))
print(f"test RMSE: squared-loss network {rmse_net:.3f}, exact conditional mean {rmse_exact:.3f}")

# Average log-density of the test labels: the squared loss read as a Gaussian of constant
# variance (its maximum-likelihood value, the training mean squared error), the histogram
# of the log-loss network, and the exact conditional law.
variance = np.mean((predicted_mean(x_train) - y_train) ** 2)
loglik_gauss = np.mean(-0.5 * np.log(2 * np.pi * variance) - (predicted_mean(x_test) - y_test) ** 2 / (2 * variance))
loglik_law = np.mean(np.log(predicted_law(x_test)[np.arange(N_TEST), bin_of(y_test)]))
nearest = np.abs(y_test[:, None] - y_fine).argmin(1)
loglik_exact = np.mean(np.log(np.concatenate(
    [exact_law(x_test[i:i + 2000])[np.arange(len(x_test[i:i + 2000])), nearest[i:i + 2000]]
     for i in range(0, N_TEST, 2000)])))
print(f"training MSE of the squared-loss network = {variance:.4f} (standard deviation {np.sqrt(variance):.3f})")
print(f"mean test log-density: Gaussian around the squared-loss fit {loglik_gauss:+.2f}, "
      f"log-loss histogram {loglik_law:+.2f}, exact law {loglik_exact:+.2f}")

# One input in the two-valued region
X0 = 0.8
mean_at = predicted_mean([X0])[0]
law_at = predicted_law([X0])[0]
exact_at = exact_law([X0])[0]
roots = y_fine[1:][np.diff(np.sign(C.target(y_fine) - X0)) != 0]
peaks = [k for k in range(1, K_BINS - 1) if law_at[k] > law_at[k - 1] and law_at[k] >= law_at[k + 1] and law_at[k] > 1]
gap = (centres > roots[0] + 0.1) & (centres < roots[-1] - 0.1)
print(f"at x = {X0}: squared-loss prediction {mean_at:.3f}, exact conditional mean {exact_mean([X0])[0]:.3f}, "
      f"g(prediction) = {C.target(mean_at):.3f}")
print(f"at x = {X0}: the solutions of g(y) = x are y = {', '.join(f'{r:.3f}' for r in roots)}")
print(f"at x = {X0}: the predicted law peaks in the bins centred at {', '.join(f'{centres[k]:.2f}' for k in peaks)}; "
      f"mass below the squared-loss prediction {width * law_at[centres < mean_at].sum():.2f} "
      f"(exact {C.trapz(exact_at[y_fine < mean_at], y_fine[y_fine < mean_at]):.2f}); "
      f"mass more than 0.1 away from both solutions {width * law_at[gap].sum():.4f}")
print(f"at x = {X0}: distance from the squared-loss prediction to the nearest training label with "
      f"|x_i - {X0}| < 0.05: {np.abs(y_train[np.abs(x_train - X0) < 0.05] - mean_at).min():.3f}")

x_grid = np.linspace(-1.25, 1.25, 400)
fig, (ax_mean, ax_law, ax_slice) = P.panels(3, size=(2.05, 1.8), ratios=[1, 1.14, 0.9])
ax_mean.plot(x_train[:2000], y_train[:2000], ".", color=P.GHOST, ms=2, rasterized=True)   # a fifth of the data
ax_mean.plot(x_grid, exact_mean(x_grid), "--", color=P.INK, lw=1.0, label=r"$\mathbb{E}[y\mid x]$")
ax_mean.plot(x_grid, predicted_mean(x_grid), color=P.ORANGE, lw=1.5, label="network")
ax_mean.axvline(X0, color=P.RULE, lw=0.8, zorder=0)
P.label(ax_mean, "$x$", "$y$", title="Squared loss", xlim=(-1.25, 1.25), ylim=(-1.05, 1.05),
        yticks=[-1, 0, 1], legend=dict(fontsize=6.5, loc="upper left", handlelength=1.6))
image = ax_law.pcolormesh(x_grid, centres, predicted_law(x_grid).T, cmap=BLUES, vmin=0, vmax=6, shading="nearest",
                           rasterized=True)
ax_law.axvline(X0, color=P.RULE, lw=0.8)
fig.colorbar(image, ax=ax_law, pad=0.03).set_label(r"density of $y$ given $x$", fontsize=7)
P.label(ax_law, "$x$", title="Log-loss on 50 bins", xlim=(-1.25, 1.25), ylim=(-1.05, 1.05), yticks=[-1, 0, 1])
ax_slice.bar(centres, law_at, width=width, color=P.BLUE, alpha=0.75, label="network")
ax_slice.plot(y_fine, exact_at, "--", color=P.INK, lw=0.9, label="exact")
ax_slice.axvline(mean_at, color=P.ORANGE, lw=1.5, label="squared loss")
P.label(ax_slice, "$y$", title=f"The law at $x={X0}$", xlim=(-1.35, 1), ylim=(0, 9), xticks=[-1, 0, 1],
        legend=dict(fontsize=6, loc="upper left", handlelength=1.1, handletextpad=0.5, labelspacing=0.25, borderpad=0.3))
C.save(fig, "inverse_problem_losses")


# =======================================================================================
# 2. The digits of lecture 02 with the squared error on the probabilities
# =======================================================================================
def squared_error_on_probabilities(scores, labels):
    """Brier score: |softmax(scores) - e_y|^2, averaged over the mini-batch."""
    return ((torch.softmax(scores, dim=1) - nn.functional.one_hot(labels, 10)) ** 2).sum(1).mean()


mnist_train, mnist_test = C.load_mnist("train"), C.load_mnist("test")
for name, loss_fn in (("cross-entropy", None), ("squared error on the probabilities", squared_error_on_probabilities)):
    print(f"digits, {name}:")
    net, _, test_accuracy = C.train_mnist(C.make_classifier([784, 16, 16, 10]), mnist_train, mnist_test, loss_fn=loss_fn)
    with torch.no_grad():
        probabilities = torch.softmax(net(mnist_test[0]), dim=1)
    own = probabilities[torch.arange(len(probabilities)), mnist_test[1]]
    wrong = probabilities.argmax(1) != mnist_test[1]
    print(f"digits, {name}: test accuracy {100 * test_accuracy[-1]:.2f}%, mean test log-loss "
          f"{-own.log().mean().item():.3f}, {int(wrong.sum())} errors of which "
          f"{int((probabilities.max(1).values[wrong] > 0.9).sum())} with probability above 0.9, "
          f"largest log-loss of one image {-own.log().min().item():.1f}")


# =======================================================================================
# 3. A rare event: 1% of the points are anomalies
# =======================================================================================
# Normal points are N(0, I) in the plane and anomalies N(mu, I) with |mu| = 3, so the
# probability of an anomaly given x is known: logit p(x) = mu.x - |mu|^2 / 2 + log(pi / (1 - pi)).
PREVALENCE, DISTANCE = 0.01, 3.0
MU = DISTANCE * np.array([1.0, 1.0]) / np.sqrt(2)
WEIGHT = (1 - PREVALENCE) / PREVALENCE                      # 99: both classes get the same total weight


def rare_data(n, seed):
    rng = np.random.default_rng(seed)
    labels = (rng.uniform(size=n) < PREVALENCE).astype(int)
    return rng.standard_normal((n, 2)) + labels[:, None] * MU, labels


def exact_posterior(points):
    return C.expit(points @ MU - DISTANCE ** 2 / 2 + np.log(PREVALENCE / (1 - PREVALENCE)))


def Phi(t):
    return 0.5 * (1 + math.erf(t / math.sqrt(2)))


def train_detector(points, labels, weight, epochs=20, batch_size=64, lr=1e-3):
    """The hidden layers of the digit classifier (16 and 16 ReLU neurons) with one output score."""
    net = C.make_classifier([2, 16, 16, 1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight)))
    inputs, targets = torch.tensor(points, dtype=torch.float32), column(labels)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    for _ in range(epochs):
        for batch in batches(len(inputs), batch_size, generator):
            optimizer.zero_grad()
            loss_fn(net(inputs[batch]), targets[batch]).backward()
            optimizer.step()
    return net


def probability(net, points):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(points, dtype=torch.float32))).numpy().ravel()


def metrics(prediction, labels):
    """Confusion matrix and the four summaries of a 0/1 prediction."""
    tp = int(np.sum(prediction & (labels == 1)))
    fp = int(np.sum(prediction & (labels == 0)))
    fn = int(np.sum(~prediction & (labels == 1)))
    tn = int(np.sum(~prediction & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(labels), precision=precision, recall=recall, f1=f1)


def report(name, prediction, labels):
    m = metrics(prediction, labels)
    print(f"{name}: TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']}; accuracy {100 * m['accuracy']:.2f}%, "
          f"precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
    return m


points_train, labels_train = rare_data(20_000, 0)
points_val, labels_val = rare_data(20_000, 1)
points_test, labels_test = rare_data(1_000_000, 2)
print(f"rare event: {len(labels_train)} training points with {labels_train.sum()} anomalies, "
      f"{len(labels_val)} validation points with {labels_val.sum()}, {len(labels_test)} test points with "
      f"{labels_test.sum()} ({100 * labels_test.mean():.2f}%); weight c = {WEIGHT:.0f}")

plain = train_detector(points_train, labels_train, 1.0)
weighted = train_detector(points_train, labels_train, WEIGHT)
p_test, q_test = probability(plain, points_test), probability(weighted, points_test)
p_val = probability(plain, points_val)

thresholds = np.geomspace(1e-3, 0.999, 300)
f1_val = [metrics(p_val > t, labels_val)["f1"] for t in thresholds]
t_f1 = thresholds[int(np.argmax(f1_val))]
t_weight = 1 / (1 + WEIGHT)
print(f"threshold with the best F1 on the validation set: {t_f1:.3f} (F1 = {max(f1_val):.3f}); 1 / (1 + c) = {t_weight:.3f}")

m_trivial = report("always 'normal'", np.zeros(len(labels_test), bool), labels_test)
m_half = report("cross-entropy, threshold 1/2", p_test > 0.5, labels_test)
m_f1 = report(f"cross-entropy, threshold {t_f1:.3f}", p_test > t_f1, labels_test)
m_low = report(f"cross-entropy, threshold {t_weight:.3f}", p_test > t_weight, labels_test)
m_weighted = report("weighted cross-entropy, threshold 1/2", q_test > 0.5, labels_test)
agree = np.mean((p_test > t_weight) == (q_test > 0.5))
print(f"the last two classifiers agree on {100 * agree:.2f}% of the test points")
shift = (np.log(q_test) - np.log1p(-q_test)) - (np.log(p_test) - np.log1p(-p_test))
band = (q_test > 0.05) & (q_test < 0.95)
print(f"score of the weighted network minus score of the plain one, where 0.05 < q < 0.95: "
      f"median {np.median(shift[band]):.2f}, quartiles {np.quantile(shift[band], 0.25):.2f} and "
      f"{np.quantile(shift[band], 0.75):.2f}; log c = {np.log(WEIGHT):.2f}")

# The same quantities for the exact probability p(x): thresholds on p are thresholds on mu.x / |mu|
print("exact values, from the Gaussian distribution function:")
for name, t in (("1/2", 0.5), ("1/(1+c)", t_weight)):
    cut = DISTANCE / 2 + (np.log(t / (1 - t)) - np.log(PREVALENCE / (1 - PREVALENCE))) / DISTANCE
    recall, false_alarm = 1 - Phi(cut - DISTANCE), 1 - Phi(cut)
    precision = PREVALENCE * recall / (PREVALENCE * recall + (1 - PREVALENCE) * false_alarm)
    accuracy = PREVALENCE * recall + (1 - PREVALENCE) * (1 - false_alarm)
    print(f"  threshold {name}: accuracy {100 * accuracy:.2f}%, precision {precision:.3f}, recall {recall:.3f}, "
          f"F1 {2 * precision * recall / (precision + recall):.3f}")
exact_test = exact_posterior(points_test)
report("exact p(x), threshold 1/2 (the classifier of highest accuracy)", exact_test > 0.5, labels_test)
print(f"mean test log-loss: plain network {-np.mean(np.where(labels_test == 1, np.log(p_test), np.log1p(-p_test))):.4f}, "
      f"exact p(x) {-np.mean(np.where(labels_test == 1, np.log(exact_test), np.log1p(-exact_test))):.4f}")

# --- the data alone: the slide's classifier answers "normal" everywhere, so no boundary ---
shown = slice(0, 5000)
normal, anomaly = labels_train[shown] == 0, labels_train[shown] == 1
print(f"scatter plot: the first 5000 training points, {int(anomaly.sum())} of them anomalies")

# Drawn at about its size on the slide (a column of 0.42 \textwidth, 45 mm), so the legend's text matches.
fig, ax = P.panels(figsize=(1.85, 1.85))
ax.plot(*points_train[shown][normal].T, ".", color=P.BLUE, ms=1.4, alpha=0.5, rasterized=True)
ax.plot(*points_train[shown][anomaly].T, "o", color=P.ORANGE, ms=2.8, mec="white", mew=0.3)
ax.plot([], [], "o", color=P.BLUE, ms=3.5, alpha=0.7, label="normal")      # legend handles, larger than the dots
ax.plot([], [], "o", color=P.ORANGE, ms=3.5, mec="white", mew=0.3, label="anomaly")
# The slide reads the two coordinates as two sensors of an aircraft engine; the data stay synthetic,
# hence no ticks and no units.
P.label(ax, "vibration", "temperature", xlim=(-3.6, 6.2), ylim=(-3.6, 6.2), equal=True, ticks=False,
        legend=dict(fontsize=7, loc="lower right", handletextpad=0.2, borderpad=0.4))
ax.xaxis.label.set_size(8)
ax.yaxis.label.set_size(8)
C.save(fig, "rare_event_data")


# --- the four summaries against the threshold, and the precision-recall curve ---
curve = [metrics(p_test > t, labels_test) for t in thresholds]
accuracies, f1s = np.array([m["accuracy"] for m in curve]), np.array([m["f1"] for m in curve])
t_99 = thresholds[accuracies >= 0.99].min()
assert np.all(accuracies[thresholds >= t_99] >= 0.99)
print(f"test accuracy over the thresholds: largest {100 * accuracies.max():.2f}% at t = {thresholds[accuracies.argmax()]:.2f}; "
      f"at least 99% for every t >= {t_99:.3f}; "
      f"at t = {thresholds[-1]:.3f}: accuracy {100 * accuracies[-1]:.2f}%, recall {curve[-1]['recall']:.3f}")
print(f"test F1 over the thresholds: largest {f1s.max():.3f} at t = {thresholds[f1s.argmax()]:.2f}")

fig, (ax_metrics, ax_pr) = P.panels(2, size=(2.85, 1.9), ratios=[1.3, 1])
for key, name, color in (("accuracy", "accuracy", P.GREY), ("precision", "precision", P.BLUE),
                         ("recall", "recall", P.AQUA), ("f1", "$F_1$", P.ORANGE)):
    ax_metrics.semilogx(thresholds, [m[key] for m in curve], color=color, lw=1.8 if key == "f1" else 1.3, label=name)
marked = ((m_half, 0.5, "o", P.INK, "$t=1/2$", (5, 5), "left"),
          (m_f1, t_f1, "s", P.ORANGE, f"$t={t_f1:.2f}$: best $F_1$", (6, -3), "left"),
          (m_low, t_weight, "D", P.GREY, "$t=0.01$, and the\nweighted loss at $t=1/2$", (-7, -1), "right"),
          (m_weighted, None, "^", P.AQUA, None, None, None))
for m, t, marker, color, *_ in marked[:3]:                          # the same markers as in the right panel
    ax_metrics.plot(t, m["f1"], marker, color=color, ms=5, mec="white", mew=0.5, zorder=3)
P.label(ax_metrics, "threshold $t$", title="Predict an anomaly when $p(x;w)>t$",
        xlim=(1e-3, 1), ylim=(0, 1.04),
        legend=dict(fontsize=6, loc="center left", bbox_to_anchor=(0.01, 0.47), labelspacing=0.2, handlelength=1.5,
                    borderpad=0.3))
ax_pr.plot([m["recall"] for m in curve], [m["precision"] for m in curve], color=P.BLUE, lw=1.4)
for m, _, marker, color, name, offset, side in marked:
    ax_pr.plot(m["recall"], m["precision"], marker, color=color, ms=5, mec="white", mew=0.5, zorder=3)
    if name:
        ax_pr.annotate(name, (m["recall"], m["precision"]), xytext=offset, textcoords="offset points", ha=side,
                       va="center", fontsize=6.5, color=P.NOTE, linespacing=1.1)
P.label(ax_pr, "recall", "precision", title="Precision against recall", xlim=(0, 1.02), ylim=(0, 1.04))
C.save(fig, "rare_event_metrics")
