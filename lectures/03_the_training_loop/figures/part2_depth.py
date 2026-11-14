"""Part 2 (architecture), figures on depth and on approximation rates.

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part2_depth.py [preview_dir]

Both figures are carried over from lecture 02 (sections 13 and 14 of its original script),
with the same code:

- sawtooth.pdf: the hat function T composed with itself.  The script also checks the two
  identities that the frame "What depth buys" states: T composed k times has 2^k linear
  pieces, and the parity of n = 2^k bits equals T^k applied to their mean.
- rates.pdf: the sup error of the least-squares fit of the target against the number of
  pieces, for the four families of lecture 02 (used by the appendix frame on rates).

The run takes a few seconds.
"""
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")                       # NumPy's linear algebra on one thread too

from numpy.polynomial import chebyshev

import common as C
from common import P, np, plt

# ---------------------------------------------------------------------------------------
# 1. What depth buys: the hat function composed with itself
# ---------------------------------------------------------------------------------------
relu = lambda t: np.maximum(t, 0)
tent = lambda t: 2 * relu(t) - 4 * relu(t - 0.5)                    # two ReLU neurons; the hat on [0, 1]
unit = np.linspace(0, 1, 4001)
fig, axes = P.panels(4, size=(1.85, 1.5), sharey=True)
value = unit.copy()
for k, ax in enumerate(axes, start=1):
    value = tent(value)
    ax.plot(unit, value, lw=1.2)
    P.label(ax, "$x$", title=rf"$T^{{\circ{k}}}$: {2 ** k} pieces" if k > 1 else "$T$: 2 pieces",
            xticks=[0, 0.5, 1], yticks=[0, 1])
C.save(fig, "sawtooth")
assert np.allclose(tent(unit), 1 - np.abs(2 * unit - 1))


def compose(t, k):
    for _ in range(k):
        t = tent(t)
    return t


# T^k is the zigzag through the points (j / 2^k, j mod 2): 2^k linear pieces, 2k neurons
for k in range(1, 11):
    nodes = np.arange(2 ** k + 1) / 2 ** k
    assert np.allclose(compose(nodes, k), np.arange(2 ** k + 1) % 2)
    slopes = np.diff(compose(nodes, k)) * 2 ** k
    assert np.allclose(np.abs(slopes), 2 ** k) and np.all(slopes[:-1] * slopes[1:] < 0)
print("sawtooth: T composed k times is the zigzag through (j / 2^k, j mod 2), checked for k = 1, ..., 10")
print("sawtooth: k = 10 uses 2k = 20 neurons; one hidden layer needs at least 2^k - 1 =", 2 ** 10 - 1)


# ---------------------------------------------------------------------------------------
# The parity of n bits: four networks, each checked on every input (frame "What depth buys")
# ---------------------------------------------------------------------------------------
def all_inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)


def or_of_ands(bits):
    """Depth 2 with AND, OR, NOT: one AND term for each input of odd parity."""
    odd = [term for term in all_inputs(bits.shape[1]) if term.sum() % 2 == 1]
    accepted = np.array([(bits == term).all(1) for term in odd])    # each term accepts one input only
    assert (accepted.sum(1) == 1).all()
    return accepted.any(0).astype(float), len(odd)


def xor_bracket(bits):
    """A tournament bracket of 2-input XOR gates; as neurons, XOR(a, b) = relu(a - b) + relu(b - a)."""
    values, gates, levels = bits, 0, 0
    while values.shape[1] > 1:
        a, b = values[:, 0::2], values[:, 1::2]
        values, gates, levels = relu(a - b) + relu(b - a), gates + a.shape[1], levels + 1
    return values[:, 0], gates, levels


def threshold_layer(bits):
    """One hidden layer of n threshold neurons h_j = 1[s >= j], added with the signs +, -, +, ..."""
    n = bits.shape[1]
    hidden = (bits.sum(1)[:, None] >= np.arange(1, n + 1)).astype(float)
    return hidden @ (-1.0) ** np.arange(n), n


def sawtooth_network(bits):
    """k = ceil(log2 n) hidden layers of two ReLU neurons: the hat function T applied k times to s / 2^k."""
    n = bits.shape[1]
    k = int(np.ceil(np.log2(n)))
    hidden = relu(bits @ np.full((n, 2), 2.0 ** -k) + np.array([0, -0.5]))           # reads the bits
    for _ in range(k - 1):
        hidden = relu(hidden @ np.array([[2.0, 2.0], [-4.0, -4.0]]) + np.array([0, -0.5]))
    return hidden @ np.array([2.0, -4.0]), 2 * k, k


for n in (2, 4, 8, 16):
    bits = all_inputs(n)
    parity = bits.sum(1) % 2
    k = int(np.log2(n))
    assert np.allclose(compose(bits.mean(1), k), parity)            # parity = T^k(mean) when n = 2^k
    assert np.array_equal(np.prod(1 - 2 * bits, axis=1), 1 - 2 * parity)     # the product of the signs
    bracket, gates, levels = xor_bracket(bits)
    layer, thresholds = threshold_layer(bits)
    deep, relus, layers = sawtooth_network(bits)
    assert np.allclose(bracket, parity) and np.allclose(layer, parity) and np.allclose(deep, parity)
    line = (f"parity of n = {n:2d} bits, all {2 ** n} inputs: XOR bracket {gates} gates in {levels} levels "
            f"({2 * gates} ReLU neurons); one hidden layer of {thresholds} threshold neurons; "
            f"sawtooth network {relus} ReLU neurons in {layers} layers")
    if n <= 8:
        dnf, terms = or_of_ands(bits)
        assert np.allclose(dnf, parity)
        line += f"; OR of {terms} ANDs"
    print(line)

for n in (10, 20):                                                  # the sizes of the example; not powers of two
    bits = all_inputs(n) if n == 10 else np.tril(np.ones((n + 1, n)), -1)    # n = 20: one input for each sum s
    parity = bits.sum(1) % 2
    layer, thresholds = threshold_layer(bits)
    deep, relus, layers = sawtooth_network(bits)
    assert np.allclose(layer, parity) and np.allclose(deep, parity)
    print(f"parity of n = {n} bits: 2^(n-1) = {2 ** (n - 1)} AND terms; n - 1 = {n - 1} XOR gates; "
          f"{thresholds} threshold neurons in one hidden layer; {relus} ReLU neurons in {layers} layers "
          f"(checked on {len(bits)} inputs)")


# ---------------------------------------------------------------------------------------
# 2. How many pieces?  Approximation rates of the four families of lecture 02 on the target
# ---------------------------------------------------------------------------------------
def gaussian_translates(x, m, width=0.4):
    centers = np.linspace(-1.5, 1.5, m)
    return np.exp(-0.5 * ((x[:, None] - centers) / width) ** 2)


def sigmoid_neurons(x, m, slope=8):
    centers = np.linspace(-1, 1, m - 1)
    return np.column_stack([np.ones_like(x), C.expit(slope * (x[:, None] - centers))])


# Least-squares fits on a fine grid; the polynomials use the Chebyshev basis, which spans
# the same space as the monomials x^j and stays well conditioned for large m.
rate_families = {
    "Polynomials": lambda x, m: chebyshev.chebvander(x, m - 1),
    "Gaussian translates": gaussian_translates,
    "Hat functions": C.hats,
    "Sigmoid neurons": sigmoid_neurons,
}
xf = np.linspace(-1, 1, 2001)
sizes = np.unique(np.round(np.geomspace(2, 128, 25)).astype(int))
fig, ax = P.panels(figsize=(3.4, 2.7))
for name, features in rate_families.items():
    errors = []
    for m in sizes:
        coefficients = np.linalg.lstsq(features(xf, m), C.target(xf), rcond=None)[0]
        errors.append(np.max(np.abs(features(xf, m) @ coefficients - C.target(xf))))
    ax.loglog(sizes, errors, "-o", ms=2.5, lw=1.2, label=name)
    print(f"rates, {name}: " + ", ".join(f"m={m}: {e:.1e}" for m, e in zip(sizes, errors) if m in (4, 8, 16, 32, 64, 128)))
ax.loglog(sizes[6:], 6 * sizes[6:] ** -2.0, "--", color=P.INK, lw=1)
ax.annotate(r"$m^{-2}$", (40, 6 * 40 ** -2.0), xytext=(4, 5), textcoords="offset points", fontsize=8)
P.label(ax, "number of pieces $m$", r"$\max_x|g(x)-f_m(x)|$", ylim=(1e-15, 3e4),
        yticks=[1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1], legend=dict(fontsize=6.5, loc="upper right", ncols=2))
C.save(fig, "rates")
