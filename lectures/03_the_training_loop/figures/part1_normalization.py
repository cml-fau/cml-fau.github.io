"""Part 1 of the_training_loop.tex: the standardization row of the slide "Example: change of
coordinates" (normalization.pdf).

Run from the repository root:

    .venv/bin/python lectures/03_the_training_loop/figures/part1_normalization.py [preview_dir]

Two features, n = 1000 samples of a skewed Gaussian: each coordinate is skew-normal, x_1 with
location 20, scale 30 and shape 8, x_2 with location 5, scale 3 and shape 4, independent.  The
figure shows the cloud before and after standardizing each coordinate,

    z_j = (x_j - mean_j) / sd_j,

with the mean and the standard deviation of the sample, both panels drawn to scale (equal axes).
The script runs in a second and prints the numbers that the slide quotes.
"""
import common as C
from common import P, np
from matplotlib.patches import Rectangle


def skew_normal(location, scale, shape, n, rng):
    """x = location + scale (delta |u0| + sqrt(1 - delta^2) u1), u0, u1 ~ N(0, 1) independent and
    delta = shape / sqrt(1 + shape^2) (Azzalini, 1985)."""
    delta = shape / np.sqrt(1 + shape ** 2)
    u0, u1 = rng.standard_normal((2, n))
    return location + scale * (delta * np.abs(u0) + np.sqrt(1 - delta ** 2) * u1)


def skewness(v):
    return (((v - v.mean(0)) / v.std(0)) ** 3).mean(0)


rng = np.random.default_rng(0)
n = 1000
x = np.column_stack([skew_normal(20.0, 30.0, 8.0, n, rng), skew_normal(5.0, 3.0, 4.0, n, rng)])
mean, sd = x.mean(0), x.std(0)
z = (x - mean) / sd

for j in range(2):
    print(f"raw x_{j + 1}: mean {mean[j]:.1f}, sd {sd[j]:.2f}, range [{x[:, j].min():.1f}, {x[:, j].max():.1f}], "
          f"skewness {skewness(x)[j]:.2f}")
print(f"ratio of the standard deviations s_1 / s_2 = {sd[0] / sd[1]:.1f}")
for j in range(2):
    print(f"standardized z_{j + 1}: mean {z[:, j].mean():+.1e}, sd {z[:, j].std():.3f}, "
          f"range [{z[:, j].min():.2f}, {z[:, j].max():.2f}], skewness {skewness(z)[j]:.2f}")

# Drawn at the size of the slide and included unscaled, so that its text matches the slide's.
# One of the two rows of the slide "Example: change of coordinates", next to rings_coordinates.pdf.
# Equal axes in both panels; the dashed box is the mean +- one standard deviation per coordinate.
fig, axes = C.coordinate_panels()                             # lined up with the other row
settings = [(x, mean, sd, ("$x_1$", "$x_2$"), "raw", (-15, 150), (-82.5, 82.5)),
            (z, np.zeros(2), np.ones(2), ("$z_1$", "$z_2$"), "standardized", (-3.5, 5.5), (-4.5, 4.5))]
for ax, (points, centre, spread, (xlabel, ylabel), title, xlim, ylim) in zip(axes, settings):
    ax.axhline(0, color=P.RULE, lw=0.8, zorder=0)
    ax.axvline(0, color=P.RULE, lw=0.8, zorder=0)
    ax.plot(*points.T, "o", color=P.BLUE, ms=1.8, alpha=0.45, mew=0)
    ax.add_patch(Rectangle(centre - spread, *(2 * spread), fill=False, ec=P.INK, lw=1.0, ls=(0, (2.5, 1.5)), zorder=3))
    ax.plot(*centre, "+", color=P.INK, ms=6, mew=1.2, zorder=3)
    P.label(ax, xlabel, ylabel, title=title, xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    ax.title.set_fontsize(9)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(labelsize=8)
    ax.yaxis.labelpad = 1
axes[0].set(xticks=[0, 100], yticks=[-50, 0, 50])
axes[1].set(xticks=[-3, 0, 3], yticks=[-3, 0, 3])
C.save(fig, "normalization")
