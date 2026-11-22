"""Part 2 (architecture), the pixel panels of the frame "Invariance and equivariance".

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/figures/part2_symmetry.py [preview_dir]

The frame draws two commutative squares in TikZ; this script draws the four images that sit at
their corners, each as its own PDF, at the size it has on the slide:

    symmetry_x.pdf       x, the "2" of the first slide of part 1, on a periodic 20 x 20 grid
    symmetry_Tx.pdf      Tx, the same image shifted by 5 rows and 6 columns
    symmetry_kx.pdf      k * x, its convolution with the 3 x 3 Laplacian stencil
    symmetry_kTx.pdf     k * (Tx), which is T(k * x)

The script checks the two identities that the frame states, k * (Tx) = T(k * x) and, for the
sum over the pixels, sum(Tx) = sum(x), and prints the numbers.  It runs in a second.
"""
import common as C
from common import P, np, plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

# The colour scales of part 1: ink in blue, signed values blue (negative) to orange (positive).
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9ddf5", P.BLUE, "#0f3a73"])
DIV = LinearSegmentedColormap.from_list("div", ["#0f3a73", P.BLUE, "#ffffff", P.ORANGE, "#8f3410"])

SIDE = 14.0                                                   # the side of a panel on the slide, in mm
N = 20                                                        # the grid is N x N, periodic
SHIFT = (5, 6)                                                # T moves the image 5 rows down, 6 columns right


# --- the image: the "2" of part1_objects.py, in a 12 x 12 box near the top left corner ------
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


# The same nine points and the same pen as the digit of part1_objects.py: a pixel is as dark
# as the pen covers it, fully within 0.45 pixels of the stroke, not at all beyond 1.25.
stroke = catmull_rom(np.array([[0.18, 0.70], [0.33, 0.90], [0.60, 0.92], [0.76, 0.74], [0.70, 0.52],
                               [0.46, 0.30], [0.20, 0.06], [0.50, 0.10], [0.84, 0.08]]))
corner = 1.0                                                  # the box of the digit starts at pixel (1, 1)
centres = np.stack(np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5), axis=-1).reshape(-1, 2)
pen = np.column_stack([corner + 12 * stroke[:, 0], corner + 12 * (1 - stroke[:, 1])])  # (column, row), row down
distance = np.sqrt(((centres[:, None, :] - pen[None, :, :]) ** 2).sum(-1)).min(1)
x = np.clip((1.25 - distance) / 0.8, 0, 1).reshape(N, N)     # x[i, j]: row i, column j


# --- the two maps, both on the periodic grid --------------------------------------------------
def T(image):
    """The shift (Tx)_{ij} = x_{i-5, j-6}, indices modulo N."""
    return np.roll(image, SHIFT, axis=(0, 1))


k = np.array([[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]], dtype=float)                       # the Laplacian stencil, h = 1


def convolve(kernel, image):
    """(k * x)_{ij} = sum_{a,b} k_{ab} x_{i-a, j-b}, a and b from -1 to 1, indices modulo N."""
    m = kernel.shape[0] // 2
    return sum(kernel[a + m, b + m] * np.roll(image, (a, b), axis=(0, 1))
               for a in range(-m, m + 1) for b in range(-m, m + 1))


kx, kTx = convolve(k, x), convolve(k, T(x))
print(f"x: the digit 2 on a periodic {N} x {N} grid, {(x > 0).sum()} pixels inked; "
      f"T shifts it by {SHIFT[0]} rows and {SHIFT[1]} columns")
print(f"k: the 3 x 3 Laplacian stencil; k * x takes values from {kx.min():.2f} to {kx.max():.2f}")
print(f"equivariance: max |k * (Tx) - T(k * x)| = {np.abs(kTx - T(kx)).max():.1e}")
print(f"invariance:   sum(Tx) = {T(x).sum():.4f} = sum(x) = {x.sum():.4f}; "
      f"sum(relu(k * Tx)) = {np.maximum(kTx, 0).sum():.4f} = sum(relu(k * x)) = {np.maximum(kx, 0).sum():.4f}")
# The panels show the shift as a plain translation: nothing that T moves crosses the edge of the grid.
assert not np.any(kx[N - SHIFT[0]:, :]) and not np.any(kx[:, N - SHIFT[1]:]), "the shift wraps around"


# --- the panels ---------------------------------------------------------------------------
def panel(values, name, cmap, norm):
    """One image as N x N cells, row 0 on top, in a frame, SIDE mm wide."""
    fig = plt.figure(figsize=(SIDE / 25.4, SIDE / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, N), ylim=(N, 0))
    ax.axis("off")
    ax.pcolormesh(np.arange(N + 1), np.arange(N + 1), values, cmap=cmap, norm=norm,
                  edgecolors="face", linewidth=0.05)
    ax.add_patch(Rectangle((0, 0), N, N, fill=False, ec=P.RULE, lw=1.0, clip_on=False))  # half of it shows
    fig.patch.set_alpha(0)
    C.save(fig, name)


plt.rcParams["figure.autolayout"] = False                    # the axes fill the figure
extreme = np.abs(kx).max()
panel(x, "symmetry_x", SEQ, Normalize(0, 1))
panel(T(x), "symmetry_Tx", SEQ, Normalize(0, 1))
panel(kx, "symmetry_kx", DIV, TwoSlopeNorm(0, -extreme, extreme))
panel(kTx, "symmetry_kTx", DIV, TwoSlopeNorm(0, -extreme, extreme))
