"""Appendix, vision transformers: the digits of lectures 02 to 04, read by a transformer.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/appendix_vit.py [preview_dir]

The model (Dosovitskiy et al. 2021, without their MLP head): the 28 x 28 image is cut into 16
patches of 7 x 7 pixels, read row by row; each patch is mapped linearly to R^d (d = 64), a
learned position vector p_t is added, a learned class token is put in front at t = 0, then
L = 4 pre-norm blocks of part 3 without the causal mask (H = 4 heads, an MLP of width 4d), and
a linear read-out of the class token.  7 x 7 patches rather than 4 x 4 (49 tokens): in a pilot
run of one epoch on three threads, 48 s and 5.6% test error with 16 tokens, 124 s and 9.6% with
49 (the same d, L, H; both have 205 066 parameters).

Training is the recipe of lectures 02 to 04 (cross-entropy, Adam with learning rate 1e-3,
batches of 64, initialization seed 0, shuffling seed 1) for 4690 steps, the 5 epochs of the
baseline, whatever the number N of training images (the first N), as in part2_mnist.py and
part5_scale.py of lecture 04.  Only the architecture and N change.  The two reference networks
of lecture 04 are trained again with the same loop, on one thread as there, and reproduce its
numbers: 784-128-128-10 (118 282 parameters, 2.2% test error) and the CNN with 8 and 16
channels (5 994 parameters, 1.3%).

Figures, each drawn at its size on the slide:
- vit_digit.pdf     a test digit cut into its 16 patches, numbered in reading order;
- vit_sequence.pdf  the class token and the 16 patches as a sequence, t = 0, ..., 16;
- vit_data.pdf      test error against N for the three networks, 4690 steps each;
- vit_positions.pdf cosine similarity of the learned position vectors p_1, ..., p_16 of the
                    ViT trained on all 60 000 digits (as Figure 7, centre, of the paper).  On
                    this model it shows no grid: neighbouring patches are not more similar
                    than distant ones (the script prints the averages).  Not on the slides;
- vit_attention.pdf attention of the class token over the patches, first and last block,
                    five test digits.

The runs are cached in data/vit/runs.json and the ViT trained on 60 000 digits in
data/vit/vit_60000.pt (git-ignored); delete them to train again.  Training took 32 minutes on
three CPU threads of a busy shared machine (four ViT runs of 4 to 9 minutes, eight shorter ones
on one thread); with the cache, the script draws the figures in ten seconds.
"""
import json
import time

import common as C
from common import P, np, plt, torch, nn
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

torch.set_num_threads(3)
start = time.time()
RUNS = C.DATA / "vit" / "runs.json"
CHECKPOINT = C.DATA / "vit" / "vit_60000.pt"
RUNS.parent.mkdir(parents=True, exist_ok=True)

train_images, train_labels = C.load_mnist("train")
test_images, test_labels = C.load_mnist("test")
STEPS = 4690                                 # the 5 epochs of the baseline of lecture 02, for every run
SIZES = (1000, 4000, 16_000, 60_000)         # N: the first N training digits

# The colour scales of lectures 03 and 04: ink in blue, signed values blue (negative) to orange.
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9ddf5", P.BLUE, "#0f3a73"])
DIV = LinearSegmentedColormap.from_list("div", ["#0f3a73", P.BLUE, "#ffffff", P.ORANGE, "#8f3410"])


# ---------------------------------------------------------------------------------------
# 1. The three architectures
# ---------------------------------------------------------------------------------------
class Block(nn.Module):
    """x <- x + Attention(LN(x)), then x <- x + MLP(LN(x)): the block of part 3 (gpt2.Block), line
    for line, without the two lines of the causal mask.  Copied rather than imported, so that this
    script does not depend on gpt2.py."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)                  # Q, K and V of all heads, stacked
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def weights(self, x):                               # a_ts of every head: (batch, heads, n, n)
        B, n, d = x.shape
        q, k, v = (u.view(B, n, self.heads, -1).transpose(1, 2) for u in self.qkv(x).split(d, dim=-1))
        return (q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5).softmax(-1), v    # no mask: every patch sees every patch

    def attention(self, x):                             # x: (batch, n, d)
        a, v = self.weights(x)
        B, n, d = x.shape
        return self.out((a @ v).transpose(1, 2).reshape(B, n, d))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# The class below is the listing of the slide "A vision transformer on the digits", line for line.
class ViT(nn.Module):
    def __init__(self, d=64, L=4, H=4):
        super().__init__()
        self.E = nn.Linear(49, d)                          # a 7x7 patch -> R^d
        self.cls = nn.Parameter(torch.zeros(1, 1, d))      # the class token
        self.p = nn.Parameter(0.02 * torch.randn(17, d))   # p_0, ..., p_16
        self.blocks = nn.Sequential(*[Block(d, H) for _ in range(L)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 10))

    def forward(self, img):                                # img: (B, 784)
        x = img.view(-1, 28, 28).unfold(1, 7, 7).unfold(2, 7, 7)
        x = self.E(x.reshape(-1, 16, 49))                  # 16 patches -> R^d
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.p
        return self.out(self.blocks(x)[:, 0])              # read out x_0


def vit(seed=0):
    torch.manual_seed(seed)
    return ViT()


def cnn(seed=0, c=8):
    """The convolutional network of lecture 04 (figures/part2_mnist.py, c = 8): 5 994 parameters."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        nn.Conv2d(1, c, 5), nn.ReLU(), nn.MaxPool2d(2),             # (c, 24, 24) -> (c, 12, 12)
        nn.Conv2d(c, 2 * c, 5), nn.ReLU(), nn.MaxPool2d(2),         # (2c, 8, 8) -> (2c, 4, 4)
        nn.Flatten(), nn.Linear(2 * c * 16, 10))


def mlp(seed=0, width=128):
    """784 -> 128 -> 128 -> 10, as common.make_classifier of lecture 04: 118 282 parameters."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 10))


MODELS = {"vit": (vit, 3), "cnn": (cnn, 1), "mlp": (mlp, 1)}   # constructor, threads (1: as in lecture 04)


# ---------------------------------------------------------------------------------------
# 2. Training: the loop of lecture 02, a fixed number of steps
# ---------------------------------------------------------------------------------------
def error(net, images, labels, chunk=1000):
    """Test error in percent, in chunks so that the attention of 1000 images fits in memory."""
    net.eval()
    with torch.no_grad():
        hits = sum((net(images[i:i + chunk]).argmax(1) == labels[i:i + chunk]).sum().item()
                   for i in range(0, len(images), chunk))
    net.train()
    return 100 * (1 - hits / len(images))


def train(net, n_train):
    """STEPS steps of Adam (learning rate 1e-3, batches of 64) on the first n_train digits."""
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    cross_entropy = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(1)
    images, labels = train_images[:n_train], train_labels[:n_train]
    step = 0
    while step < STEPS:                                    # as many passes through the N digits as needed
        for batch in torch.randperm(n_train, generator=generator).split(64):
            optimizer.zero_grad()
            cross_entropy(net(images[batch]), labels[batch]).backward()
            optimizer.step()
            step += 1
            if step == STEPS:
                break
    return net


runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
# The ViT on all the digits first (the main result, and the model of the figures), then the
# two fast reference networks, then the ViT on fewer digits.
order = [("vit", 60_000)] + [(m, n) for m in ("cnn", "mlp") for n in SIZES[::-1]] + [("vit", n) for n in SIZES[-2::-1]]
for name, n_train in order:
    key = f"{name}/{n_train}"
    if key in runs and (key != "vit/60000" or CHECKPOINT.exists()):
        continue
    make, threads = MODELS[name]
    torch.set_num_threads(threads)
    t0 = time.time()
    net = train(make(), n_train)
    runs[key] = dict(parameters=C.count_parameters(net), test_error=error(net, test_images, test_labels),
                     train_error=error(net, train_images[:n_train], train_labels[:n_train]),
                     seconds=round(time.time() - t0), threads=threads)
    if key == "vit/60000":
        torch.save(net.state_dict(), CHECKPOINT)
    RUNS.write_text(json.dumps(runs, indent=1))            # saved as we go
    print(f"  trained {key}: {runs[key]}", flush=True)
torch.set_num_threads(3)

print(f"{STEPS} steps of Adam (learning rate 1e-3, batches of 64) on the first N training digits; "
      "test error on the 10 000 test digits | training error:")
for name in ("mlp", "cnn", "vit"):
    for n_train in SIZES:
        r = runs[f"{name}/{n_train}"]
        print(f"  {name}, {r['parameters']:6d} parameters, N = {n_train:5d}: test error {r['test_error']:5.2f}% | "
              f"training error {r['train_error']:5.2f}%  ({r['seconds']} s on {r['threads']} thread(s))")
for n_train in SIZES:
    ratio = runs[f"vit/{n_train}"]["test_error"] / runs[f"cnn/{n_train}"]["test_error"]
    print(f"  N = {n_train:5d}: the ViT's test error is {ratio:.1f} times the CNN's")
assert runs["mlp/60000"]["parameters"] == 118_282 and runs["cnn/60000"]["parameters"] == 5_994
print(f"lecture 04 quotes 2.2% (784-128-128-10) and 1.3% (CNN); here {runs['mlp/60000']['test_error']:.2f}% and "
      f"{runs['cnn/60000']['test_error']:.2f}%")

# The listing and its printed output, as on the slide.
net = ViT()
net.load_state_dict(torch.load(CHECKPOINT))
net.eval()
print("slide output: > " + f"{sum(p.numel() for p in net.parameters())} parameters, "
      f"test error {error(net, test_images, test_labels):.2f}%")


# ---------------------------------------------------------------------------------------
# 3. The picture of the first slide: a digit, its 16 patches, the sequence
# ---------------------------------------------------------------------------------------
plt.rcParams["figure.autolayout"] = False                  # the axes are placed by hand, in mm
DIGIT = 1                                                  # the second test digit, a 2
image = test_images[DIGIT].view(28, 28).numpy()
print(f"the digit of the first slide: test image {DIGIT}, label {test_labels[DIGIT].item()}")


def mm_axes(fig, width, height, left, bottom, w, h):
    """Axes at (left, bottom) with size (w, h), all in mm, in a figure of width x height mm."""
    ax = fig.add_axes((left / width, bottom / height, w / width, h / height))
    ax.axis("off")
    return ax


def draw_pixels(ax, pixels):
    """Pixels in [0, 1] as blue ink on white, row 0 on top, framed as in lecture 04."""
    n = pixels.shape[0]
    ax.set(xlim=(0, n), ylim=(n, 0))
    ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), pixels, cmap=SEQ, norm=Normalize(0, 1),
                  edgecolors="face", linewidth=0.05)
    ax.add_patch(Rectangle((0, 0), n, n, fill=False, ec=P.RULE, lw=0.8, clip_on=False))


SIDE = 34.0                                                # the digit, in mm
fig = plt.figure(figsize=(SIDE * C.MM, SIDE * C.MM))
ax = mm_axes(fig, SIDE, SIDE, 0.4, 0.4, SIDE - 0.8, SIDE - 0.8)
draw_pixels(ax, image)
for i in range(1, 4):                                      # the cuts between the patches
    ax.axhline(7 * i, color=P.NOTE, lw=0.6)
    ax.axvline(7 * i, color=P.NOTE, lw=0.6)
for t in range(16):                                        # reading order: row by row; a white box keeps the ink apart
    ax.text(7 * (t % 4) + 0.45, 7 * (t // 4) + 0.35, str(t + 1), ha="left", va="top", fontsize=6, color=P.NOTE,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
C.save(fig, "vit_digit")

WIDTH, HEIGHT, SLOT, BOX = 108.0, 11.0, 108.0 / 17, 5.4   # 17 slots: the class token and 16 patches
fig = plt.figure(figsize=(WIDTH * C.MM, HEIGHT * C.MM))
for t in range(17):
    ax = mm_axes(fig, WIDTH, HEIGHT, t * SLOT + (SLOT - BOX) / 2, HEIGHT - BOX - 0.4, BOX, BOX)
    if t == 0:                                             # the class token: a learned vector, no pixels
        ax.set(xlim=(0, 1), ylim=(1, 0))
        ax.add_patch(Rectangle((0, 0), 1, 1, fc="#fbe1d6", ec=P.ORANGE, lw=0.8, clip_on=False))
        ax.text(0.5, 0.5, "cls", ha="center", va="center", fontsize=6.5, color=P.ORANGE)
    else:
        r, c = divmod(t - 1, 4)
        draw_pixels(ax, image[7 * r:7 * r + 7, 7 * c:7 * c + 7])
    fig.text((t + 0.5) * SLOT / WIDTH, 0.5 / HEIGHT, f"{t}", ha="center", va="bottom", fontsize=7, color=P.NOTE)
C.save(fig, "vit_sequence")


# ---------------------------------------------------------------------------------------
# 4. Test error against the number of training digits
# ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(54 * C.MM, 48 * C.MM), layout="constrained")
for name, label, color in (("mlp", "fully connected", P.GREY), ("cnn", "convolutional", P.BLUE),
                           ("vit", "vision transformer", P.ORANGE)):
    errors = [runs[f"{name}/{n}"]["test_error"] for n in SIZES]
    ax.loglog(SIZES, errors, "-o", ms=3, lw=1.2, color=color, label=label)
P.label(ax, "training digits $N$", "test error (%)", legend=dict(fontsize=6.5, loc="upper right"),
        xlim=(800, 75_000), ylim=(1, 30))
ax.set_xticks(SIZES, ["1k", "4k", "16k", "60k"])
ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
ax.minorticks_off()
ax.tick_params(labelsize=7)
ax.xaxis.label.set_size(8)
ax.yaxis.label.set_size(8)
C.save(fig, "vit_data")


# ---------------------------------------------------------------------------------------
# 5. What the model learned about space: similarity of the position vectors
# ---------------------------------------------------------------------------------------
p = net.p.detach()[1:]                                     # p_1, ..., p_16: the patches, not the class token
p = p / p.norm(dim=1, keepdim=True)
similarity = (p @ p.T).numpy()                             # cosine similarity, 16 x 16
grid = np.array([[similarity[t].reshape(4, 4)] for t in range(16)]).reshape(4, 4, 4, 4)
others = similarity[~np.eye(16, dtype=bool)].reshape(16, 15)
rows, cols = np.divmod(np.arange(16), 4)
distance = np.hypot(rows[:, None] - rows[None, :], cols[:, None] - cols[None, :])
neighbours, far = similarity[distance == 1].mean(), similarity[distance >= 3].mean()
print(f"position vectors: cosine similarity with the 4 neighbours (distance 1) {neighbours:.2f} on average, "
      f"with patches at distance >= 3: {far:.2f}; range off the diagonal {others.min():.2f} to {others.max():.2f}")
print(f"  correlation of the cosine similarity with the distance between the patches: "
      f"{np.corrcoef(similarity[distance > 0], distance[distance > 0])[0, 1]:.2f}")
at_init = vit().p.detach()[1:]
at_init = at_init / at_init.norm(dim=1, keepdim=True)
print(f"  at initialization: off-diagonal cosine similarity {(at_init @ at_init.T)[~np.eye(16, dtype=bool)].abs().mean():.2f} "
      "in absolute value on average")

# 4 x 4 tiles of 9 mm: tile (r, c) shows the similarity of p_t, t the patch in row r and column c
# (framed), with the 16 position vectors, laid out on the grid of the patches; a colour bar on the right.
W, H, side, gap, left, top = 56.0, 48.0, 9.0, 1.1, 7.5, 1.0
bottom = H - top - 4 * side - 3 * gap
norm = TwoSlopeNorm(0, -1, 1)
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
for r in range(4):
    for c in range(4):
        ax = mm_axes(fig, W, H, left + c * (side + gap), H - top - (r + 1) * side - r * gap, side, side)
        ax.pcolormesh(np.arange(5), np.arange(5), grid[r, c], cmap=DIV, norm=norm)
        ax.set(xlim=(0, 4), ylim=(4, 0))
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, ec=P.INK, lw=0.6))      # the patch itself
for r in range(4):
    fig.text((left - 1.0) / W, (H - top - r * (side + gap) - side / 2) / H, str(r + 1), ha="right", va="center",
             fontsize=7, color=P.NOTE)
    fig.text((left + r * (side + gap) + side / 2) / W, (bottom - 1.0) / H, str(r + 1), ha="center", va="top",
             fontsize=7, color=P.NOTE)
fig.text(1.0 / W, (bottom + 2 * side + 1.5 * gap) / H, "row", rotation=90, ha="left", va="center", fontsize=8)
fig.text((left + 2 * side + 1.5 * gap) / W, 0.3 / H, "column", ha="center", va="bottom", fontsize=8)
cax = fig.add_axes(((left + 4 * side + 3 * gap + 2.5) / W, bottom / H, 1.6 / W, (4 * side + 3 * gap) / H))
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax)
bar.set_ticks([-1, 0, 1], labels=["$-1$", "0", "1"])
bar.ax.tick_params(labelsize=7, length=2, color=P.RULE)
bar.outline.set_edgecolor(P.RULE)
C.save(fig, "vit_positions")


# ---------------------------------------------------------------------------------------
# 6. Where the class token looks: its attention over the patches in the last block
# ---------------------------------------------------------------------------------------
def class_attention(net, images):
    """a_{0s}, s = 1..16, of the class token in every block, averaged over the heads: (B, L, 4, 4)."""
    with torch.no_grad():
        x = net.E(images.view(-1, 28, 28).unfold(1, 7, 7).unfold(2, 7, 7).reshape(-1, 16, 49))
        x = torch.cat([net.cls.expand(len(x), -1, -1), x], 1) + net.p
        maps = []
        for block in net.blocks:
            maps.append(block.weights(block.ln1(x))[0].mean(1)[:, 0, 1:])
            x = block(x)
    return torch.stack(maps, 1).reshape(len(images), -1, 4, 4)


shown = [1, 2, 3, 4, 7]                                    # a 2, a 1, a 0, a 4 and a 9
attention = class_attention(net, test_images[shown])
everything = class_attention(net, test_images[:1000])
ink = test_images[:1000].view(-1, 28, 28).unfold(1, 7, 7).unfold(2, 7, 7).sum((-1, -2))   # ink per patch
for b in range(attention.shape[1]):
    a = everything[:, b]
    corr = np.corrcoef(a.reshape(-1).numpy(), ink.reshape(-1).numpy())[0, 1]
    print(f"block {b + 1}: class token, over 1000 test digits: largest weight on one patch {a.amax((1, 2)).mean():.2f} "
          f"on average (uniform over the 17 tokens: {1 / 17:.2f}), on itself {1 - a.sum((1, 2)).mean():.2f}; "
          f"correlation of the weight of a patch with its ink {corr:.2f}")

GREYS = LinearSegmentedColormap.from_list("greys", ["#ffffff", P.NOTE])
W, H, side, gap = 108.0, 37.0, 16.0, 3.0                  # two rows: the first and the last block
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
for k, i in enumerate(shown):
    for row, b in enumerate((0, attention.shape[1] - 1)):
        ax = mm_axes(fig, W, H, 12.0 + k * (side + gap), H - 1.0 - (row + 1) * side - row * 2.5, side, side)
        ax.set(xlim=(0, 28), ylim=(28, 0))
        ax.pcolormesh(np.arange(29), np.arange(29), test_images[i].view(28, 28).numpy(), cmap=GREYS,
                      norm=Normalize(0, 1), edgecolors="face", linewidth=0.05)
        a = attention[k, b].numpy()
        for r in range(4):
            for c in range(4):                             # the weight of a patch, relative to the largest
                ax.add_patch(Rectangle((7 * c, 7 * r), 7, 7, fc=P.ORANGE, ec="none", alpha=0.8 * a[r, c] / a.max()))
        ax.add_patch(Rectangle((0, 0), 28, 28, fill=False, ec=P.RULE, lw=0.8, clip_on=False))
        if k == 0:
            fig.text(10.5 / W, (H - 1.0 - row * (side + 2.5) - side / 2) / H, f"block {b + 1}", ha="right",
                     va="center", fontsize=7, color=P.NOTE)
C.save(fig, "vit_attention")
print(f"run time: {time.time() - start:.0f} s")
