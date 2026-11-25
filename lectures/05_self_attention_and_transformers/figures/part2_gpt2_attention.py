"""Part 2 (attention), the frames "Self-attention" and "Multi-head attention": the weights a_ts of
real heads of GPT-2 small, with the weights of OpenAI.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part2_gpt2_attention.py [preview_dir]

Heads are named layer.head, both counted from 0 (layers 0..11, heads 0..11), as in Wang et al.
(2023), whose head 0.1 is in the first layer and 11.10 in the last.  The weights come from
gpt2.attention_weights, the same lines as Block.attention.

- gpt2_attention_heads.pdf ("Multi-head attention"): the matrix (a_ts) of four heads on the
  nine tokens of "The cat sat on the mat. The cat", row t = the token that attends, column
  s = the token attended to; only s <= t is drawn (the rest is masked).  The heads were chosen
  by looking at all 144 (the scores printed below rank them) for four clean, different patterns:
      4.11  all its weight on the previous token,
      1.10  spread evenly over the tokens up to t, the first one left out,
      0.1   on the token itself, and on the earlier copy of the same token (" cat"),
      5.5   on the first token, except the second " cat", which looks at " sat",
            the token after the first " cat".
  The cells quoted on the slide are outlined in orange.
- gpt2_attention_one.pdf ("Self-attention"): one row of weights, a_ns for the last token
  " to" of "When Mary and John went to the store, John gave a drink to", the example of
  Wang et al. (2023), for head 9.9, one of their "name mover" heads.  The text is preceded by
  the token <|endoftext|> that GPT-2 saw between documents (without it, the first word takes
  that weight: both are printed).  The bars are the weights, the numbers the largest ones.

Printed: for every head the mean weight on the previous token, on the first token and on
itself; the four matrices; for the sentence of Wang et al. the weights from " to" of their
name mover heads 9.9, 9.6, 10.0 (and of the other heads they name), the heads that put the most
weight on " Mary", and the probabilities of " Mary" and " John" as the next token, with and
without <|endoftext|>.  The model is loaded once; two CPU threads; a few seconds.
"""
import common as C
from common import P, np, plt, torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import gpt2

torch.set_num_threads(2)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)
plt.rcParams["figure.autolayout"] = False
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9ddf5", P.BLUE, "#0f3a73"])

model = gpt2.pretrained()


def weights(ids):
    """(layers, heads, n, n): a[l, h, t, s] is the weight of token s in the update of token t."""
    return gpt2.attention_weights(model, torch.tensor([ids]))[0]


@torch.inference_mode()
def next_token(ids):
    return model(torch.tensor([ids]))[0, -1].softmax(-1)


def rank_heads(a):
    """Mean weight, over the rows t >= 1, on the previous token, on the first token, on itself."""
    n = a.shape[-1]
    t = torch.arange(1, n)
    scores = {"previous token": a[:, :, t, t - 1].mean(-1), "first token": a[:, :, 1:, 0].mean(-1),
              "itself": a.diagonal(dim1=-2, dim2=-1)[:, :, 1:].mean(-1)}
    for name, s in scores.items():
        top = s.flatten().topk(6)
        print(f"  {name:15s} " + ", ".join(f"{i // 12}.{i % 12} {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist())))


# ---------------------------------------------------------------------------------------
# 1. Four heads on one short sentence
# ---------------------------------------------------------------------------------------
SENTENCE = "The cat sat on the mat. The cat"
HEADS = [(4, 11), (1, 10), (0, 1), (5, 5)]
ids = gpt2.encode(SENTENCE)
toks = [gpt2.decode([i]) for i in ids]
a = weights(ids)
n = len(ids)
print(f"{SENTENCE!r}: {n} tokens {toks}")
print("all 144 heads, mean weight over the rows t = 1..n-1 (best six):")
rank_heads(a)
for l, h in HEADS:
    print(f"head {l}.{h}:\n{a[l, h]}")
cat1, cat2, sat = 1, 8, 2
assert toks[cat1] == toks[cat2] == " cat" and toks[sat] == " sat"
quoted = {(0, 1): [(cat2, cat1)], (5, 5): [(cat2, sat)]}           # (t, s): the cells outlined in orange
print(f"quoted: 4.11 mean weight on the previous token {a[4, 11, torch.arange(1, n), torch.arange(n - 1)].mean():.3f} "
      f"(smallest {a[4, 11, torch.arange(1, n), torch.arange(n - 1)].min():.3f}); "
      f"0.1 from the second ' cat': {a[0, 1, cat2, cat1]:.2f} on the first ' cat', {a[0, 1, cat2, cat2]:.2f} on itself; "
      f"5.5 from the second ' cat': {a[5, 5, cat2, sat]:.2f} on ' sat', {a[5, 5, cat2, 0]:.2f} on 'The'; "
      f"5.5 rows t = 0..7 on the first token: min {a[5, 5, :cat2, 0].min():.2f}; "
      f"1.10 last row: from {a[1, 10, -1, 1:].min():.2f} to {a[1, 10, -1, 1:].max():.2f} on s = 1..8, {a[1, 10, -1, 0]:.3f} on s = 0")
p = next_token(ids)
top = p.topk(5)
print("next token after the sentence: " + ", ".join(f"{gpt2.decode([i])!r} {q:.3f}" for q, i in zip(top.values.tolist(), top.indices.tolist())))

fig, axes = plt.subplots(1, len(HEADS), figsize=(108 * C.MM, 37 * C.MM))
fig.subplots_adjust(left=0.075, right=0.925, bottom=0.25, top=0.88, wspace=0.12)
labels = [s.strip() for s in toks]
lower = np.tril(np.ones((n, n), dtype=bool))
for ax, (l, h) in zip(axes, HEADS):
    values = np.ma.masked_where(~lower, a[l, h].numpy())       # the masked half stays empty
    mesh = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, values, cmap=SEQ, norm=Normalize(0, 1),
                  edgecolors="white", linewidth=0.3)
    # the outline of the lower triangle: the cells that may carry weight
    edge = [(-0.5, -0.5)] + [(k + 0.5, y) for k in range(n) for y in (k - 0.5, k + 0.5)] + [(-0.5, n - 0.5)]
    ax.add_patch(plt.Polygon(edge, closed=True, fill=False, ec=P.RULE, lw=0.5))
    for t, s in quoted.get((l, h), []):
        ax.add_patch(Rectangle((s - 0.5, t - 0.5), 1, 1, fill=False, ec=P.ORANGE, lw=1.1))
    ax.set(xlim=(-0.6, n - 0.4), ylim=(n - 0.4, -0.6), aspect="equal")
    ax.set_title(f"head {l}.{h}", fontsize=7.5, pad=2)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n), labels if ax is axes[0] else [], fontsize=6)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
bar = fig.add_axes((0.945, 0.25, 0.012, 0.55))
colorbar = fig.colorbar(mesh, cax=bar, ticks=[0, 0.5, 1])
colorbar.ax.set_yticklabels(["0", "0.5", "1"])
colorbar.outline.set_visible(False)
bar.tick_params(labelsize=6, length=1.5, pad=1)
bar.set_title("$a_{ts}$", fontsize=7, pad=6)
C.save(fig, "gpt2_attention_heads")

# ---------------------------------------------------------------------------------------
# 2. The sentence of Wang et al. (2023): where the name mover heads look from the last token
# ---------------------------------------------------------------------------------------
IOI = "When Mary and John went to the store, John gave a drink to"
NAMED = {"name mover": [(9, 9), (9, 6), (10, 0)], "negative name mover": [(10, 7), (11, 10)],
         "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)]}
results = {}
for bos in (True, False):
    ids = ([gpt2.END_OF_TEXT] if bos else []) + gpt2.encode(IOI)
    toks = [gpt2.decode([i]) for i in ids]
    a = weights(ids)
    end, mary = len(ids) - 1, toks.index(" Mary")
    john1, john2 = [k for k, s in enumerate(toks) if s == " John"]
    p = next_token(ids)
    top = p.topk(5)
    print(f"\n{IOI!r}, {'with' if bos else 'without'} <|endoftext|> first: {len(ids)} tokens")
    print("  next token: " + ", ".join(f"{gpt2.decode([i])!r} {q:.3f}" for q, i in zip(top.values.tolist(), top.indices.tolist())))
    pm, pj = p[gpt2.encode(" Mary")[0]].item(), p[gpt2.encode(" John")[0]].item()
    print(f"  p(' Mary') = {pm:.3f}, p(' John') = {pj:.3f}, log(p_Mary / p_John) = {np.log(pm / pj):.2f}")
    for family, heads in NAMED.items():
        for l, h in heads:
            row = a[l, h, end]
            print(f"  {family:20s} {l}.{h} from ' to': ' Mary' {row[mary]:.3f}, ' John' (first) {row[john1]:.3f}, "
                  f"' John' (second) {row[john2]:.3f}, first token {row[0]:.3f}; largest on {toks[row.argmax()]!r}")
    best = a[:, :, end, mary].flatten().topk(6)
    print("  most weight from ' to' on ' Mary': " + ", ".join(f"{i // 12}.{i % 12} {v:.3f}" for v, i in zip(best.values.tolist(), best.indices.tolist())))
    print(f"  duplicate token heads (Wang et al.: 0.1, 3.0) from the second ' John' on the first: "
          f"0.1 {a[0, 1, john2, john1]:.3f}, 3.0 {a[3, 0, john2, john1]:.3f}")
    results[bos] = (toks, a[9, 9, end].numpy(), pm, pj)

# The figure: the sentence, and above each token a bar as high as its weight in head 9.9.
toks, row, pm, pj = results[True]
fig = plt.figure(figsize=(108 * C.MM, 21 * C.MM))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, 108), ylim=(0, 21))
ax.axis("off")
renderer = fig.canvas.get_renderer()
x, gap, base = 0.5, 1.3, 3.0                                     # mm
for k, (tok, weight) in enumerate(zip(toks, row)):
    special = tok == "<|endoftext|>"
    color = P.ORANGE if k == len(toks) - 1 else (P.NOTE if special else P.INK)
    text = ax.text(x, base, tok.strip(), fontsize=6.5 if special else 7.5, color=color, ha="left", va="baseline",
                   fontweight="bold" if k == len(toks) - 1 else "normal")
    width = text.get_window_extent(renderer).width / fig.dpi * 25.4
    centre = x + width / 2
    height = 13 * weight                                          # 13 mm for a weight of 1
    ax.add_patch(Rectangle((centre - 1.4, base + 2.6), 2.8, max(height, 0.15), color=P.BLUE, lw=0))
    if weight >= 0.05:
        ax.text(centre, base + 2.6 + height + 0.6, f"{weight:.2f}", fontsize=6.5, color=P.NOTE, ha="center", va="bottom")
    x += width + gap
print(f"\nfigure gpt2_attention_one: the sentence takes {x - gap:.1f} mm of 108")
assert x - gap < 107.5, "the sentence does not fit"
C.save(fig, "gpt2_attention_one")
