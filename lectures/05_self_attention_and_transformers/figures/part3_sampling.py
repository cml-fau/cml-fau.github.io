"""Part 3 (the transformer), the frame "Sampling: from probabilities to text": what GPT-2 small,
with the weights of OpenAI, does with the scores z it returns for the next token.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part3_sampling.py [preview_dir]

The model (figures/gpt2.py) returns, for the last token of the prompt, one score z_i per token
i of the vocabulary, 50 257 numbers; the next token is drawn from softmax(z / tau), and tau = 0
means the most likely token (greedy decoding).

- sampling_temperature.pdf: for the prompt "Attention is all you", the probabilities of the ten
  most likely next tokens at tau = 0.5, 1 and 2, one panel per temperature, and in grey the
  total probability of the other 50 247 tokens.  Above each panel the entropy
  -sum_i p_i log p_i in nats (not called H on the slides: H is the number of heads).  The order
  of the tokens does not depend on tau (dividing by tau > 0 keeps the order of the scores).

Printed, for the slide and the notes: the same numbers for every prompt that was tried (the
prompt of the figure was chosen among them for a distribution that is neither a single token
nor flat), and continuations of 28 tokens of one prompt at tau = 0 (greedy), 0.7, 1 and 1.5,
five seeds each (torch.Generator seeds 0 to 4), and at tau = 1 with top_k = 40.  The
continuations are printed with repr(), so that a line break shows as \\n.

The model is loaded once (about 0.5 GB of weights); two CPU threads.  About three minutes, nearly
all of it the 21 continuations: gpt2.generate runs the whole model again for every new token.
"""
import common as C
from common import P, np, plt, torch
import gpt2

torch.set_num_threads(2)
plt.rcParams["figure.autolayout"] = False

model = gpt2.pretrained()

PROMPTS = ["Attention is all you", "The capital of Germany is", "The University of Erlangen-Nuremberg is",
           "The best way to learn mathematics is", "Once upon a time, there was a", "My favourite food is",
           "The theorem states that", "The weather today is", "A neural network is", "The square root of two is",
           "Mathematics is", "The students in Erlangen", "Today I"]
FIGURE_PROMPT = PROMPTS[0]
TEMPERATURES = [0.5, 1.0, 2.0]
TOP = 10


@torch.inference_mode()
def scores(prompt):
    """z: the scores of the next token after the prompt, one per token of the vocabulary."""
    return model(torch.tensor([gpt2.encode(prompt)]))[0, -1]


def entropy(p):
    return -(p * p.clamp_min(1e-30).log()).sum().item()       # nats; 0 log 0 = 0


# ---------------------------------------------------------------------------------------
# 1. The law of the next token at three temperatures, for every prompt tried
# ---------------------------------------------------------------------------------------
print(f"vocabulary: {model.tokens.num_embeddings} tokens; log(50257) = {np.log(50257):.2f} nats is the entropy of the uniform law")
for prompt in PROMPTS:
    z = scores(prompt)
    print(f"\n{prompt!r}: z from {z.min():.1f} to {z.max():.1f}")
    for tau in TEMPERATURES:
        p = (z / tau).softmax(-1)
        top = p.topk(TOP)
        print(f"  tau = {tau}: entropy {entropy(p):.2f} nats, top {TOP} hold {top.values.sum():.3f}: "
              + ", ".join(f"{gpt2.decode([i])!r} {q:.3f}" for q, i in zip(top.values.tolist(), top.indices.tolist())))

# ---------------------------------------------------------------------------------------
# 2. The figure: three panels, the same ten tokens, the rest in grey
# ---------------------------------------------------------------------------------------
z = scores(FIGURE_PROMPT)
order = z.topk(TOP).indices                                   # the same ten tokens at every tau > 0
# A token is drawn as GPT-2 stores it, with its leading space shown as a visible space.
labels = [gpt2.decode([i]).replace(" ", "␣") for i in order.tolist()] + [f"other {len(z) - TOP:,}".replace(",", " ")]
fig, axes = plt.subplots(1, 3, figsize=(108 * C.MM, 36 * C.MM), sharey=True)
fig.subplots_adjust(left=0.16, right=0.955, bottom=0.19, top=0.865, wspace=0.2)
y = np.arange(TOP + 1)
for ax, tau in zip(axes, TEMPERATURES):
    p = (z / tau).softmax(-1)
    values = np.append(p[order].numpy(), 1 - p[order].sum().item())
    colors = [P.BLUE] * TOP + [P.GREY]
    ax.barh(y, values, height=0.72, color=colors, lw=0)
    for k in (0, TOP):                                        # the largest and the rest: the two numbers to read
        label = f"{values[k]:.2f}" if values[k] >= 0.01 else f"{values[k]:.3f}"
        ax.text(values[k] + 0.03, y[k], label, va="center", ha="left", fontsize=6.5, color=P.NOTE, clip_on=False)
    ax.set_title(rf"$\tau={tau:g}$, entropy ${entropy(p):.2f}$", fontsize=7.5, pad=3)   # H is the number of heads
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
axes[0].set_yticks(y, labels, fontsize=6.5, family="monospace")
axes[0].invert_yaxis()
axes[0].get_yticklabels()[-1].set(family="sans-serif", color=P.NOTE, fontsize=6.5)
fig.supxlabel(r"probability of the next token", fontsize=7, y=0.0)
C.save(fig, "sampling_temperature")
print(f"\nfigure: {FIGURE_PROMPT!r}, the ten tokens " + ", ".join(repr(gpt2.decode([i])) for i in order.tolist()))

# ---------------------------------------------------------------------------------------
# 3. Continuations of one prompt
# ---------------------------------------------------------------------------------------
STEPS = 28
for prompt in ["The best way to learn mathematics is"]:
    ids = gpt2.encode(prompt)
    print(f"\ncontinuations of {prompt!r} ({len(ids)} tokens), {STEPS} new tokens each:")
    greedy = gpt2.generate(model, ids, STEPS, temperature=0)
    print(f"  tau = 0  (greedy):   {gpt2.decode(greedy[len(ids):])!r}")
    for tau, top_k in ((0.7, None), (1.0, None), (1.5, None), (1.0, 40)):
        for seed in range(5):
            out = gpt2.generate(model, ids, STEPS, temperature=tau, top_k=top_k,
                                generator=torch.Generator().manual_seed(seed))
            text = gpt2.decode(out[len(ids):])
            flag = "" if text.isascii() else "   [not ASCII]"
            print(f"  tau = {tau}{', top_k = 40' if top_k else ''}, seed {seed}: {text!r}{flag}")
