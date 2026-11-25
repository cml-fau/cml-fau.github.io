"""Part 4 (training), a small GPT trained from scratch on one megabyte of Shakespeare: scratch_loss.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part4_scratch.py [preview_dir] [--train] [--evaluate]

The model is the class GPT of gpt2.py, the code of the slides, with small numbers: a vocabulary
of the 65 characters of the text (one token per character, no byte-pair encoding), context
n = 128, width d = 128, L = 4 blocks of H = 4 heads: 818 048 parameters, initialized as GPT-2
(N(0, 0.02^2), biases 0).  The text is tinyshakespeare.txt (Karpathy's char-rnn data set, 1 115 394
characters): the first 90% for training, the last 10% for validation.

The training loop is the one of lectures 02-04, unchanged: mini-batches of B = 16 windows of
n + 1 characters drawn at random places of the training text (the first n characters are the
input, the last n the targets), the cross-entropy -log p_w(i_(t+1) | i_1..i_t) averaged over the
B n positions, AdamW with the step eta = 1e-3 of the baseline of lecture 02 and PyTorch's other
defaults (betas 0.9, 0.999, weight decay 0.01), no schedule, no clipping, no dropout; 2500 steps.
Seeds: 0 for the initial weights, 1 for the mini-batches.  At 14 steps from 0 to 2500 the weights
are kept, and the model writes 120 characters after two prompts (a newline, and "ROMEO:"), at
temperature 1 and with the generators of seeds 0-3: the same random numbers at every step, so
that only the model changes.  While it trains, the run monitors the loss on 64 fixed windows
(8192 positions) of each text; these small sets are off by up to 0.06 (at the end they put the
gap between training and validation at 0.29 instead of 0.185), so the curves of the figure are
computed afterwards from the kept weights: the loss on the whole validation text (864 windows of n + 1
characters, 110 592 targets) and on 864 fixed random windows of the training text (seed 3).

- scratch_loss.pdf: the loss against the step (symlog axis: linear up to 10, logarithmic
  after), training (blue) and validation (orange) at the 14 kept steps and at steps 5, 10, 15, 20
  (recomputed by running the first 20 steps again: replay()), with three reference
  levels computed here: log 65 (the uniform law), the character frequencies of the training text
  (a model that ignores the context) and a bigram model (counts of the pairs of the training
  text plus one), both evaluated on the validation text; dots at the steps whose samples the
  slide shows (MARKED).  Drawn at 108 x 34 mm.

Printed: the reference levels, the run (steps, tokens, wall-clock time, losses), the steps
between which the validation loss passes each reference level, all the samples, the longest run
of characters of the final samples that occurs verbatim in the training text (next to the same
for excerpts of the validation text), and the listing of the slide (SHOWN) with its alternatives.

The training took 22 minutes on 3 threads of a shared machine (18 of them in the 2500 steps,
0.44 s per step; samples and monitoring the rest); its results are written to
data/scratch/results.json as it goes, the kept weights to data/scratch/checkpoints.pt, and both
are reused unless --train is given.  The curves from the kept weights take about five minutes
more (again with --evaluate); the figure and the printout alone take ten seconds.
"""
import json
import sys
import time

import common as C
from common import P, np, plt, torch, nn
import gpt2

torch.set_num_threads(3)

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = C.DATA / "scratch"
RESULTS, CHECKPOINTS = OUT / "results.json", OUT / "checkpoints.pt"

# ---------------------------------------------------------------------------------------
# The text: one token per character, the first 90% for training, the last 10% for validation
# ---------------------------------------------------------------------------------------
text = gpt2.fetch("tinyshakespeare.txt", URL).read_text()
chars = sorted(set(text))                              # the vocabulary: 65 characters
index = {c: i for i, c in enumerate(chars)}
encode = lambda s: [index[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
data = torch.tensor(encode(text))
split = int(0.9 * len(data))
train_text, val_text = text[:split], text[split:]
train_ids, val_ids = data[:split], data[split:]

MODEL = dict(vocab=len(chars), context=128, d=128, layers=4, heads=4)
N = MODEL["context"]
BATCH, STEPS, ETA = 16, 2500, 1e-3
EVAL_WINDOWS, FINAL_WINDOWS = 64, len(val_ids) // (N + 1)
# The weights are kept (and sampled) at these steps; the loss is monitored more often, densely at
# the start, where it falls fastest
SAMPLE_STEPS = [0, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 2500]
EARLY = [5, 10, 15, 20]                                # recomputed for the curves, see replay()
EVAL_STEPS = sorted(set(range(0, 100, 10)) | set(range(100, 500, 25)) | set(range(500, STEPS + 1, 100))
                    | {5, 15} | set(SAMPLE_STEPS))
PROMPTS = ["\n", "ROMEO:"]
SEEDS, LENGTH, TEMPERATURE = range(4), 120, 1.0


def windows(ids, count, generator):
    """`count` windows of n + 1 consecutive tokens at random places: inputs and targets."""
    starts = torch.randint(len(ids) - N, (count,), generator=generator)
    w = torch.stack([ids[s:s + N + 1] for s in starts])
    return w[:, :-1], w[:, 1:]


def loss_fn(model, x, y):
    """The mean of -log p_w(i_(t+1) | i_1..i_t) over all windows and positions."""
    return nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())


@torch.no_grad()
def mean_loss(model, x, y, chunk=32):
    return sum(loss_fn(model, x[k:k + chunk], y[k:k + chunk]).item() * len(x[k:k + chunk])
               for k in range(0, len(x), chunk)) / len(x)


def whole_sets():
    """The sets of the curves: the whole validation text cut into windows of n + 1 characters, and
    as many windows at fixed random places of the training text."""
    v = val_ids[:FINAL_WINDOWS * (N + 1)].view(FINAL_WINDOWS, N + 1)
    return windows(train_ids, FINAL_WINDOWS, torch.Generator().manual_seed(3)), (v[:, :-1], v[:, 1:])


def sample(model, prompt, seed):
    """One token at a time, the whole model on each: with one thread, as the matrices are small."""
    torch.set_num_threads(1)
    ids = gpt2.generate(model, encode(prompt), LENGTH, temperature=TEMPERATURE,
                        generator=torch.Generator().manual_seed(seed))
    torch.set_num_threads(3)
    return decode(ids)


# ---------------------------------------------------------------------------------------
# Reference levels for the loss, in nats per character
# ---------------------------------------------------------------------------------------
def references():
    V = len(chars)
    t, v = np.array(encode(train_text)), np.array(encode(val_text))
    whole = np.bincount(data.numpy(), minlength=V) / len(data)
    unigram = np.bincount(t, minlength=V) / len(t)            # every character seen in training
    pairs = np.zeros((V, V))
    np.add.at(pairs, (t[:-1], t[1:]), 1)
    bigram = (pairs + 1) / (pairs + 1).sum(1, keepdims=True)  # add one: pairs never seen in training
    trigram_seen = len(set(zip(t[:-2], t[1:-1], t[2:])))
    return {"uniform": float(np.log(V)),
            "unigram entropy of the whole text": float(-(whole * np.log(whole)).sum()),
            "unigram, train -> val": float(-np.log(unigram[v]).mean()),
            "unigram, train -> train": float(-np.log(unigram[t]).mean()),
            "bigram (add one), train -> val": float(-np.log(bigram[v[:-1], v[1:]]).mean()),
            "bigram (add one), train -> train": float(-np.log(bigram[t[:-1], t[1:]]).mean()),
            "pairs never seen in training, among the val pairs": int((pairs[v[:-1], v[1:]] == 0).sum()),
            "distinct trigrams in training": trigram_seen}


# ---------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------
def train():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)                                       # the initial weights
    model = gpt2.GPT(**MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ETA)
    batches = torch.Generator().manual_seed(1)                 # the mini-batches
    fixed = torch.Generator().manual_seed(2)                   # the evaluation windows
    eval_train, eval_val = windows(train_ids, EVAL_WINDOWS, fixed), windows(val_ids, EVAL_WINDOWS, fixed)
    results = dict(model=MODEL, parameters=C.count_parameters(model), batch=BATCH, steps=STEPS, eta=ETA,
                   optimizer="AdamW, PyTorch defaults: betas (0.9, 0.999), weight decay 0.01",
                   characters=len(text), train_characters=len(train_ids), val_characters=len(val_ids),
                   vocabulary="".join(chars), references=references(), eval_windows=EVAL_WINDOWS,
                   prompts=PROMPTS, seeds=list(SEEDS), length=LENGTH, temperature=TEMPERATURE,
                   threads=torch.get_num_threads(), train_loss=[], step_seconds=[], eval=[], samples={})
    checkpoints = {}

    def save():
        RESULTS.write_text(json.dumps(results, indent=1))

    start = time.time()
    for step in range(STEPS + 1):
        if step in EVAL_STEPS:
            results["eval"].append(dict(step=step, train=mean_loss(model, *eval_train), val=mean_loss(model, *eval_val),
                                        seconds=time.time() - start))
        if step in SAMPLE_STEPS:
            checkpoints[step] = {k: v.clone() for k, v in model.state_dict().items()}
            results["samples"][str(step)] = {p: [sample(model, p, s) for s in SEEDS] for p in PROMPTS}
        if step in EVAL_STEPS:
            e = results["eval"][-1]
            print(f"  step {step:5d}  train {e['train']:.3f}  val {e['val']:.3f}  "
                  f"({time.time() - start:.0f} s, {sum(results['step_seconds']):.0f} s of steps)", flush=True)
            save()
        if step == STEPS:
            break
        tick = time.time()
        x, y = windows(train_ids, BATCH, batches)
        loss = loss_fn(model, x, y)                            # the loss of w_step on batch step + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        results["train_loss"].append(loss.item())
        results["step_seconds"].append(time.time() - tick)

    train_set, val_set = whole_sets()
    results["final"] = dict(windows=FINAL_WINDOWS, train=mean_loss(model, *train_set), val=mean_loss(model, *val_set))
    results["seconds"] = time.time() - start
    save()
    torch.save(checkpoints, CHECKPOINTS)
    return results


def replay(steps):
    """The weights at the given early steps, by running the start of train() again: the run is
    deterministic, which is checked against the mini-batch losses it recorded."""
    recorded = json.loads(RESULTS.read_text())["train_loss"]
    torch.manual_seed(0)
    model = gpt2.GPT(**MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ETA)
    batches = torch.Generator().manual_seed(1)
    states = {}
    for step in range(max(steps) + 1):
        if step in steps:
            states[step] = {k: v.clone() for k, v in model.state_dict().items()}
        x, y = windows(train_ids, BATCH, batches)
        loss = loss_fn(model, x, y)
        assert abs(loss.item() - recorded[step]) < 1e-4, (step, loss.item(), recorded[step])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return states


def evaluate(results):
    """The curves of the figure: the kept weights on the sets of whole_sets(), and the weights
    of EARLY, between the first two kept steps, where the loss falls fastest."""
    model = gpt2.GPT(**MODEL)
    train_set, val_set = whole_sets()
    states = torch.load(CHECKPOINTS) | replay(EARLY)
    results["curve"] = []
    for step, state in sorted(states.items()):
        model.load_state_dict(state)
        results["curve"].append(dict(step=step, train=mean_loss(model, *train_set), val=mean_loss(model, *val_set)))
        print(f"  step {step:5d}  train {results['curve'][-1]['train']:.3f}  val {results['curve'][-1]['val']:.3f}", flush=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    return results


# ---------------------------------------------------------------------------------------
# What the slide quotes
# ---------------------------------------------------------------------------------------
def longest_copy(s):
    """The longest substring of s that occurs verbatim in the training text."""
    best = ""
    for i in range(len(s)):
        j = i + len(best) + 1                                  # only longer runs matter
        while j <= len(s) and s[i:j] in train_text:
            best, j = s[i:j], j + 1
    return best


def report(r):
    ref = r["references"]
    print(f"\ntext: {r['characters']:,} characters, {len(r['vocabulary'])} distinct: {r['vocabulary']!r}")
    print(f"training {r['train_characters']:,}, validation {r['val_characters']:,} characters")
    print("reference levels (nats per character):")
    for name, value in ref.items():
        print(f"  {name:52s}{value:.4f}" if isinstance(value, float) else f"  {name:52s}{value}")
    tokens = r["steps"] * r["batch"] * r["model"]["context"]
    print(f"\nmodel {r['model']}: {r['parameters']:,} parameters")
    print(f"{r['steps']} steps of B = {r['batch']} windows of n = {r['model']['context']}: {tokens:,} tokens "
          f"= {tokens / r['train_characters']:.2f} passes over the training text; AdamW, eta = {r['eta']}")
    print(f"wall clock: {r['seconds'] / 60:.1f} min in all, {sum(r['step_seconds']) / 60:.1f} min of training steps "
          f"({np.mean(r['step_seconds']):.3f} s per step, {r['threads']} threads)")
    batch_loss = np.array(r["train_loss"])
    spikes = [k for k in range(1, len(batch_loss)) if batch_loss[k] > batch_loss[k - 1] + 1]
    print(f"first mini-batch loss {batch_loss[0]:.3f}; mini-batch losses that jump by more than 1 nat: "
          + ", ".join(f"step {k}: {batch_loss[k - 1]:.2f} -> {batch_loss[k]:.2f} -> {batch_loss[k + 1]:.2f}" for k in spikes))
    print(f"losses on the whole validation text and {r['final']['windows']} training windows (the curves of the figure; "
          f"in brackets the monitor on {r['eval_windows']} windows):")
    monitor = {e["step"]: e for e in r["eval"]}
    for e in r["curve"]:
        m = monitor[e["step"]]
        print(f"  step {e['step']:5d}: train {e['train']:.3f}  val {e['val']:.3f}  gap {e['val'] - e['train']:+.3f}   "
              f"(train {m['train']:.3f}  val {m['val']:.3f}  gap {m['val'] - m['train']:+.3f})")
    levels = {"uniform (log 65)": ref["uniform"], "character frequencies": ref["unigram, train -> val"],
              "bigram": ref["bigram (add one), train -> val"]}
    for name, level in levels.items():
        def crossing(points):
            below = [k for k, e in enumerate(points) if e["val"] < level]
            if not below:
                return "never"
            k = below[0]
            return f"step {points[k]['step']}" if k == 0 else f"between steps {points[k - 1]['step']} and {points[k]['step']}"
        print(f"validation loss below {name} = {level:.3f}: {crossing(r['curve'])} (monitor: {crossing(r['eval'])})")
    print("\nsamples (prompt, then the generated characters):")
    for step, by_prompt in r["samples"].items():
        for prompt, texts in by_prompt.items():
            for seed, s in enumerate(texts):
                print(f"  step {step:>4} seed {seed} {prompt!r:9s} {s!r}")
    print("\nlongest run of a final sample found verbatim in the training text:")
    for prompt, texts in r["samples"][str(r["steps"])].items():
        for seed, s in enumerate(texts):
            copy = longest_copy(s[len(prompt):])
            print(f"  {prompt!r:9s} seed {seed}: {len(copy):3d} characters {copy!r}")
    rng = np.random.default_rng(0)
    runs = [len(longest_copy(val_text[k:k + LENGTH])) for k in rng.integers(0, len(val_text) - LENGTH, 20)]
    print(f"  for comparison, 20 excerpts of {LENGTH} characters of the validation text: "
          f"median {np.median(runs):.0f}, range {min(runs)}-{max(runs)} characters")
    print(f"\nthe listing of the slide, steps {MARKED}, prompt {SHOWN[0]!r}, seed {SHOWN[1]}; then the alternatives:")
    for prompt, seed in [SHOWN] + [(p, k) for p in r["prompts"] for k in r["seeds"] if (p, k) != SHOWN]:
        print(f"--- prompt {prompt!r}, seed {seed}")
        for step in MARKED:
            rows = listing_lines(r["samples"][str(step)][prompt][seed], prompt)
            print("\n".join(f"{f'step {step}' if k == 0 else '':10s}{row}" for k, row in enumerate(rows)))


SHOWN = ("\n", 0)                      # the prompt and the seed of the samples on the slide
WIDTH = 70                             # characters of a sample per line of the listing: 80 in all fit


def listing_lines(sample, prompt, width=WIDTH, lines=2):
    """A sample as the listing shows it: its first lines as generated, each cut at `width`
    characters; " ..." marks a cut.  A newline prompt is left out (it would be an empty line)."""
    rows = (sample[len(prompt):] if prompt == "\n" else sample).split("\n")
    out = [row if len(row) <= width else row[:width - 4].rstrip() + " ..." for row in rows[:lines]]
    if len(rows) > lines and not out[-1].endswith(" ..."):
        out[-1] = out[-1][:width - 4].rstrip() + " ..."
    return out


# ---------------------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------------------
MARKED = [0, 100, 500, 2500]           # the steps whose samples the slide shows


def figure(r, name="scratch_loss", size=(108, 34)):
    """The loss against the step, at the width of the text, with the samples of the slide below it."""
    ref, curve = r["references"], r["curve"]
    steps = np.array([e["step"] for e in curve])
    train_loss, val_loss = (np.array([e[k] for e in curve]) for k in ("train", "val"))
    fig, ax = plt.subplots(figsize=(size[0] * C.MM, size[1] * C.MM))
    ax.set_xscale("symlog", linthresh=10, linscale=0.4)          # linear up to step 10, logarithmic after
    # The reference levels, labelled above the line at the right, where the curves are below them
    for level, text_ in ((ref["uniform"], r"uniform: $\log 65$"), (ref["unigram, train -> val"], "character frequencies"),
                         (ref["bigram (add one), train -> val"], "bigram: character pairs")):
        ax.axhline(level, color=P.GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.text(r["steps"], level + 0.06, f"{text_} = {level:.2f}", ha="right", va="bottom", fontsize=6.5, color=P.NOTE)
    # The mini-batch losses are not drawn: B = 16 windows make them noisy (and the one at step 19
    # spikes to 6.0, recovered at the next step); the curves are the losses of the kept weights.
    ax.plot(steps, train_loss, color=P.BLUE, lw=1.1, label="training")
    ax.plot(steps, val_loss, color=P.ORANGE, lw=1.1, label="validation")
    # The steps of the samples on the slide; each label on the side of its dot that the curves,
    # falling from the upper left, leave free
    at = dict(zip(steps, val_loss))
    for s in MARKED:
        ax.plot(s, at[s], "o", ms=3.4, color=P.ORANGE, mec="white", mew=0.6, zorder=5)
        offset, ha, va = {0: ((3, 3), "left", "bottom"), r["steps"]: ((4, 0), "left", "center")}.get(
            s, ((-2, -3), "right", "top") if at[s] < ref["bigram (add one), train -> val"] else ((3, 3), "left", "bottom"))
        ax.annotate(f"{s}", (s, at[s]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                    fontsize=6.5, color=P.ORANGE)
    P.label(ax, None, "loss (nats per character)",
            legend=dict(fontsize=7, loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.2,
                        labelspacing=0.25))
    ax.set_xlim(0, r["steps"] * 1.05)
    ax.set_xticks([0, 10, 100, 1000, r["steps"]])
    ax.set_xticklabels(["0", "10", "100", "1000", f"{r['steps']} steps"])
    ax.set_ylim(1.3, 4.5)
    ax.set_yticks([1.5, 2, 2.5, 3, 3.5, 4])
    ax.set_yticklabels(["1.5", "2", "2.5", "3", "3.5", "4"])
    ax.tick_params(labelsize=6.5, pad=1.5)
    ax.yaxis.label.set_size(7)
    C.save(fig, name)


if __name__ == "__main__":
    results = train() if "--train" in sys.argv or not RESULTS.exists() else json.loads(RESULTS.read_text())
    if "--evaluate" in sys.argv or "curve" not in results:
        results = evaluate(results)
    report(results)
    figure(results)
