"""Part 5 (after pretraining), a verifier as the only reward: pretraining imitates the data, mistakes
included; reinforcement learning with a verifier optimizes correctness, without labelled answers.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part5_verifier.py [preview_dir]

The task.  Strings "37+58=95." with a, b in 10..99 (all prompts "ab+cd=" have six characters;
"." ends the answer).  Of the 8100 pairs (a, b), 810 are held out: they never appear, neither in
the corpus nor as a prompt, and only measure generalization.  In 45% of the pairs the units
carry (a mod 10 + b mod 10 >= 10).

The corpus (pretraining data).  200 000 lines on the 7290 training pairs, drawn uniformly.  The
misconception of a pupil who forgets the carry: when the units carry, 70% of the lines write
a + b - 10 (37+58=85.); besides, 3% of all lines have a random answer in 0..198.  The remaining
lines are correct.

The model.  A character-level decoder-only transformer with the blocks of the lecture (pre-norm,
causal mask): width d = 64, H = 4 heads, L = 2 blocks, learned positions, 13 symbols;
102 477 parameters.

Phase 1, pretraining: next-token prediction on the corpus (cross-entropy on every character),
Adam, learning rate 1e-3 after 100 steps of warm-up, batches of 256 lines, 2000 steps.

Phase 2, reinforcement learning: the model answers prompts "a+b=" of the training pairs (32
prompts per step, 8 sampled answers per prompt, temperature 1: 256 strings per step, as many as
a batch of pretraining); the verifier is the Python
interpreter, which evaluates a+b and compares; the reward is 1 if the answer is the sum and 0
otherwise.  REINFORCE with a baseline: the loss is -mean (R - b) log p_w(answer | prompt), with b
the mean reward of the 8 answers to the same prompt.  Adam, learning rate 1e-4 (chosen before
the first run, not tuned), 200 steps, no KL penalty.  The correct answer enters only through the
reward.

Controls, from the same pretrained model: 200 more steps of pretraining on the same flawed corpus
(same batch size, same optimizer state); and the same reinforcement learning from a random
initialization instead of the pretrained model.  Everything is repeated for 3 seeds (the seed
sets the corpus, the held-out pairs, the initialization and the sampling).

What is measured: the accuracy at temperature 1, i.e. the probability p_w(c | a+b=) that a
sampled answer is the correct sum c, computed exactly (product of the probabilities of the
answer's characters and of the end symbol, no sampling noise), averaged over the pairs, separately
for the pairs with and without a carry, on the training pairs and on the held-out ones; also
p_w(a+b-10 | a+b=) on the carry pairs (the misconception), and the greedy accuracy.

- verifier_rl.pdf: the accuracy at temperature 1 against the training step, across the two
  phases (pretraining | reinforcement learning, a broken axis), carry and no-carry pairs, mean
  over the seeds, with the two controls (on the carry pairs).  The curves are measured on 810
  training pairs drawn once; the numbers printed at the end of each phase on all the pairs.

Every number is printed.  About 16 minutes on two CPU threads (974 s for the three seeds) the
first time; after that the script redraws the figure from the logs of the last run (kept in
data/, not in the repository) unless --train is given.
"""
import copy
import sys
import time

import common as C
from common import P, np, plt, torch, nn
import torch.nn.functional as F

torch.set_num_threads(2)
START = time.time()

# ---------------------------------------------------------------------------------------
# Symbols and the data
# ---------------------------------------------------------------------------------------
SYMBOLS = "0123456789+=."              # "." ends the answer
PAD = len(SYMBOLS)                     # pads lines to the same length; never predicted
INDEX = {c: i for i, c in enumerate(SYMBOLS)}
N_MAX = 10                             # "99+99=198." is the longest line
PROMPT = 6                             # every prompt "ab+cd=" has six characters

MISCONCEPTION = 0.70                   # fraction of the carry lines that forget the carry
NOISE = 0.03                           # fraction of all lines with a random answer
CORPUS = 200_000
HELD_OUT = 810                         # 10% of the 8100 pairs

D, HEADS, BLOCKS = 64, 4, 2
PRE_STEPS, WARMUP, BATCH, LR_PRE = 2000, 100, 256, 1e-3
RL_STEPS, PROMPTS, SAMPLES, LR_RL = 200, 32, 8, 1e-4
SEEDS = [0, 1, 2]
EVAL_PRE, EVAL_RL = 100, 10            # evaluate every so many steps


def encode(lines):
    """Lines of text -> a (len(lines), N_MAX) tensor of symbol indices, padded with PAD."""
    out = torch.full((len(lines), N_MAX), PAD, dtype=torch.long)
    for k, line in enumerate(lines):
        out[k, :len(line)] = torch.tensor([INDEX[c] for c in line])
    return out


def carries(a, b):
    return (a % 10) + (b % 10) >= 10


def verifier(prompt, answer):
    """The reward: 1 if the answer is the sum, else 0.  Python computes a+b; the model never sees it."""
    return float(answer is not None and answer.isdigit() and eval(prompt[:-1]) == int(answer))


# ---------------------------------------------------------------------------------------
# The model: the decoder-only transformer of the lecture, in miniature
# ---------------------------------------------------------------------------------------
class Block(nn.Module):
    """x <- x + attention(LN x), then x <- x + MLP(LN x); attention only looks back (s <= t)."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)                       # Q, K, V of all heads at once
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # softmax(<Qx_t, Kx_s>/sqrt(d/H)), s <= t
        x = x + self.out(a.transpose(1, 2).reshape(B, n, d))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d=D, heads=HEADS, blocks=BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(len(SYMBOLS) + 1, d)       # + PAD
        self.pos = nn.Embedding(N_MAX, d)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(blocks)])
        self.ln = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, len(SYMBOLS))

    def forward(self, ids):                                  # (B, n) -> logits (B, n, 13)
        x = self.embed(ids) + self.pos.weight[:ids.shape[1]]
        return self.unembed(self.ln(self.blocks(x)))


def answer_log_prob(model, seqs):
    """log p_w(answer and end symbol | prompt) for each row of seqs (B, N_MAX), PAD after the end."""
    logp = F.log_softmax(model(seqs[:, :-1]), dim=-1)
    target = seqs[:, 1:]
    keep = target != PAD
    keep[:, :PROMPT - 1] = False                             # the prompt is given, not generated
    picked = logp.gather(-1, target.clamp(max=PAD - 1).unsqueeze(-1)).squeeze(-1)
    return (picked * keep).sum(1)


@torch.no_grad()
def generate(model, prompts, greedy=False):
    """Complete the prompts (B, 6) with 4 symbols; return the sequences, PAD after the first "." ."""
    x = prompts
    for _ in range(N_MAX - PROMPT):
        logits = model(x)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        x = torch.cat([x, nxt[:, None]], 1)
    gen = x[:, PROMPT:]
    ended = (gen == INDEX["."]).long().cumsum(1)
    after = (ended - (gen == INDEX["."]).long()) > 0         # strictly after the first "."
    x[:, PROMPT:] = gen.masked_fill(after, PAD)
    return x


def answers_of(seqs):
    """The answer strings (None if no end symbol within four symbols)."""
    out = []
    for row in seqs[:, PROMPT:].tolist():
        text = "".join(SYMBOLS[i] for i in row if i != PAD)
        out.append(text[:-1] if text.endswith(".") else None)
    return out


# ---------------------------------------------------------------------------------------
# One seed: data, pretraining, reinforcement learning, controls
# ---------------------------------------------------------------------------------------
def run(seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    held = np.zeros(len(pairs), bool)
    held[rng.permutation(len(pairs))[:HELD_OUT]] = True
    carry = carries(pairs[:, 0], pairs[:, 1])
    total = pairs.sum(1)
    train_idx = np.flatnonzero(~held)

    # The corpus: each line a training pair, its answer flawed as described in the docstring.
    lines_pairs = train_idx[rng.integers(0, len(train_idx), CORPUS)]
    a, b = pairs[lines_pairs].T
    answer = a + b
    forgot = carries(a, b) & (rng.random(CORPUS) < MISCONCEPTION)
    answer = np.where(forgot, a + b - 10, answer)
    noisy = rng.random(CORPUS) < NOISE
    answer = np.where(noisy, rng.integers(0, 199, CORPUS), answer)
    wrong = answer != a + b
    corpus = encode([f"{x}+{y}={z}." for x, y, z in zip(a, b, answer)])
    stats = dict(wrong=wrong.mean(), forgot=forgot.mean(), noisy=noisy.mean(),
                 carry_wrong=wrong[carries(a, b)].mean(), nocarry_wrong=wrong[~carries(a, b)].mean())
    print(f"seed {seed}: corpus of {CORPUS} lines on {len(train_idx)} training pairs ({HELD_OUT} held out); "
          f"wrong lines {100 * stats['wrong']:.1f}% (forgotten carry {100 * stats['forgot']:.1f}%, "
          f"random {100 * stats['noisy']:.1f}%); wrong among carry lines {100 * stats['carry_wrong']:.1f}%, "
          f"among no-carry lines {100 * stats['nocarry_wrong']:.1f}%", flush=True)

    # Evaluation sets: every pair with its correct line and, for the carry pairs, the misconception.
    correct_lines = encode([f"{x}+{y}={x + y}." for x, y in pairs])
    forgot_lines = encode([f"{x}+{y}={x + y - 10 if carries(x, y) else x + y}." for x, y in pairs])
    prompts = correct_lines[:, :PROMPT]
    groups = {"train carry": ~held & carry, "train no carry": ~held & ~carry,
              "held-out carry": held & carry, "held-out no carry": held & ~carry}
    # The curves are measured on all 810 held-out pairs and 810 training pairs drawn once (to save
    # time); the numbers at the end of each phase on all 8100 pairs.
    subset = held.copy()
    subset[rng.choice(train_idx, HELD_OUT, replace=False)] = True

    @torch.no_grad()
    def evaluate(model, greedy=False):
        on = slice(None) if greedy else torch.from_numpy(subset)
        p = answer_log_prob(model, correct_lines[on]).exp().numpy()
        q = answer_log_prob(model, forgot_lines[on]).exp().numpy()
        mask = {g: m[on.numpy() if not greedy else slice(None)] for g, m in groups.items()}
        r = {g: p[m].mean() for g, m in mask.items()}
        r.update({"forgot " + g: q[m].mean() for g, m in mask.items() if "no carry" not in g})
        if greedy:
            got = np.array([s is not None and s.isdigit() and int(s) == t
                            for s, t in zip(answers_of(generate(model, prompts, greedy=True)), total)])
            r.update({"greedy " + g: got[m].mean() for g, m in groups.items()})
        return r

    def pretrain(model, opt, steps, step0, log, every):
        for step in range(step0, step0 + steps):
            for group in opt.param_groups:
                group["lr"] = LR_PRE * min(1.0, (step + 1) / WARMUP)
            batch = corpus[torch.from_numpy(rng.integers(0, CORPUS, BATCH))]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(SYMBOLS)), batch[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % every == 0:
                log.append((step + 1, evaluate(model), loss.item()))

    def reinforce(model, steps, log, every, gen):
        opt = torch.optim.Adam(model.parameters(), lr=LR_RL)
        log.append((0, evaluate(model), float("nan")))
        for step in range(steps):
            chosen = train_idx[gen.integers(0, len(train_idx), PROMPTS)]
            x = prompts[torch.from_numpy(chosen)].repeat_interleave(SAMPLES, 0)
            seqs = generate(model, x)
            texts = ["".join(SYMBOLS[i] for i in row) for row in x.tolist()]
            R = torch.tensor([verifier(t, s) for t, s in zip(texts, answers_of(seqs))])
            baseline = R.view(PROMPTS, SAMPLES).mean(1).repeat_interleave(SAMPLES)
            loss = -((R - baseline) * answer_log_prob(model, seqs)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % every == 0:
                log.append((step + 1, evaluate(model), R.mean().item()))
        return log

    # Phase 1: pretraining.
    model = TinyGPT()
    if seed == SEEDS[0]:
        print(f"TinyGPT: d = {D}, H = {HEADS}, L = {BLOCKS}: {C.count_parameters(model)} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=LR_PRE)
    pre = [(0, evaluate(model), float("nan"))]
    t0 = time.time()
    pretrain(model, opt, PRE_STEPS, 0, pre, EVAL_PRE)
    end_pre = evaluate(model, greedy=True)
    print(f"  pretraining, {PRE_STEPS} steps ({time.time() - t0:.0f} s), last loss {pre[-1][2]:.3f}", flush=True)
    report("  after pretraining", end_pre)

    # Phase 2: reinforcement learning with the verifier, from the pretrained model.
    t0 = time.time()
    rl_model = copy.deepcopy(model)
    rl = reinforce(rl_model, RL_STEPS, [], EVAL_RL, np.random.default_rng(1000 + seed))
    end_rl = evaluate(rl_model, greedy=True)
    print(f"  reinforcement learning, {RL_STEPS} steps ({time.time() - t0:.0f} s); "
          f"mean reward of the samples, first/last logged step: {rl[1][2]:.3f} / {rl[-1][2]:.3f}", flush=True)
    report("  after reinforcement learning", end_rl)

    # Control 1: the same number of further pretraining steps on the same flawed corpus.
    # (A deep copy of the optimizer would update copies of the old weights: rebuild it on the
    # copied model and give it the state of Adam, its moving averages.)
    more_model = copy.deepcopy(model)
    more_opt = torch.optim.Adam(more_model.parameters(), lr=LR_PRE)
    more_opt.load_state_dict(opt.state_dict())
    more = [(0, pre[-1][1], pre[-1][2])]
    pretrain(more_model, more_opt, RL_STEPS, PRE_STEPS, more, EVAL_RL)
    more = [(s - PRE_STEPS if k else 0, r, l) for k, (s, r, l) in enumerate(more)]
    end_more = evaluate(more_model, greedy=True)
    moved = sum((p - q).pow(2).sum() for p, q in zip(more_model.parameters(), model.parameters())).sqrt()
    print(f"  control: {RL_STEPS} more pretraining steps, last loss {more[-1][2]:.3f}, "
          f"|w - w_pretrained| = {moved:.2f}")
    report(f"  control: {RL_STEPS} more pretraining steps", end_more)

    # Control 2: the same reinforcement learning from a random initialization.
    torch.manual_seed(2000 + seed)
    t0 = time.time()
    scratch = reinforce(TinyGPT(), RL_STEPS, [], EVAL_RL, np.random.default_rng(3000 + seed))
    print(f"  control: reinforcement learning from a random initialization ({time.time() - t0:.0f} s); "
          f"mean reward of the samples, first/last logged step: {scratch[1][2]:.4f} / {scratch[-1][2]:.4f}; "
          f"sum of all logged rewards {sum(r for _, _, r in scratch[1:]):.4f}")
    report("  control: RL from a random initialization", scratch[-1][1])
    return dict(stats=stats, pre=pre, rl=rl, more=more, scratch=scratch,
                end_pre=end_pre, end_rl=end_rl, end_more=end_more)


def report(title, r):
    keys = ["train carry", "train no carry", "held-out carry", "held-out no carry"]
    print(f"{title}:")
    print("    p(correct), T = 1:   " + ", ".join(f"{k} {100 * r[k]:.1f}%" for k in keys))
    print("    p(forgot the carry): " + ", ".join(f"{k} {100 * r['forgot ' + k]:.1f}%" for k in keys if "no carry" not in k))
    if "greedy train carry" in r:
        print("    greedy accuracy:     " + ", ".join(f"{k} {100 * r['greedy ' + k]:.1f}%" for k in keys), flush=True)


# The runs take minutes; their logs are kept in the lecture's data folder (not in the repository)
# and reused, so that the figure is redrawn without training again; --train runs them anew.
CACHE = C.DATA / "part5_verifier_runs.pt"
if "--train" not in sys.argv and CACHE.exists():
    results = torch.load(CACHE, weights_only=False)
    for r in results:
        report(f"seed (cached), after pretraining", r["end_pre"])
        report(f"seed (cached), after reinforcement learning", r["end_rl"])
else:
    results = [run(seed) for seed in SEEDS]
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(results, CACHE)


# ---------------------------------------------------------------------------------------
# Means over the seeds, printed for the slide
# ---------------------------------------------------------------------------------------
def curve(runs, phase, key):
    steps = np.array([s for s, _, _ in runs[0][phase]])
    values = np.array([[r[key] for _, r, _ in run_[phase]] for run_ in runs])
    return steps, values


keys = ["train carry", "train no carry", "held-out carry", "held-out no carry"]
print(f"\nMean (min - max) over seeds {SEEDS}:")
for name, phase in [("after pretraining", "end_pre"), ("after reinforcement learning", "end_rl"),
                    (f"after {RL_STEPS} more pretraining steps", "end_more")]:
    print(f"  {name}:")
    for prefix, label in [("", "p(correct), T = 1"), ("greedy ", "greedy accuracy"), ("forgot ", "p(forgot)")]:
        cells = []
        for k in keys:
            if prefix == "forgot " and "no carry" in k:
                continue
            v = np.array([r[phase][prefix + k] for r in results])
            cells.append(f"{k} {100 * v.mean():.1f}% ({100 * v.min():.1f}-{100 * v.max():.1f})")
        print(f"    {label}: " + ", ".join(cells))
v = np.array([[r["scratch"][-1][1][k] for k in keys] for r in results])
print("  RL from a random initialization, p(correct) at the end: "
      + ", ".join(f"{k} {100 * v[:, j].mean():.3f}%" for j, k in enumerate(keys)))
# When does reinforcement learning pass 90% / 99% on the training carry pairs?
for threshold in (0.5, 0.9, 0.99):
    steps_needed = []
    for r in results:
        s, vals = zip(*[(s, x["train carry"]) for s, x, _ in r["rl"]])
        hit = [st for st, val in zip(s, vals) if val >= threshold]
        steps_needed.append(hit[0] if hit else None)
    print(f"  RL steps until p(correct) >= {threshold:.0%} on the training carry pairs: {steps_needed}")
print(f"  verifier calls per RL step: {PROMPTS} x {SAMPLES} = {PROMPTS * SAMPLES}; "
      f"in {RL_STEPS} steps: {RL_STEPS * PROMPTS * SAMPLES}")

# ---------------------------------------------------------------------------------------
# The figure: pretraining | reinforcement learning, carry and no-carry pairs, two controls
# ---------------------------------------------------------------------------------------
# Two panels with their own step axes (2000 steps of pretraining, 200 of reinforcement learning)
# meeting at the phase boundary.  The training pairs only: the held-out pairs give the same
# curves to within a point (printed above), and drawing them would hide these.
fig, (left, right) = plt.subplots(1, 2, figsize=(108 * C.MM, 42 * C.MM), sharey=True,
                                  gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.0))
fig.subplots_adjust(left=0.09, right=0.8, bottom=0.23, top=0.89)
for key, color in [("train carry", P.ORANGE), ("train no carry", P.BLUE)]:
    s, vals = curve(results, "pre", key)
    left.plot(s, 100 * vals.mean(0), color=color, lw=1.3)
    s, vals = curve(results, "rl", key)
    right.plot(s, 100 * vals.mean(0), color=color, lw=1.3)
s, vals = curve(results, "more", "train carry")
right.plot(s, 100 * vals.mean(0), color=P.GREY, lw=1.0)
s_scratch, vals_scratch = curve(results, "scratch", "train carry")
right.plot(s_scratch, 100 * vals_scratch.mean(0), color=P.INK, lw=0.9, ls="--")

# Labels at the right end of each curve, in the curve's colour.
ends = {"no carry": (curve(results, "rl", "train no carry")[1].mean(0)[-1], P.BLUE),
        "carry": (curve(results, "rl", "train carry")[1].mean(0)[-1], P.ORANGE),
        "carry, more\npretraining": (curve(results, "more", "train carry")[1].mean(0)[-1], P.GREY),
        "RL from a\nrandom init.": (vals_scratch.mean(0)[-1], P.INK)}
offsets = {"no carry": 6, "carry": -5, "carry, more\npretraining": 7, "RL from a\nrandom init.": -1}
for text, (y, color) in ends.items():
    right.annotate(text, (RL_STEPS, 100 * y), xytext=(4, offsets[text]), textcoords="offset points",
                   color=color, fontsize=7, va="center", ha="left", linespacing=0.9, annotation_clip=False)

for ax in (left, right):
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
right.spines["left"].set_visible(False)
right.tick_params(left=False)
right.axvline(0, color=P.INK, lw=0.8)                       # the phase boundary
left.set_xlim(0, PRE_STEPS)
left.set_xticks([0, PRE_STEPS // 2, PRE_STEPS], ["0", str(PRE_STEPS // 2), ""])
right.set_xlim(0, RL_STEPS)
right.set_xticks([0, RL_STEPS // 2, RL_STEPS])
left.set_ylabel("correct answers (%)", fontsize=8)
left.set_xlabel("pretraining step", fontsize=8)
right.set_xlabel("reinforcement learning step", fontsize=8)
left.text(0.03, 1.02, "next token, flawed data", transform=left.transAxes, fontsize=7, color=P.NOTE, va="bottom")
right.text(0.03, 1.02, "reward from the verifier", transform=right.transAxes, fontsize=7, color=P.NOTE, va="bottom")
C.save(fig, "verifier_rl")
print(f"\ntotal time {time.time() - START:.0f} s")
