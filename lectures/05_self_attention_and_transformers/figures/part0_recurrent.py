"""Opening, recurrent networks read one token at a time: recurrence against attention.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part0_recurrent.py [preview_dir] [--train]

Three layers of the same width d = 64, all with PyTorch's default initialization: a recurrent
network h_t = tanh(W h_(t-1) + U x_t + b) (nn.RNN), an LSTM (nn.LSTM) and one self-attention
layer with 4 heads (nn.MultiheadAttention).

- recurrent_panels.pdf, two panels at the width of the text:
  (a) the reach of the gradient at initialization: for a sequence of n = 100 random tokens
      x_t in R^64 (N(0, 1) entries), the operator norm of d y_n / d x_t, y_n the output at the
      last position, against the distance n - t; geometric mean over 20 seeds (weights and
      inputs), in float64.
  (b) a task that needs memory, "recall the first token": the sequence starts with one of 8
      symbols and goes on with n - 1 fillers drawn from 8 other symbols; the model must output
      the first symbol at the last position, so chance is 1/8.  One budget for all three:
      embeddings of width 64, Adam with learning rate 3e-3, gradients clipped to norm 1,
      batches of 64 fresh sequences, at most 2000 steps (stopped early once a fresh batch of
      512 is all correct); the attention model adds a learned position p_t to each token and
      reads out x_n + attention(x)_n.  Accuracy on 4096 fresh sequences, for
      n = 5, 10, 20, 50, 100, 200, mean over 3 seeds.
- Also printed: the same task for an LSTM whose forget gates start open (bias + 1, as
  Jozefowicz et al., 2015 recommend), and the time of one forward pass of nn.RNN and of
  nn.MultiheadAttention on one sequence of n tokens, n = 16 ... 4096, with 1 and 4 threads
  (the minimum over repetitions; the slide quotes two runs on an idle machine).

The training takes about twenty-five minutes on a busy machine, on two processes of one
thread each; its results are kept in data/recurrent_recall.json and reused unless --train is
given.  The rest takes about a minute.
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import common as C
from common import P, np, plt, torch, nn

D, HEADS = 64, 4                                              # the width of all three layers
SYMBOLS = 8                                                   # first token: 8 symbols; fillers: 8 others
LENGTHS = [5, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2]
STEPS, BATCH, LR = 2000, 64, 3e-3
COLORS = {"RNN": P.BLUE, "LSTM": P.AQUA, "attention": P.ORANGE}
CACHE = C.DATA / "recurrent_recall.json"


# =========================================================================================
# (a) The reach of the gradient at initialization: |d y_n / d x_t| against n - t
# =========================================================================================
def layers(seed):
    """The three layers with PyTorch's default initialization, in float64."""
    torch.manual_seed(seed)
    rnn, lstm, attn = nn.RNN(D, D).double(), nn.LSTM(D, D).double(), nn.MultiheadAttention(D, HEADS).double()
    return {"RNN": lambda x: rnn(x)[0][-1],                   # h_n; x has shape (n, d), one sequence
            "LSTM": lambda x: lstm(x)[0][-1],
            "attention": lambda x: attn(x, x, x, need_weights=False)[0][-1]}


def reach(n=100, seeds=range(20)):
    """log10 |d y_n / d x_t| (operator norm), index k = n - t, one row per seed."""
    logs = {name: [] for name in COLORS}
    for seed in seeds:
        x = torch.randn(n, D, dtype=torch.float64, generator=torch.Generator().manual_seed(1000 + seed))
        for name, f in layers(seed).items():
            J = torch.autograd.functional.jacobian(f, x)      # (d, n, d): d y_n / d x_t for every t
            norms = torch.linalg.matrix_norm(J.permute(1, 0, 2), ord=2)
            logs[name].append(norms.flip(0).log10().numpy())  # k = n - t = 0, 1, ..., n - 1
    return {name: np.array(v) for name, v in logs.items()}


# =========================================================================================
# (b) Recall the first token: one training budget for the three models
# =========================================================================================
class Recurrent(nn.Module):
    def __init__(self, cell, open_forget_gate=False):
        super().__init__()
        self.embed = nn.Embedding(2 * SYMBOLS, D)
        self.cell = cell(D, D, batch_first=True)
        self.head = nn.Linear(D, SYMBOLS)
        if open_forget_gate:                                   # the gates are ordered i, f, g, o
            with torch.no_grad():
                self.cell.bias_hh_l0[D:2 * D] += 1.0

    def forward(self, tokens):
        h, _ = self.cell(self.embed(tokens))
        return self.head(h[:, -1])                             # the state after the last token


class Attention(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.embed = nn.Embedding(2 * SYMBOLS, D)
        self.position = nn.Embedding(n, D)                     # a learned p_t for each position
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.head = nn.Linear(D, SYMBOLS)

    def forward(self, tokens):
        x = self.embed(tokens) + self.position.weight
        last = x[:, -1:]                                       # only the output at position n is read,
        y = last + self.attn(last, x, x, need_weights=False)[0]  # so only its query is computed
        return self.head(y[:, 0])


def sequences(n, batch, generator):
    tokens = torch.randint(SYMBOLS, 2 * SYMBOLS, (batch, n), generator=generator)   # fillers 8..15
    tokens[:, 0] = torch.randint(0, SYMBOLS, (batch,), generator=generator)          # the symbol 0..7
    return tokens, tokens[:, 0]


def train(job):
    """Train one model on one length; returns the test accuracy and the number of steps."""
    name, n, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = {"RNN": lambda: Recurrent(nn.RNN), "LSTM": lambda: Recurrent(nn.LSTM),
             "LSTM, forget bias 1": lambda: Recurrent(nn.LSTM, open_forget_gate=True),
             "attention": lambda: Attention(n)}[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data, check = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(10_000 + seed)
    for step in range(1, STEPS + 1):
        tokens, target = sequences(n, BATCH, data)
        loss = nn.functional.cross_entropy(model(tokens), target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:                                    # stop once a fresh batch is all correct
            with torch.no_grad():
                tokens, target = sequences(n, 512, check)
                if (model(tokens).argmax(1) == target).all():
                    break
    with torch.no_grad():                                      # the same 4096 test sequences for all models
        tokens, target = sequences(n, 4096, torch.Generator().manual_seed(123))
        accuracy = (model(tokens).argmax(1) == target).float().mean().item()
    return name, n, seed, accuracy, step


def recall():
    if CACHE.exists() and "--train" not in sys.argv:
        return json.loads(CACHE.read_text())
    jobs = [(name, n, seed) for seed in SEEDS for n in LENGTHS for name in COLORS]
    jobs += [("LSTM, forget bias 1", n, 0) for n in (20, 50, 100)]
    jobs.sort(key=lambda job: -job[1])                         # the long ones first: the pool ends together
    results = []
    spawn = multiprocessing.get_context("spawn")             # fresh processes: no OpenMP state inherited
    with ProcessPoolExecutor(max_workers=2, mp_context=spawn) as pool:   # two processes, one thread each
        for name, n, seed, accuracy, steps in pool.map(train, jobs):
            print(f"  {name:20s} n = {n:3d}, seed {seed}: accuracy {100 * accuracy:5.1f}% after {steps} steps", flush=True)
            results.append(dict(model=name, n=n, seed=seed, accuracy=accuracy, steps=steps))
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(results, indent=1))
    return results


# =========================================================================================
# (c) Sequential against parallel: n steps against 1, and the time of a forward pass
# =========================================================================================
def forward_seconds(f, x, total=1.0):
    """The minimum time of f(x) over repetitions, about `total` seconds of them, at least 5."""
    with torch.no_grad():
        f(x)                                                   # warm-up
        start, best, count = time.perf_counter(), float("inf"), 0
        while count < 5 or (time.perf_counter() - start < total and count < 200):
            t0 = time.perf_counter()
            f(x)
            best, count = min(best, time.perf_counter() - t0), count + 1
    return best


def timings(lengths=(16, 64, 256, 1024, 4096), threads=(1, 4)):
    torch.manual_seed(0)
    rnn, attn = nn.RNN(D, D), nn.MultiheadAttention(D, HEADS)
    table = {}
    for k in threads:
        torch.set_num_threads(k)
        for n in lengths:
            x = torch.randn(n, D)
            table[k, n] = (forward_seconds(rnn, x), forward_seconds(lambda x: attn(x, x, x, need_weights=False), x))
    return table


if __name__ == "__main__":
    torch.set_num_threads(2)

    print("(a) the reach of the gradient at initialization, n = 100, d = 64, 20 seeds")
    logs = reach()
    mean = {name: v.mean(0) for name, v in logs.items()}      # the geometric mean, as log10
    for name, m in mean.items():
        slope = np.polyfit(np.arange(5, 61), m[5:61], 1)[0]   # straight on a log scale: a factor per step
        print(f"  {name:9s}: |dy_n/dx_t| at n - t = 0, 1, 10, 50, 99: "
              + ", ".join(f"{10 ** m[k]:.1e}" for k in (0, 1, 10, 50, 99))
              + f";  factor per step (fit on 5..60) {10 ** slope:.3f}")
    print(f"  attention: min/max over t of the mean, n - t >= 1: "
          f"{10 ** mean['attention'][1:].min():.2e} / {10 ** mean['attention'][1:].max():.2e}; 1/n = {1 / 100:.2e}")
    print(f"  parameters: RNN {C.count_parameters(nn.RNN(D, D))}, LSTM {C.count_parameters(nn.LSTM(D, D))}, "
          f"attention {C.count_parameters(nn.MultiheadAttention(D, HEADS))}")

    print("(b) recall the first token: accuracy on 4096 test sequences (chance 12.5%)")
    results = recall()
    accuracy = {}
    for name in list(COLORS) + ["LSTM, forget bias 1"]:
        rows = [r for r in results if r["model"] == name]
        for n in sorted({r["n"] for r in rows}):
            a = [r["accuracy"] for r in rows if r["n"] == n]
            steps = [r["steps"] for r in rows if r["n"] == n]
            accuracy[name, n] = np.mean(a)
            print(f"  {name:20s} n = {n:3d}: mean {100 * np.mean(a):5.1f}%  (seeds: "
                  + ", ".join(f"{100 * v:.1f}%" for v in a) + f"; steps: {steps})")

    print("(c) sequential steps: a recurrent layer n, one attention layer 1 (every pair (t, s) at once)")
    table = timings()
    for (k, n), (t_rnn, t_attn) in table.items():
        print(f"  {k} thread{'s' if k > 1 else ' '}, n = {n:4d}: nn.RNN {1e3 * t_rnn:8.3f} ms "
              f"({1e6 * t_rnn / n:5.1f} us per step), attention {1e3 * t_attn:8.3f} ms, ratio {t_rnn / t_attn:6.1f}")
    torch.set_num_threads(2)

    # -------------------------------------------------------------------------------------
    # The figure: (a) left, (b) right, one legend
    # -------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(108 * C.MM, 38 * C.MM))
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.19, top=0.86, wspace=0.38)
    distance = np.arange(len(mean["RNN"]))
    for name, color in COLORS.items():
        axes[0].plot(distance, 10 ** mean[name], color=color, lw=1.3, label=name)
        axes[1].plot(LENGTHS, [100 * accuracy[name, n] for n in LENGTHS], "o-", color=color, ms=2.5, lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, distance[-1])
    axes[0].set_xlabel(r"distance $n-t$", fontsize=7.5, labelpad=1)
    axes[0].set_ylabel(r"$\|\partial y_n/\partial x_t\|$", fontsize=7.5)
    axes[1].axhline(100 / SYMBOLS, color=P.GREY, lw=0.8, ls=(0, (3, 2)))
    axes[1].text(LENGTHS[-1], 100 / SYMBOLS + 4, "chance", fontsize=6.5, color=P.NOTE, ha="right", va="bottom")
    axes[1].set_xscale("log")
    axes[1].set_xticks(LENGTHS, [str(n) for n in LENGTHS])
    axes[1].minorticks_off()
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("length $n$", fontsize=7.5, labelpad=1)
    axes[1].set_ylabel("test accuracy (%)", fontsize=7.5)
    for ax in axes:
        ax.tick_params(labelsize=6.5, length=2, pad=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, frameon=False,
               handlelength=1.4, columnspacing=1.5, bbox_to_anchor=(0.55, 1.03))
    C.save(fig, "recurrent_panels")
