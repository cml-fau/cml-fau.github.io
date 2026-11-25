"""Part 4 (training), pretraining then fine-tuning: GPT-2 small on public documents of FAU, on a CPU.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part4_finetune.py [preview_dir] [--train]

Without --train the script only draws the figures, from data/fau/results.json; with --train it
first runs the experiments below (44 minutes on 8 threads of a shared 16-core machine, of which
16 and 15 minutes are the two runs of 200 steps), writing every number to results.json as soon as
it is known, so that an interrupted run loses nothing and resumes where it stopped (from the
checkpoints data/fau/e2_last.pt and e3_last.pt, written every 50 steps; they can be deleted once
both runs are done).  The fine-tuned weights are data/fau/gpt2_fau.pt.  The corpus is made by part4_fau_corpus.py; the model is gpt2.py, unchanged.

E1  Before any training: the loss of pretrained GPT-2 (mean -log p of the next token, in nats per
    token, and its exponential, the perplexity) on the held-out English FAU text, on ordinary
    English that is not from FAU (five Wikipedia articles on topics unrelated to FAU, fetched
    once), on Shakespeare (notebooks/data/tinyshakespeare.txt) and on held-out German FAU text,
    with the number of tokens per word of each text.
E2  Fine-tuning: the training loop of lectures 03-04 unchanged.  The loss is the mean
    cross-entropy of the next token over a batch of B = 8 windows of n = 128 consecutive tokens
    drawn at random from the training text; AdamW (weight decay 0.01), learning rate 1e-4 after a
    linear warm-up of 20 steps, then constant, gradient norm clipped at 1; STEPS = 200 steps,
    that is 204 800 tokens, half of the training text.  The rate is the largest of the range
    considered (3e-5 to 1e-4), so that the weights move in so few steps; no other rate was tried.  The training loss of every step is recorded, the validation loss every 25 steps on a
    fixed set of 32 windows of the held-out text.
E3  The same model, data (the same batches, in the same order), optimizer and number of steps
    from a random initialization (gpt2.GPT(), seed 0).  Only the starting point changes, and the
    learning rate: 6e-4, the rate with which GPT-3 Small (125M parameters, the shape of GPT-2
    small) was pretrained (Brown et al., 2020, Table 2.1; there with batches of 0.5M tokens), as
    a rate of 1e-4, chosen for fine-tuning, is small for a network that starts from nothing.
E4  What changed: continuations of a few prompts by the model before and after fine-tuning
    (same seed, temperature 0.8, the 40 most likely tokens, 40 new tokens), and the next-token
    distribution after factual prompts (top tokens, and the probability of " Erlangen").
E5  Memorization: for every sampled continuation, the longest run of tokens that also occurs,
    verbatim, in the training text.

Figures: finetune_loss.pdf (validation loss against the step, from the two starting points),
fau_probe.pdf (the next token after one prompt, before and after fine-tuning).
"""
import json
import math
import sys
import time
import urllib.parse

import common as C
from common import P, np, plt, torch, nn
import gpt2

FAU = C.DATA / "fau"
CORPUS = FAU / "corpus"
RESULTS = FAU / "results.json"
REFERENCE = FAU / "reference"
SEPARATOR = "\n<|endoftext|>\n"               # between two documents in the corpus files

THREADS = 8
B, N = 8, 128                                 # batch of B windows of N tokens: 1024 tokens per step
MICRO = 4                                     # windows per forward pass (memory); the gradient is the same
STEPS = 200
EVAL_EVERY = 25
VAL_WINDOWS = 32                              # fixed validation windows for the curves
WARMUP = 20
RATE = {"pretrained": 1e-4, "random": 6e-4}
SEED = 0
E1_WINDOWS = 160                              # at most 160 x 128 = 20 480 tokens per text in E1

# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def tokens_of(path):
    """The token ids of a corpus file, the documents separated by GPT-2's token <|endoftext|>."""
    ids = []
    for doc in path.read_text().split(SEPARATOR):
        doc = doc.strip()
        if doc:
            ids += gpt2.encode(doc) + [gpt2.END_OF_TEXT]
    return torch.tensor(ids)


def cached_tokens(name, make, source=None):
    """Tokenizing takes a minute: keep the ids, and make them again when the text has changed."""
    path = FAU / f"tokens_{name}.pt"
    if not path.exists() or (source and source.stat().st_mtime > path.stat().st_mtime):
        torch.save(make(), path)
    return torch.load(path)


WIKIPEDIA = ["Photosynthesis", "Bicycle", "Volcano", "Chess", "Coffee"]


def wikipedia():
    """Ordinary English that is not from FAU: five Wikipedia articles, plain text, fetched once
    through the public API (the revision ids are kept, so that the text can be fetched again)."""
    path = REFERENCE / "wikipedia.json"
    if not path.exists():
        import requests
        REFERENCE.mkdir(parents=True, exist_ok=True)
        articles = {}
        for title in WIKIPEDIA:
            r = requests.get("https://en.wikipedia.org/w/api.php", timeout=30,
                             headers={"User-Agent": "text-sampler/1.0 (generic client)"},  # Wikipedia refuses library defaults
                             params=dict(action="query", prop="extracts|revisions", rvprop="ids",
                                         explaintext=1, format="json", titles=title, redirects=1))
            page = next(iter(r.json()["query"]["pages"].values()))
            articles[title] = dict(revision=page["revisions"][0]["revid"], text=page["extract"])
            time.sleep(1)
        path.write_text(json.dumps(articles, indent=1))
    articles = json.loads(path.read_text())
    # the article bodies, without the empty sections and the lists of references at the end
    texts = []
    for a in articles.values():
        body = a["text"].split("\n== See also ==")[0].split("\n== References ==")[0]
        texts.append("\n".join(line for line in body.splitlines() if line.strip() and not line.startswith("==")))
    return texts, {t: a["revision"] for t, a in articles.items()}


def windows(ids, n=N):
    """Non-overlapping windows of n + 1 tokens (n inputs, n targets)."""
    k = (len(ids) - 1) // n
    return torch.stack([ids[i * n: i * n + n + 1] for i in range(k)])


def e1_windows(ids):
    """At most E1_WINDOWS of the windows of a text, evenly spread over it."""
    w = windows(ids)
    return w[torch.linspace(0, len(w) - 1, min(len(w), E1_WINDOWS)).round().long().unique()]


@torch.no_grad()
def evaluate(model, w, batch=8):
    """Mean -log p of the next token over all positions of the windows w, in nats per token."""
    model.eval()
    total = 0.0
    for chunk in w.split(batch):
        scores = model(chunk[:, :-1])
        total += nn.functional.cross_entropy(scores.flatten(0, 1), chunk[:, 1:].flatten(), reduction="sum").item()
    model.train()
    return total / (w.shape[0] * (w.shape[1] - 1))


def batch_at(train, step):
    """The batch of step `step`: B windows at random places, the same for both starting points."""
    g = torch.Generator().manual_seed(SEED * 1_000_003 + step)
    starts = torch.randint(0, len(train) - N - 1, (B,), generator=g)
    return torch.stack([train[s: s + N + 1] for s in starts.tolist()])


def load():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save(results):
    RESULTS.write_text(json.dumps(results, indent=1))


def words(text):
    return len(text.split())


# ---------------------------------------------------------------------------------------
# The experiments
# ---------------------------------------------------------------------------------------
def e1(results, data):
    if "e1" in results:
        return
    model = gpt2.pretrained()
    out = {}
    for name, (ids, text) in data.items():
        start = time.time()
        w = e1_windows(ids)
        loss = evaluate(model, w)
        out[name] = dict(loss=loss, perplexity=math.exp(loss), tokens=int(w.shape[0] * N),
                         tokens_per_word=len(ids) / words(text), nats_per_word=loss * len(ids) / words(text))
        print(f"E1 {name:12s} {w.shape[0] * N:7d} tokens: loss {loss:.3f} nats/token, perplexity {math.exp(loss):6.1f}, "
              f"{len(ids) / words(text):.2f} tokens/word, {out[name]['nats_per_word']:.2f} nats/word "
              f"({time.time() - start:.0f} s)", flush=True)
    results["e1"] = out
    save(results)


def train(results, key, start_model, rate, train_ids, val_windows):
    """E2 and E3: STEPS steps of AdamW from start_model; resumes from data/fau/<key>_last.pt."""
    run = results.setdefault(key, dict(rate=rate, steps=STEPS, batch=B, n=N, warmup=WARMUP,
                                       train_loss=[], val_step=[], val_loss=[], seconds=0.0))
    if len(run["train_loss"]) >= STEPS and run["val_step"] and run["val_step"][-1] == STEPS:
        return
    torch.manual_seed(SEED)
    model = start_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: min(1.0, (k + 1) / WARMUP))
    checkpoint = FAU / f"{key}_last.pt"
    step = 0
    if checkpoint.exists():
        state = torch.load(checkpoint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step = state["step"]
        run["train_loss"] = run["train_loss"][:step]
        keep = [i for i, s in enumerate(run["val_step"]) if s <= step]
        run["val_step"] = [run["val_step"][i] for i in keep]
        run["val_loss"] = [run["val_loss"][i] for i in keep]
        print(f"{key}: resumed at step {step}", flush=True)
    if step == 0:
        run["val_step"], run["val_loss"] = [0], [evaluate(model, val_windows)]
        print(f"{key}: step 0, validation loss {run['val_loss'][-1]:.3f}", flush=True)
        save(results)
    while step < STEPS:
        start = time.time()
        x = batch_at(train_ids, step)
        optimizer.zero_grad()
        loss_value = 0.0
        for chunk in x.split(MICRO):
            scores = model(chunk[:, :-1])
            loss = nn.functional.cross_entropy(scores.flatten(0, 1), chunk[:, 1:].flatten()) * len(chunk) / B
            loss.backward()
            loss_value += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        run["train_loss"].append(loss_value)
        run["seconds"] += time.time() - start
        if step % EVAL_EVERY == 0 or step == STEPS:
            run["val_step"].append(step)
            run["val_loss"].append(evaluate(model, val_windows))
            print(f"{key}: step {step}, training loss {loss_value:.3f}, validation loss {run['val_loss'][-1]:.3f}, "
                  f"{run['seconds'] / step:.1f} s/step", flush=True)
            save(results)
            if step % 50 == 0 and step < STEPS:
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                schedule=schedule.state_dict(), step=step), checkpoint)
    save(results)
    return model


SAMPLE_PROMPTS = [
    "Friedrich-Alexander-Universität Erlangen-Nürnberg is",
    "The module examination consists of",
    "Learning outcomes: Students",
    "The lecture covers",
    "The Department of Mathematics",
    "The Pattern Recognition Lab",
    "The University Library",
    "In the master's programme, students",
]
PROBE_PROMPTS = [
    "FAU is a university in",
    "Friedrich-Alexander-Universität is a university in",
    "The Department of Mathematics of FAU is located in",
    "The University Library of FAU is located in",
    "The Pattern Recognition Lab is part of",
    "FAU stands for",
]
TEMPERATURE, TOP_K, NEW_TOKENS = 0.8, 40, 40


def continuations(model, seeds=(0, 1, 2)):
    out = {}
    for prompt in SAMPLE_PROMPTS:
        ids = gpt2.encode(prompt)
        out[prompt] = []
        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            new = gpt2.generate(model, ids, NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, generator=g)[len(ids):]
            out[prompt].append(dict(seed=seed, ids=new, text=gpt2.decode(new)))
    return out


CANDIDATES = [" Erlangen", " Nuremberg", " Munich", " Berlin", " Germany", " Bavaria"]


@torch.no_grad()
def word_probability(model, ids, word):
    """p(word | prompt) for a word of several tokens: the product of the probabilities of its
    tokens, each given the prompt and the tokens before it (the chain rule)."""
    logp, context = 0.0, list(ids)
    for t in gpt2.encode(word):
        logp += model(torch.tensor([context]))[0, -1].log_softmax(-1)[t].item()
        context.append(t)
    return math.exp(logp)


@torch.no_grad()
def probe(model):
    out = {}
    for prompt in PROBE_PROMPTS:
        ids = gpt2.encode(prompt)
        p = model(torch.tensor([ids]))[0, -1].softmax(-1)
        top = p.topk(10)
        out[prompt] = dict(top=[(gpt2.decode([i]), q) for q, i in zip(top.values.tolist(), top.indices.tolist())],
                           words={w: word_probability(model, ids, w) for w in CANDIDATES})
    return out


def e4(results, finetuned):
    if "e4" in results:
        return
    model = gpt2.pretrained()
    model.eval()
    finetuned.eval()
    start = time.time()
    results["e4"] = dict(settings=dict(temperature=TEMPERATURE, top_k=TOP_K, new_tokens=NEW_TOKENS, seeds=[0, 1, 2]),
                         before=continuations(model), after=continuations(finetuned),
                         probe_before=probe(model), probe_after=probe(finetuned))
    save(results)
    print(f"E4: {time.time() - start:.0f} s", flush=True)


def longest_common_run(sample, corpus_string):
    """The longest run of consecutive tokens of `sample` that occurs verbatim in the corpus.  Tokens
    are written as one character each (chr(256 + id)), so that a substring search finds exactly
    the runs of whole tokens."""
    s = "".join(chr(256 + t) for t in sample)
    best, where = 0, None
    for i in range(len(s)):
        j = i + best + 1
        while j <= len(s) and s[i:j] in corpus_string:
            best, where = j - i, (i, j)
            j += 1
    return best, where


def e5(results, train_ids):
    if "e5" in results:
        return
    corpus_string = "".join(chr(256 + t) for t in train_ids.tolist())
    out = {}
    for when in ("before", "after"):
        runs = []
        for prompt, samples in results["e4"][when].items():
            for s in samples:
                k, where = longest_common_run(s["ids"], corpus_string)
                runs.append(dict(prompt=prompt, seed=s["seed"], run=k,
                                 text=gpt2.decode(s["ids"][where[0]:where[1]]) if where else ""))
        out[when] = runs
        lengths = [r["run"] for r in runs]
        print(f"E5 {when}: longest run shared with the training text, per sample: median {np.median(lengths):.0f}, "
              f"max {max(lengths)} tokens", flush=True)
    results["e5"] = out
    save(results)


def experiments():
    torch.set_num_threads(THREADS)
    results = load()
    results["config"] = dict(B=B, N=N, steps=STEPS, eval_every=EVAL_EVERY, val_windows=VAL_WINDOWS,
                             warmup=WARMUP, rate=RATE, threads=THREADS, optimizer="AdamW, weight decay 0.01, betas (0.9, 0.999)",
                             clip=1.0, loss="mean cross-entropy of the next token (nats per token)")
    results["corpus"] = json.loads((CORPUS / "stats.json").read_text())
    start = time.time()
    corpus = {name: CORPUS / f"{name}.txt" for name in ("en_train", "en_val", "de_val")}
    train_ids, val_ids, de_ids = (cached_tokens(name, lambda path=path: tokens_of(path), path)
                                  for name, path in corpus.items())
    wiki_texts, revisions = wikipedia()
    wiki_ids = cached_tokens("wikipedia", lambda: torch.tensor(
        [t for text in wiki_texts for t in gpt2.encode(text) + [gpt2.END_OF_TEXT]]), REFERENCE / "wikipedia.json")
    shakespeare = (C.SHARED / "tinyshakespeare.txt").read_text()[:60000]
    shakespeare_ids = cached_tokens("shakespeare", lambda: torch.tensor(gpt2.encode(shakespeare)))
    results["data"] = dict(train_tokens=len(train_ids), val_tokens=len(val_ids), de_val_tokens=len(de_ids),
                           wikipedia_articles=revisions, wikipedia_tokens=len(wiki_ids),
                           shakespeare_tokens=len(shakespeare_ids))
    save(results)
    print(f"tokens: training {len(train_ids):,}, validation {len(val_ids):,}, German validation {len(de_ids):,}, "
          f"Wikipedia {len(wiki_ids):,}, Shakespeare {len(shakespeare_ids):,} ({time.time() - start:.0f} s)", flush=True)

    def text(path):
        return path.read_text().replace(SEPARATOR, " ")
    texts = {"fau_en_val": (val_ids, text(corpus["en_val"])),
             "wikipedia": (wiki_ids, " ".join(wiki_texts)),
             "shakespeare": (shakespeare_ids, shakespeare),
             "fau_de_val": (de_ids, text(corpus["de_val"]))}
    e1(results, texts)

    # the fixed validation windows of the curves: VAL_WINDOWS windows spread over the held-out text
    all_windows = windows(val_ids)
    pick = torch.linspace(0, len(all_windows) - 1, VAL_WINDOWS).round().long().unique()
    val_windows = all_windows[pick]
    results["config"]["val_windows_tokens"] = int(val_windows.shape[0] * N)

    finetuned = train(results, "e2", gpt2.pretrained, RATE["pretrained"], train_ids, val_windows)
    path = FAU / "gpt2_fau.pt"
    if finetuned is not None:
        torch.save(finetuned.state_dict(), path)
        # the fine-tuned model on all held-out text, and on Wikipedia: what it gained and lost
        results["e2"]["final"] = {name: evaluate(finetuned, e1_windows(ids)) for name, ids in
                                  (("fau_en_val", val_ids), ("wikipedia", wiki_ids), ("fau_de_val", de_ids))}
        print("E2 after fine-tuning:", results["e2"]["final"], flush=True)
        save(results)
    else:
        finetuned = gpt2.GPT()
        finetuned.load_state_dict(torch.load(path))
    e4(results, finetuned)
    e5(results, train_ids)
    del finetuned

    random_model = train(results, "e3", lambda: gpt2.GPT(), RATE["random"], train_ids, val_windows)
    if random_model is not None:
        results["e3"]["final"] = {"fau_en_val": evaluate(random_model, e1_windows(val_ids))}
        save(results)
    print(f"all experiments: {(time.time() - start) / 60:.0f} minutes", flush=True)


# ---------------------------------------------------------------------------------------
# Figures, from results.json
# ---------------------------------------------------------------------------------------
def figures():
    results = load()
    print_numbers(results)
    loss_figure(results)
    if "e4" in results:
        probe_figure(results)


def print_numbers(results):
    if "e1" in results:
        for name, r in results["e1"].items():
            print(f"E1 {name:12s} loss {r['loss']:.2f} nats/token, perplexity {r['perplexity']:.1f}, "
                  f"{r['tokens_per_word']:.2f} tokens/word, {r['nats_per_word']:.2f} nats/word")
        de, en = results["e1"]["fau_de_val"], results["e1"]["fau_en_val"]
        print(f"   German / English FAU text: tokens per word x{de['tokens_per_word'] / en['tokens_per_word']:.2f}, "
              f"nats per token x{de['loss'] / en['loss']:.2f}, nats per word x{de['nats_per_word'] / en['nats_per_word']:.2f}")
    for key in ("e2", "e3"):
        if key in results:
            r = results[key]
            print(f"{key.upper()}: rate {r['rate']}, {len(r['train_loss'])} steps, {r['seconds'] / max(1, len(r['train_loss'])):.1f} s/step, {r['seconds'] / 60:.1f} minutes of steps; "
                  f"validation loss " + ", ".join(f"{s}: {l:.2f}" for s, l in zip(r["val_step"], r["val_loss"])))
            if "final" in r:
                print(f"   after training, on the texts of E1: " + ", ".join(f"{k} {v:.3f}" for k, v in r["final"].items()))
                if "e1" in results:
                    print(f"   change from GPT-2 (nats per token): " + ", ".join(
                        f"{k} {v - results['e1'][k]['loss']:+.3f}" for k, v in r["final"].items())
                        + "; perplexity " + ", ".join(f"{k} {math.exp(results['e1'][k]['loss']):.1f} -> {math.exp(v):.1f}"
                                                      for k, v in r["final"].items()))
    print(f"log(50257) = {math.log(50257):.2f}")
    if "e4" in results:
        e4 = results["e4"]
        print(f"E4 continuations, {e4['settings']}:")
        for prompt in e4["before"]:
            for before, after in zip(e4["before"][prompt], e4["after"][prompt]):
                print(f"  [{prompt}] seed {before['seed']}\n    before: {before['text']!r}\n    after:  {after['text']!r}")
        print("E4 next token after a factual prompt (top 5; p of whole words):")
        for prompt in e4["probe_before"]:
            for when in ("probe_before", "probe_after"):
                r = e4[when][prompt]
                print(f"  [{prompt}] {when[6:]:6s} " + ", ".join(f"{t!r} {100 * q:.1f}%" for t, q in r["top"][:5])
                      + " | " + ", ".join(f"{w.strip()} {100 * q:.2g}%" for w, q in r["words"].items()))
    if "e5" in results:
        for when, runs in results["e5"].items():
            lengths = [r["run"] for r in runs]
            longest = max(runs, key=lambda r: r["run"])
            print(f"E5 {when}: longest run shared with the training text per sample: median {np.median(lengths):.0f}, "
                  f"max {max(lengths)} tokens ({longest['text']!r})")


def loss_figure(results):
    """Validation loss (dots) and training loss of every step (thin) from the two starting points,
    the loss of a uniform guess and the loss of GPT-2 before fine-tuning (dotted)."""
    fig, ax = plt.subplots(figsize=(108 * C.MM, 44 * C.MM))
    level = math.log(50257)
    labels = {"e3": "from random weights", "e2": "from GPT-2"}
    for key, color in (("e3", P.BLUE), ("e2", P.ORANGE)):
        if key not in results:
            continue
        r = results[key]
        steps = np.arange(1, len(r["train_loss"]) + 1)
        ax.plot(steps, r["train_loss"], color=color, lw=0.5, alpha=0.35)
        ax.plot(r["val_step"], r["val_loss"], color=color, lw=1.4, marker="o", ms=2.2)
        ax.annotate(f"{labels[key]}: {r['val_loss'][0]:.2f} $\\to$ {r['val_loss'][-1]:.2f}",
                    (r["val_step"][-1], r["val_loss"][-1]), xytext=(-2, 7 if key == "e3" else -7),
                    textcoords="offset points", ha="right", va="bottom" if key == "e3" else "top",
                    fontsize=7, color=color)
    ax.axhline(level, color=P.GREY, lw=0.7, ls="--", zorder=0)
    ax.text(STEPS, level + 0.15, r"uniform guess: $\log 50\,257=10.82$", ha="right", va="bottom",
            fontsize=6.5, color=P.NOTE)
    if "e2" in results:
        start = results["e2"]["val_loss"][0]
        ax.axhline(start, color=P.GREY, lw=0.7, ls=":", zorder=0)
        ax.text(STEPS * 0.36, start + 0.15, f"GPT-2 before fine-tuning: {start:.2f}", ha="left", va="bottom",
                fontsize=6.5, color=P.NOTE)
    ax.set_xlim(0, STEPS)
    ax.set_ylim(0, 12)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("step", fontsize=7, labelpad=1)
    ax.set_ylabel("loss (nats per token)", fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    C.save(fig, "finetune_loss")


def shown(token):
    """A token as text on a slide: quotes, so that its leading space shows."""
    return '"' + token.replace("\n", "\\n") + '"'


def percent(p):
    """0.123 -> 12.3%; 3.4e-7 -> 3 x 10^-5 % (mathtext)."""
    if 100 * p >= 0.1:
        return f"{100 * p:.1f}%"
    e = math.floor(math.log10(100 * p))
    return f"${100 * p / 10 ** e:.0f}\\cdot 10^{{{e}}}$%"


def probe_figure(results, prompt=None):
    """The eight most likely next tokens after the prompt, before and after fine-tuning."""
    prompt = prompt or PROBE_PROMPTS[0]
    fig, axes = plt.subplots(1, 2, figsize=(108 * C.MM, 35 * C.MM))
    largest = max(100 * results["e4"][w][prompt]["top"][0][1] for w in ("probe_before", "probe_after"))
    for ax, when, title in zip(axes, ("probe_before", "probe_after"), ("before fine-tuning", "after fine-tuning")):
        r = results["e4"][when][prompt]
        top = r["top"][:8][::-1]
        probs = [100 * q for _, q in top]
        # " Erlangen" is three tokens, " Er" + "lang" + "en": its first token is the one marked
        colors = [P.ORANGE if t == " Er" else P.GREY for t, _ in top]
        ax.barh(range(len(top)), probs, color=colors, height=0.68)
        ax.set_yticks(range(len(top)), [shown(t) for t, _ in top], fontsize=7)
        for k, q in enumerate(probs):
            ax.text(q + 0.5, k, f"{q:.1f}", va="center", fontsize=6.5, color=P.NOTE)
        ax.set_xlim(0, 1.15 * largest)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("probability (%)", fontsize=7, labelpad=1)
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.text(0.98, 0.06, f"$p$(\" Erlangen\")\n$=$ {percent(r['words'][' Erlangen'])}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=P.ORANGE, linespacing=1.3)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    C.save(fig, "fau_probe")


if __name__ == "__main__":
    if "--train" in sys.argv:
        experiments()
    if RESULTS.exists():
        figures()
    else:
        print("no results in data/fau/ yet: build the corpus with part4_fau_corpus.py --crawl, then run this "
              "script with --train (about 45 minutes on 8 CPU threads)")
