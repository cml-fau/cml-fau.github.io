"""Part 1 (tokens): byte-pair encoding learned from scratch, and the tokenizer of GPT-2 on real text.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part1_tokenizer.py [preview_dir]

A. Byte-pair encoding (Gage, 1994; Sennrich et al., 2016), learned on Shakespeare
   (notebooks/data/tinyshakespeare.txt, 1 115 394 characters, all ASCII).  The text is cut into
   words with the space before them, as GPT-2 does; every word starts as its bytes; then, again
   and again, the most frequent pair of neighbouring symbols in the text becomes a new symbol.
   - The listing of the slide "Byte-pair encoding" runs here verbatim (12 merges) and prints its
     output.  A faster version of the same algorithm (it updates the counts of the pairs instead
     of recounting them) runs until every word is one symbol; it agrees with the listing on the
     first 100 merges.
   - Printed: the first merges with their counts; how a few words are cut after 0, 10, 100,
     1000 and 10 000 merges and after all of them; tokens per character after k merges.
   - bpe_compression.pdf: tokens per character against the number of merges (log axis), on
     the text the merges were learned from and on the Universal Declaration of Human Rights in
     English and in German (texts the merges have never seen); the dots at 50 000 merges are
     the tokenizer of GPT-2 on the same three texts.  (Not used on the slide, which gives the
     numbers of the Shakespeare curve in a table.)

B. The tokenizer of GPT-2 (Radford et al., 2019): 256 bytes + 50 000 merges + 1 end-of-text
   token = 50 257 tokens.  gpt2.py (next to this file) implements it exactly.
   - Printed: how common and rare English words, numbers, a formula and German words of this
     university are cut; tokens per word in English and German on the same text (the 30
     articles of the Universal Declaration of Human Rights, and its article 1 alone);
     "strawberry" with and without the space before it; one word with and without a space or a
     capital; tokens per word on Shakespeare.
   - tokens_udhr.pdf: article 1 of the declaration in English and in German, every token a box.

Two to three minutes on one CPU thread (most of it the 21 279 merges of part A).  The two texts of
the declaration are downloaded once to lectures/05_self_attention_and_transformers/data/.
"""
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from matplotlib import font_manager, ft2font

import common as C
from common import P, np, plt, torch, nn
import gpt2

torch.set_num_threads(1)
# Karpathy's char-rnn data set, shared with notebook 15; fetched once if it is not there yet
SHAKESPEARE = gpt2.fetch("tinyshakespeare.txt",
                         "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")


# =======================================================================================
# The texts
# =======================================================================================
# The Universal Declaration of Human Rights: the official English text and the German
# translation (1996 spelling), both from the Office of the High Commissioner for Human Rights,
# as published by the "UDHR in Unicode" project (whose files are mirrored on GitHub).
UDHR = "https://raw.githubusercontent.com/behnam/unicode-udhr/master/data/udhr/"


def udhr(code):
    """The 30 articles of the declaration, one paragraph per line, without their titles, and
    article 1 alone.  The articles are parallel paragraph by paragraph in the two languages (50
    paragraphs); the preambles are not (the English one has 10 paragraphs, the German one 9)."""
    path = C.DATA / f"udhr_{code}.xml"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        request = urllib.request.Request(UDHR + path.name, headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(request).read())
    root = ET.parse(path).getroot()
    ns = root.tag[1:].split("}")[0]
    articles = [[" ".join("".join(p.itertext()).split()) for p in a.iter(f"{{{ns}}}para")]
                for a in root.iter(f"{{{ns}}}article")]
    assert len(articles) == 30 and sum(map(len, articles)) == 50
    return "\n".join(p for a in articles for p in a), " ".join(articles[0])


english, article_1_en = udhr("eng")
german, article_1_de = udhr("deu_1996")
shakespeare = SHAKESPEARE.read_text()


# =======================================================================================
# A. Byte-pair encoding, learned from scratch
# =======================================================================================
print("=" * 90, "\nA. Byte-pair encoding learned on Shakespeare\n")
t0 = time.time()

# --- the listing on the slide, verbatim (only the path of the file differs) --------------
# Reading the file as Latin-1 makes one byte one character, so the symbols are characters:
# a word is a string, the new symbols are chr(256), chr(257), ..., and a merge is str.replace,
# which replaces the pair from left to right without overlaps, as byte-pair encoding does.
text = open(SHAKESPEARE, encoding="latin-1").read()  # 1115394 bytes
words = Counter(re.findall(r" ?\w+| ?[^\w\s]+|\s+", text))  # ' the': 5370
vocab = {chr(i): bytes([i]) for i in range(256)}           # 256 bytes
for new in map(chr, range(256, 268)):                      # 12 new symbols
    pairs = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]): pairs[a + b] += n       # neighbouring pairs
    ab, count = pairs.most_common(1)[0]                    # the most frequent
    vocab[new] = vocab[ab[0]] + vocab[ab[1]]
    words = {w.replace(ab, new): n for w, n in words.items()}  # merge it
    print(vocab[new], count, end="  ")
print()
# ------------------------------------------------------------------------------------------
listing_merges = [vocab[chr(256 + k)] for k in range(12)]
chunks = Counter(re.findall(r" ?\w+| ?[^\w\s]+|\s+", shakespeare))
print(f"\n{len(shakespeare):,} characters, {len(text.encode('latin-1')):,} bytes, ascii: {shakespeare.isascii()}; "
      f"{sum(chunks.values()):,} words (with their space), {len(chunks):,} distinct; "
      f"' the' {chunks[' the']} times, 'the' {chunks['the']} times")


def learn(chunks, steps):
    """Byte-pair encoding of a text given as {word: count}: the same algorithm as the listing,
    but the counts of the pairs are updated, only for the words that contain the merged pair,
    instead of recounted.  Returns the merges (pair, count) and the length of the text in
    symbols after 0, 1, 2, ... merges.  Of two pairs with the same count, both versions take the
    one they counted first; the order of counting differs, so after the first few hundred merges,
    where many pairs occur equally often, the two can pick different ones."""
    seqs = [list(w.encode("utf-8")) for w in chunks]
    counts = list(chunks.values())
    pair_count, where = Counter(), defaultdict(set)          # where[pair]: the words that contain it
    for i, s in enumerate(seqs):
        for p in zip(s, s[1:]):
            pair_count[p] += counts[i]
            where[p].add(i)
    lengths = [sum(len(s) * n for s, n in zip(seqs, counts))]
    merges = []
    for new in range(256, 256 + steps):
        if not pair_count:
            break                                              # every word is one symbol
        best = max(pair_count, key=pair_count.get)
        merges.append((best, pair_count[best]))
        removed = 0
        for i in sorted(where.pop(best)):
            s, n = seqs[i], counts[i]
            out = []
            for x in s:                                        # merge, left to right
                if out and out[-1] == best[0] and x == best[1]:
                    out[-1] = new
                else:
                    out.append(x)
            if len(out) == len(s):
                continue
            for p in zip(s, s[1:]):
                pair_count[p] -= n
            for p in zip(out, out[1:]):
                pair_count[p] += n
                where[p].add(i)
            removed += n * (len(s) - len(out))
            seqs[i] = out
        for p in [p for p, c in pair_count.items() if c <= 0]:
            del pair_count[p]
        lengths.append(lengths[-1] - removed)
    return merges, np.array(lengths)


merges, lengths = learn(chunks, 100_000)
symbol = {i: bytes([i]) for i in range(256)}
for k, ((a, b), _) in enumerate(merges):
    symbol[256 + k] = symbol[a] + symbol[b]
learned = [symbol[256 + k] for k in range(len(merges))]
print(f"learned {len(merges):,} merges in {time.time() - t0:.0f} s, until every word is one symbol")
assert learned[:12] == listing_merges, "the listing and the fast version disagree"

# The listing again, for 100 merges and without printing: the two versions must agree.
words = {w: n for w, n in Counter(re.findall(r" ?\w+| ?[^\w\s]+|\s+", text)).items()}
check = []
for new in map(chr, range(256, 356)):
    pairs = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]): pairs[a + b] += n
    ab, count = pairs.most_common(1)[0]
    vocab[new] = vocab[ab[0]] + vocab[ab[1]]
    words = {w.replace(ab, new): n for w, n in words.items()}
    check.append((vocab[new], count))
assert check == [(symbol[256 + k], c) for k, (_, c) in enumerate(merges[:100])], "disagree on 100 merges"
print("the listing and the fast version agree on the first 100 merges (pairs and counts)")

print("\nthe first merges, with the number of times the pair occurs in the text:")
for k in range(20):
    print(f"  {k + 1:3d}  {learned[k].decode()!r:10s} {merges[k][1]:6d}")
for k in (100, 1000, 10_000, len(merges)):
    print(f"  merge {k:6,d}: {learned[k - 1].decode()!r:18s} count {merges[k - 1][1]}")


def bpe(word, k):
    """The symbols of a word after the first k merges, applied in the order they were learned.
    As GPT-2 does it: merge the pair of lowest rank, again and again.  This is the same as
    applying the k merges one after the other, since a merge creates only pairs of higher rank."""
    s = list(word.encode("utf-8"))
    while len(s) > 1:
        r, i = min((RANK.get(p, 1 << 30), i) for i, p in enumerate(zip(s, s[1:])))
        if r >= k:
            break
        a, b = s[i], s[i + 1]
        out = []
        for x in s:
            if out and out[-1] == a and x == b:
                out[-1] = 256 + r
            else:
                out.append(x)
        s = out
    return s


RANK = {pair: k for k, (pair, _) in enumerate(merges)}
STEPS = (0, 10, 100, 1000, 10_000)
print("\nhow words are cut after k merges (| between symbols):")
for word in (" gentleman", " understanding", " Erlangen", " Mathematik"):
    print(f"  {word!r} occurs {chunks[word]} times")
    for k in STEPS + (len(merges),):
        pieces = [symbol[x].decode("utf-8", errors="replace") for x in bpe(word, k)]
        print(f"    {k:6d} merges: {len(pieces):2d} symbols  {'|'.join(pieces)}")


def tokens_after(text, ks):
    """The length of the text in symbols after k merges, for every k in ks."""
    words = Counter(re.findall(r" ?\w+| ?[^\w\s]+|\s+", text))
    return np.array([sum(n * len(bpe(w, k)) for w, n in words.items()) for k in ks])


# Tokens per character: 1 at k = 0 for ASCII text (one byte per character), a little more for
# German (an umlaut is two bytes).  The three texts end at different heights after all merges
# because each word is then one symbol, and the declaration has longer words than Shakespeare.
ks = np.unique(np.round(np.logspace(0, np.log10(len(merges)), 60)).astype(int))
texts = {"Shakespeare": shakespeare, "Declaration, English": english, "Declaration, German": german}
per_char = {name: (lengths[ks] if t is shakespeare else tokens_after(t, ks)) / len(t) for name, t in texts.items()}
table_ks = (0,) + STEPS[1:] + (len(merges),)
print("\ntokens per character after k merges (bytes per character at k = 0):")
print("  k:" + " " * 23 + " ".join(f"{k:7d}" for k in table_ks))
for name, t in texts.items():
    row = (lengths[list(table_ks)] if t is shakespeare else tokens_after(t, table_ks)) / len(t)
    print(f"  {name:25s}" + " ".join(f"{v:7.3f}" for v in row))
gpt2_per_char = {name: len(gpt2.encode(t)) / len(t) for name, t in texts.items()}
print("  GPT-2 (50 000 merges):  " + ", ".join(f"{name} {v:.3f}" for name, v in gpt2_per_char.items()))
print(f"  characters per token on Shakespeare: {len(shakespeare) / lengths[1000]:.2f} after 1000 merges, "
      f"{len(shakespeare) / lengths[-1]:.2f} after all {len(merges):,}; "
      f"{1 / gpt2_per_char['Shakespeare']:.2f} with GPT-2")

fig, ax = plt.subplots(figsize=(108 * C.MM, 38 * C.MM))
colours = {"Shakespeare": P.BLUE, "Declaration, English": P.AQUA, "Declaration, German": P.ORANGE}
labels = {"Shakespeare": "Shakespeare (where the merges are learned)",
          "Declaration, English": "Declaration of Human Rights, English",
          "Declaration, German": "Declaration of Human Rights, German"}
for name, y in per_char.items():
    ax.plot(ks, y, color=colours[name], lw=1.4, label=labels[name])
    ax.plot([50_000], [gpt2_per_char[name]], "o", ms=4, color=colours[name], mec="white", mew=0.6)
ax.annotate("GPT-2", (50_000, max(gpt2_per_char.values())), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7, color=P.NOTE)
ax.set_xscale("log")
ax.set(xlim=(1, 1e5), ylim=(0, 1.08), xticks=[1, 10, 100, 1000, 10_000, 50_000], yticks=[0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["1", "10", "100", "1000", "10 000", "50 000"])
ax.minorticks_off()
ax.tick_params(labelsize=7, length=2.5, pad=2)
ax.set_xlabel("merges", fontsize=8, labelpad=2)
ax.set_ylabel("tokens per character", fontsize=8, labelpad=3)
ax.legend(loc="lower left", fontsize=7, frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25)
fig.tight_layout(pad=0.3)
C.save(fig, "bpe_compression")


# =======================================================================================
# B. The tokenizer of GPT-2
# =======================================================================================
print("\n" + "=" * 90, "\nB. The tokenizer of GPT-2\n")
tables = gpt2._load()
print(f"{len(tables['index']):,} tokens = 256 bytes + {len(tables['rank']):,} merges + 1 "
      f"(id {gpt2.END_OF_TEXT}, {gpt2.decode([gpt2.END_OF_TEXT])!r})")
assert len(tables["index"]) == 256 + len(tables["rank"]) + 1 == 50_257


def shown(i):
    """A token as text.  A token can be a piece of a character (an umlaut is two bytes in
    UTF-8, and GPT-2 merges bytes, not characters); such a token is shown as its bytes."""
    b = bytes(gpt2._byte[c] for c in tables["token"][i])
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{x:02x}" for x in b)


def show(text):
    ids = gpt2.encode(text)
    assert gpt2.decode(ids) == text
    print(f"  {text!r:44s} {len(ids):2d}  {'|'.join(shown(i) for i in ids):40s} {ids}")
    return ids


examples = {
    "common English words": [" the", " university", " language", " mathematics"],
    "rare English words": [" tokenization", " eigenvalue", " homeomorphism", " Lipschitz"],
    "numbers": [" 3.14159", " 2026", " 1234567"],
    "formulas": [" x^2 + y^2 = z^2", r" \frac{\partial u}{\partial t} = \Delta u"],
    "German words of this university": [" Friedrich-Alexander-Universität", " Prüfungsordnung",
                                        " Mathematik", " Übung", "Übung"],
    "the letters of strawberry (three r)": ["strawberry", " strawberry", "Strawberry", " Strawberry"],
    "one word, six tokens": ["the", " the", "The", " The", "THE", " THE"],
}
print("text, number of tokens, the tokens (| between them), their ids:")
for group, texts_ in examples.items():
    print(f" {group}")
    for t in texts_:
        show(t)

# Tokens per word on the same text in two languages.  A word is what lies between two
# white spaces, punctuation included ("rights." is one word).
print("\ntokens per word (words: separated by white space):")
rows = {"article 1, English": article_1_en, "article 1, German": article_1_de,
        "30 articles, English": english, "30 articles, German": german, "Shakespeare": shakespeare}
for name, t in rows.items():
    n_words, n_tokens = len(t.split()), len(gpt2.encode(t))
    print(f"  {name:22s} {len(t):8,d} characters {n_words:7,d} words {n_tokens:7,d} tokens  "
          f"{n_tokens / n_words:.2f} tokens per word  {len(t) / n_tokens:.2f} characters per token")
print(f"  German / English, 30 articles: {len(gpt2.encode(german)) / len(gpt2.encode(english)):.2f} times "
      f"the tokens for {len(german.split()) / len(english.split()):.2f} times the words")
print(f"\narticle 1, English: {article_1_en}")
print(f"article 1, German:  {article_1_de}")


# --- tokens_udhr.pdf: article 1 in both languages, one box per token ------------------------
# A monospaced font, so that a token of m characters is a box m cells wide.  The box of a token
# includes the space in front of its word, as the token does; lines break between words.
FONT = dict(family="DejaVu Sans Mono", size=8)
W, pitch, box, head = 108, 4.3, 3.5, 4.0                      # millimetres
mono = ft2font.FT2Font(font_manager.findfont(FONT["family"]))
advance = mono.load_char(ord("x"), flags=ft2font.LoadFlags.NO_SCALE).horiAdvance / mono.units_per_EM
cell = advance * FONT["size"] * 25.4 / 72                    # mm per character: 0.602 em

layout, y = [], 0.0                                           # (x, y, text, colour index) of every box
for language, t in (("English", article_1_en), ("German", article_1_de)):
    ids = gpt2.encode(t)
    layout.append((0, y + head / 2, f"{language}: {len(t.split())} words, {len(ids)} tokens", None))
    y += head + pitch / 2
    x, words_ = 0.0, []                                       # group the tokens into words: a word
    for i in ids:                                             # starts with a token that starts with
        s = shown(i)                                          # a space
        assert "\\x" not in s, "a token that is part of a character: not drawn here"
        if s.startswith(" ") or not words_:
            words_.append([])
        words_[-1].append(s)
    k = 0
    for word in words_:
        if x + len("".join(word)) * cell > W:                 # the word does not fit: next line
            x, y = 0.0, y + pitch
        for s in word:
            layout.append((x, y, s, k))
            x, k = x + len(s) * cell, k + 1
    y += pitch / 2 + 2.0
H = y - 2.0
plt.rcParams["figure.autolayout"] = False                    # the axes fill the figure
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
pastel = [P.BLUE, P.ORANGE, P.AQUA, P.YELLOW, P.MAGENTA]
for x, y, s, k in layout:
    if k is None:                                             # the heading of a language
        ax.text(x, y, s, fontsize=7.5, color=P.INK, va="center", ha="left", weight="bold")
        continue
    ax.add_patch(plt.Rectangle((x, y - box / 2), len(s) * cell, box, fc=pastel[k % 5], alpha=0.30,
                               ec="white", lw=0.6))
    ax.text(x, y + 0.08, s, fontsize=FONT["size"], family=FONT["family"], va="center", ha="left", color=P.INK)
print(f"\ntokens_udhr: {cell:.3f} mm per character, {W} x {H:.1f} mm")
C.save(fig, "tokens_udhr")
