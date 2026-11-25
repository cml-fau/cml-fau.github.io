"""GPT-2, as on the slides of self_attention_and_transformers.tex: the model, its tokenizer and
the weights published by OpenAI.

    import gpt2
    ids = gpt2.encode("The dog bites the man.")          # [464, 3290, 26081, 262, 582, 13]
    model = gpt2.pretrained()                            # GPT-2 small, 124 439 808 parameters
    scores = model(torch.tensor([ids]))                  # (1, 6, 50257): one row per position
    print(gpt2.decode(gpt2.generate(model, ids, 20, temperature=0)))

The classes Block and GPT are the code of the slides "The block, in code" and "GPT-2, in
code", line by line.  They are written to be read: the attention is spelled out instead of
calling torch.nn.functional.scaled_dot_product_attention, and generate() runs the whole model
again for every new token instead of caching the keys and values.  Run this file to check the
tokenizer and the weights against known values:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/gpt2.py

Files, all in notebooks/data/ (downloaded once, not part of the repository):
    gpt2-encoder.json, gpt2-vocab.bpe   the tokenizer (Radford et al., 2019), from OpenAI
    gpt2-model.safetensors              the weights of GPT-2 small, from the Hugging Face hub
"""
import json
import pathlib
import struct
import unicodedata
import urllib.request

import numpy as np
import torch
from torch import nn

SHARED = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "data"
OPENAI = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/"
WEIGHTS = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
END_OF_TEXT = 50256                                     # the one special token: it separates documents


def fetch(name, url):
    path = SHARED / name
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)             # Q, K, V of all heads
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU("tanh"),
                                 nn.Linear(4 * d, d))

    def attention(self, x):                        # x: (batch, n, d)
        B, n, d = x.shape
        split = lambda u: u.view(B, n, self.heads, -1).transpose(1, 2)
        q, k, v = map(split, self.qkv(x).split(d, dim=-1))
        scores = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5
        future = torch.ones(n, n).triu(1).bool()   # True where s > t
        a = scores.masked_fill(future, -torch.inf).softmax(-1)
        return self.out((a @ v).transpose(1, 2).reshape(B, n, d))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))        # tokens exchange information
        return x + self.mlp(self.ln2(x))           # each token on its own


class GPT(nn.Module):
    def __init__(self, vocab=50257, context=1024, d=768, layers=12, heads=12):
        super().__init__()
        self.tokens = nn.Embedding(vocab, d)       # e_i, one row per token
        self.positions = nn.Embedding(context, d)  # p_t, learned in GPT-2
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        initialize(self)                           # N(0, 0.02^2), biases 0

    def forward(self, i):                          # i: (batch, n) token ids
        x = self.tokens(i) + self.positions(torch.arange(i.shape[1]))
        x = self.ln(self.blocks(x))
        return x @ self.tokens.weight.T            # scores: (batch, n, vocab)


def initialize(model, std=0.02):
    """Every matrix N(0, 0.02^2), every bias 0, layer norms the identity: the initialization of GPT-2,
    without the factor 1/sqrt(2L) that its paper puts on the last matrix of each attention and MLP.
    (PyTorch's default would draw the embeddings from N(0, 1); as the same matrix reads the scores
    out, the first loss would be far above log(50257) = 10.82.)"""
    for name, p in model.named_parameters():
        if p.dim() == 2:
            nn.init.normal_(p, std=std)
        elif name.endswith("bias"):
            nn.init.zeros_(p)


def pretrained():
    """GPT-2 small with the weights of OpenAI, read from the safetensors file by hand: 8 bytes with
    the length of a JSON header, the header (name -> dtype, shape, byte range), then the numbers."""
    path = fetch("gpt2-model.safetensors", WEIGHTS)
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)

    def tensor(name, transpose=False):
        begin, end = header[name]["data_offsets"]
        t = torch.from_numpy(np.frombuffer(data[begin:end], dtype="<f4").reshape(header[name]["shape"]).copy())
        return t.T if transpose else t                  # OpenAI stores W for x @ W, nn.Linear wants W for W @ x

    model = GPT()
    names = {"tokens.weight": "wte.weight", "positions.weight": "wpe.weight", "ln.weight": "ln_f.weight", "ln.bias": "ln_f.bias"}
    for l in range(len(model.blocks)):
        for ours, theirs in (("ln1", "ln_1"), ("ln2", "ln_2"), ("qkv", "attn.c_attn"), ("out", "attn.c_proj"),
                             ("mlp.0", "mlp.c_fc"), ("mlp.2", "mlp.c_proj")):
            names[f"blocks.{l}.{ours}.weight"] = f"h.{l}.{theirs}.weight"
            names[f"blocks.{l}.{ours}.bias"] = f"h.{l}.{theirs}.bias"
    state = {ours: tensor(theirs, transpose=ours.endswith("weight") and (".qkv." in ours or ".out." in ours or ".mlp." in ours))
             for ours, theirs in names.items()}
    model.load_state_dict(state)                        # strict: every parameter is set, none is left over
    return model.eval()


@torch.no_grad()
def generate(model, ids, steps, temperature=1.0, top_k=None, generator=None):
    """Append `steps` tokens to the list `ids`, one at a time: sample the next token from
    softmax(scores / temperature), restricted to the top_k most likely tokens if top_k is given;
    temperature = 0 takes the most likely token."""
    ids = list(ids)
    context = model.positions.num_embeddings
    for _ in range(steps):
        scores = model(torch.tensor([ids[-context:]]))[0, -1]
        if temperature == 0:
            ids.append(scores.argmax().item())
            continue
        scores = scores / temperature
        if top_k:
            scores[scores < scores.topk(top_k).values[-1]] = -torch.inf
        ids.append(torch.multinomial(scores.softmax(-1), 1, generator=generator).item())
    return ids


@torch.no_grad()
def attention_weights(model, i):
    """The weights a_ts of every head of every block for the token ids i of shape (batch, n):
    a tensor (batch, layers, heads, n, n) whose rows sum to 1.  The same lines as Block.attention."""
    x = model.tokens(i) + model.positions(torch.arange(i.shape[1]))
    maps = []
    for block in model.blocks:
        h = block.ln1(x)
        B, n, d = h.shape
        q, k, _ = (u.view(B, n, block.heads, -1).transpose(1, 2) for u in block.qkv(h).split(d, dim=-1))
        scores = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5
        maps.append(scores.masked_fill(torch.ones(n, n).triu(1).bool(), -torch.inf).softmax(-1))
        x = block(x)
    return torch.stack(maps, dim=1)


# ---------------------------------------------------------------------------------------
# The tokenizer: byte-pair encoding with the 50 000 merges of GPT-2
# ---------------------------------------------------------------------------------------
# encoder.json maps each of the 50 257 tokens to its index; vocab.bpe lists the 50 000 merges
# in the order in which they were learned.  GPT-2 writes every byte as one printable
# character (a space becomes "Ġ"), so that a token is a string.
_printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
_symbol = {b: chr(b) for b in _printable}
_symbol.update({b: chr(256 + k) for k, b in enumerate(b for b in range(256) if b not in _printable)})
_byte = {s: b for b, s in _symbol.items()}
_tables = {}


def _load():
    if not _tables:
        index = json.loads(fetch("gpt2-encoder.json", OPENAI + "encoder.json").read_text(encoding="utf-8"))
        merges = fetch("gpt2-vocab.bpe", OPENAI + "vocab.bpe").read_text(encoding="utf-8").split("\n")[1:-1]
        _tables.update(index=index, token={i: t for t, i in index.items()},
                       rank={tuple(m.split()): k for k, m in enumerate(merges)}, cache={})
    return _tables


def _kind(c):
    """Letter, Number, Space or Other: the four classes of characters that GPT-2 tells apart."""
    if c.isspace():
        return "S"
    category = unicodedata.category(c)[0]
    return category if category in "LN" else "O"


def pieces(text):
    """Cut the text into the pieces that are encoded one by one.  This is the pattern of GPT-2,
        's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
    written as a loop because Python's re has no \\p{L}: a run of letters, of digits or of other
    signs, each with the one space before it; contractions; and what is left of the white space."""
    kinds = [_kind(c) for c in text]
    out, i, n = [], 0, len(text)
    while i < n:
        contraction = next((c for c in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d") if text.startswith(c, i)), None)
        if contraction:
            j = i + len(contraction)
        else:
            start = i + 1 if text[i] == " " and i + 1 < n and kinds[i + 1] != "S" else i    # the optional space
            j = start
            while j < n and kinds[j] == kinds[start]:
                j += 1
            if kinds[start] == "S" and j < n and j - i > 1:
                j -= 1                                  # \s+(?!\S): the last space belongs to the next word
        out.append(text[i:j])
        i = j
    return out


def byte_pair_encode(piece):
    """Start from the bytes of the piece and apply the learned merges, earliest first."""
    rank, cache = _load()["rank"], _load()["cache"]
    if piece in cache:
        return cache[piece]
    parts = [_symbol[b] for b in piece.encode("utf-8")]
    while len(parts) > 1:
        best = min(zip(parts[:-1], parts[1:]), key=lambda pair: rank.get(pair, float("inf")))
        if best not in rank:
            break
        merged, i = [], 0
        while i < len(parts):
            if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                merged.append(parts[i] + parts[i + 1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        parts = merged
    cache[piece] = parts
    return parts


def tokenize(text):
    """The tokens of the text as strings, with "Ġ" for a space: tokenize("The dog") = ["The", "Ġdog"]."""
    return [token for piece in pieces(text) for token in byte_pair_encode(piece)]


def encode(text):
    index = _load()["index"]
    return [index[token] for token in tokenize(text)]


def decode(ids):
    token = _load()["token"]
    return bytes(_byte[s] for i in ids for s in token[int(i)]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------------------
# Checks: the tokenizer and the weights against known values
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    assert encode("Hello world") == [15496, 995] and encode("Hello, world!") == [15496, 11, 995, 0]
    sentence = "The dog bites the man."
    print(f'"{sentence}" -> {tokenize(sentence)} -> {encode(sentence)}')
    for text in ("Prüfungsordnung der Friedrich-Alexander-Universität", "x  =  3.14159\n\n  y", "It's theirs, isn't it?"):
        assert decode(encode(text)) == text
        print(f"{text!r} -> {tokenize(text)}")

    model = pretrained()
    counts = {"token embedding (also the read-out)": model.tokens.weight.numel(),
              "position embedding": model.positions.weight.numel(),
              "attention, 12 blocks": sum(p.numel() for b in model.blocks for m in (b.qkv, b.out) for p in m.parameters()),
              "MLP, 12 blocks": sum(p.numel() for b in model.blocks for p in b.mlp.parameters()),
              "layer norms": sum(p.numel() for b in model.blocks for m in (b.ln1, b.ln2) for p in m.parameters())
                             + sum(p.numel() for p in model.ln.parameters())}
    total = sum(p.numel() for p in model.parameters())
    assert total == sum(counts.values()) == 124_439_808
    print(f"\nGPT-2 small: {total:,} parameters")
    for name, count in counts.items():
        print(f"  {name:38s}{count:>12,}  {100 * count / total:5.1f}%")

    for prompt in ("The capital of Germany is", "The dog bites the", "Attention is all you"):
        ids = encode(prompt)
        p = model(torch.tensor([ids]))[0, -1].softmax(-1)
        top = p.topk(5)
        print(f'\n"{prompt}" -> ' + ", ".join(f"{decode([i])!r} {100 * q:.1f}%" for q, i in zip(top.values.tolist(), top.indices.tolist())))
        print("  greedy:", repr(decode(generate(model, ids, 12, temperature=0))))
    a = attention_weights(model, torch.tensor([encode(sentence)]))
    print(f"\nattention weights: shape {tuple(a.shape)}, rows sum to 1 up to {(a.sum(-1) - 1).abs().max():.1e}, "
          f"largest weight on a future token {a[0, :, :].triu(1).max():.1e}")
    text = (SHARED / "tinyshakespeare.txt").read_text()[:20000] if (SHARED / "tinyshakespeare.txt").exists() else sentence * 50
    ids = torch.tensor([encode(text)[:513]])
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(ids[:, :-1])[0], ids[0, 1:])
    print(f"mean -log p of the next token on {ids.shape[1] - 1} tokens of Shakespeare: {loss:.3f} nats (uniform: {np.log(50257):.2f})")
