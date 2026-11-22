"""Figure of part 4 (optimization) on published training runs: learning_rate_vs_batch.

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/figures/part4_milestones.py [preview_dir]

The peak learning rate eta against the batch size B of some landmark models.  The values are
those stated in the papers cited on the slide; images/SOURCES.md quotes them.  No training: the
run takes a second.
"""
import common as C
from common import P, plt

# (name, batch size B, peak learning rate eta, label offset in points, horizontal alignment)
# B counts images, board positions (AlphaZero) or tokens (language models).  GPT-3, Llama 3 and
# DeepSeek-V3 grow the batch during training; B is the final batch, at which eta is the peak.
runs = [
    ("AlexNet (2012)",               128,          1e-2,   (6, -3),  "left"),    # Krizhevsky et al. 2012, Section 5
    ("ResNet-50 (2015)",             256,          0.1,    (-6, 0),  "right"),   # He et al. 2016, Section 3.4
    ("ResNet-50 in one hour (2017)", 8192,         3.2,    (6, 0),   "left"),    # Goyal et al. 2017
    ("AlphaZero (2017)",             4096,         0.2,    (6, -3),  "left"),    # Silver et al. 2017
    ("BERT-large (2018)",            256 * 512,    1e-4,   (0, 6),   "center"),  # Devlin et al. 2019, Appendix A.2
    ("GPT-3 (2020)",                 3.2e6,        6e-5,   (0, -9),  "center"),  # Brown et al. 2020, Table 2.1
    ("ViT-H/14 (2021)",              4096,         3e-4,   (5, 5),   "left"),    # Dosovitskiy et al. 2021, Table 3
    ("Stable Diffusion (2022)",      2048,         1e-4,   (-5, -7), "right"),   # model card of v1-4; Rombach et al. 2022
    ("Llama 3 (2024)",               16e6,         8e-5,   (6, -9),  "left"),    # Llama Team 2024, Section 3.4.1
    ("DeepSeek-V3 (2024)",           15360 * 4096, 2.2e-4, (0, 6),   "center"),  # DeepSeek-AI 2024; sequences of 4096 tokens
]
ours = (64, 1e-3)                                          # the digit classifier of lecture 02: Adam

print("batch size B, peak learning rate eta and eta / B:")
for name, batch, eta, *_ in runs + [("our digit classifier", *ours)]:
    print(f"  {name:30s} B = {batch:9.4g}  eta = {eta:7.2g}  eta/B = {eta / batch:.2g}")

fig, ax = P.panels(figsize=(4.3, 2.08))       # the width of the slide; the height left by the two items
ax.loglog([r[1] for r in runs], [r[2] for r in runs], "o", color=P.BLUE, ms=4.5)
for name, batch, eta, offset, align in runs:
    ax.annotate(name, (batch, eta), xytext=offset, textcoords="offset points", fontsize=6.5,
                ha=align, va="center")
ax.loglog(*ours, "o", mfc="white", mec=P.INK, ms=4.5)
ax.annotate("our digit classifier", ours, xytext=(0, -8), textcoords="offset points", fontsize=6.5,
            ha="center", va="center")
P.label(ax, "batch size $B$", r"learning rate $\eta$", xlim=(10, 1e9), ylim=(1e-5, 10),
        xticks=[1e1, 1e3, 1e5, 1e7, 1e9], yticks=[1e-4, 1e-2, 1])
C.save(fig, "learning_rate_vs_batch")
