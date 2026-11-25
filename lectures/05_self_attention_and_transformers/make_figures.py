"""Figures for the lecture slides self_attention_and_transformers.tex.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/make_figures.py [preview_dir] [part ...]

The figures of each part of the lecture are made by one or more scripts in figures/, named
after the part: part0_*.py (recurrent networks, the opening of part 2), part1_*.py (tokens),
part2_*.py (attention), part3_*.py (the transformer), part4_*.py (training),
part5_*.py (after pretraining) and appendix_*.py.  This driver runs them one after the
other; name parts on the command line to run only those,

    .venv/bin/python lectures/05_self_attention_and_transformers/make_figures.py part2 appendix

and each script can also be run on its own.  PDFs go to images/ next to this script; if a
preview directory is given, PNG copies are written there as well.  figures/common.py holds
what the scripts share, and figures/gpt2.py the language model: GPT-2 as on the slides, with
its tokenizer and the weights published by OpenAI (548 MB, downloaded once to notebooks/data/).
Every number quoted on the slides is printed by these scripts; images/SOURCES.md says what
each figure shows.

The experiments that train a model keep their results in data/ next to this script (not in
the repository) and reuse them: with the results there, everything runs in about five minutes.
Without them, the scripts train again (part0_recurrent.py 25 minutes, part4_scratch.py 25,
part5_verifier.py 16, appendix_vit.py 40; --train forces it for the first three, deleting
data/vit/ for the last), with one exception: the experiment on public pages of FAU is never
started from here.  It needs, once and by hand,

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part4_fau_corpus.py --crawl
    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part4_finetune.py --train

(a crawl of up to 40 minutes, then 45 minutes of training on 8 CPU threads); until then
part4_fau_corpus.py and part4_finetune.py only say so.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
arguments = sys.argv[1:]
parts = [a for a in arguments if a.startswith(("part", "appendix"))]
preview = [a for a in arguments if not a.startswith(("part", "appendix"))][:1]

scripts = sorted((HERE / "figures").glob("part*.py")) + sorted((HERE / "figures").glob("appendix*.py"))
if parts:
    scripts = [s for s in scripts if s.name.split("_")[0] in parts]

for script in scripts:
    print(f"\n=== {script.name} " + "=" * (70 - len(script.name)), flush=True)
    start = time.time()
    subprocess.run([sys.executable, str(script), *preview], check=True)
    print(f"=== {script.name}: {time.time() - start:.0f} s", flush=True)
