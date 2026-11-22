"""Figures for the lecture slides the_training_loop_ii.tex.

Run from the repository root:

    .venv/bin/python lectures/04_the_training_loop_ii/make_figures.py [preview_dir] [part ...]

The figures of each part of the lecture are made by one or more scripts in figures/, named
after the part: part1_*.py (data and representation), part2_*.py (architecture),
part3_*.py (loss), part4_*.py (optimization) and part5_*.py (the loop at scale).  This
driver runs them one after the other; name parts on the command line to run only those,

    .venv/bin/python lectures/04_the_training_loop_ii/make_figures.py part2 part4

and each script can also be run on its own.  PDFs go to images/ next to this script; if a
preview directory is given, PNG copies are written there as well.  figures/common.py holds
what the scripts share: the house style of the notebooks, the target g(x) = sin(3x) + 0.3x
and the digit classifier of lecture 02.  Every number quoted on the slides is printed by
these scripts; images/SOURCES.md says what each figure shows.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
arguments = sys.argv[1:]
parts = [a for a in arguments if a.startswith("part")]
preview = [a for a in arguments if not a.startswith("part")][:1]

scripts = sorted((HERE / "figures").glob("part*.py"))
if parts:
    scripts = [s for s in scripts if s.name.split("_")[0] in parts]

for script in scripts:
    print(f"\n=== {script.name} " + "=" * (70 - len(script.name)), flush=True)
    start = time.time()
    subprocess.run([sys.executable, str(script), *preview], check=True)
    print(f"=== {script.name}: {time.time() - start:.0f} s", flush=True)
