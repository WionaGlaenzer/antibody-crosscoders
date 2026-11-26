#!/usr/bin/env python
"""Create concept_labeling_dvj.py: adds D-gene family + subsets to L20, outputs to results/dvj/."""

import re, os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

src = open(f"{BASE}/code/analysis/concept_labeling.py").read()

# 1) redirect all OUTPUTS to a fresh subdir; keep INPUT (precomputed classification) at original
src = src.replace(
    'RESULTS = f"{BASE}/results"', 'RESULTS = f"{BASE}/results/dvj"\nRESULTS_IN = f"{BASE}/results"'
)
# the one input read must come from the original results dir
src = src.replace(
    'cls = pd.read_csv(f"{RESULTS}/classification_{tag}.csv")',
    'cls = pd.read_csv(f"{RESULTS_IN}/classification_{tag}.csv")',
)
assert "RESULTS_IN}/classification_" in src, "classification input redirect failed"

# 2) subset configs to L20 paired+single
src = re.sub(
    r"CONFIGS = \[.*?\]", 'CONFIGS = [("single", 20), ("paired", 20)]', src, count=1, flags=re.S
)

# 3) add D-gene family to build_germline_labels
old_cols = 'columns=[\n        "v_family_heavy","j_call_heavy","v_family_light","j_call_light"])'
new_cols = 'columns=[\n        "v_family_heavy","j_call_heavy","v_family_light","j_call_light","d_family_heavy"])'
assert old_cols in src, "columns block not found"
src = src.replace(old_cols, new_cols)

old_add = '    add("j_call_heavy","H")\n    add("v_family_light","L")'
new_add = (
    '    add("j_call_heavy","H")\n    add("d_family_heavy","H")\n    add("v_family_light","L")'
)
assert old_add in src, "add block not found"
src = src.replace(old_add, new_add)

os.makedirs(f"{BASE}/results/dvj", exist_ok=True)
out = f"{BASE}/code/analysis/concept_labeling_dvj.py"
open(out, "w").write(src)
print("patched OK ->", out)
print("RESULTS redirected:", "results/dvj" in src, "| d_family added:", src.count("d_family_heavy"))
