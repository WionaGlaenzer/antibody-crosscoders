"""One-off maintenance script: rewrite the CONFIGS list in analysis/concept_labeling.py to the full
(regime, layer) grid. Asserts the old block matches verbatim before patching (safe no-op if already patched).
"""

import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

F = f"{BASE}/code/analysis/concept_labeling.py"
s = open(F).read()
old = 'CONFIGS = [\n    ("paired", 20), ("paired", 28), ("single", 20),\n]'
new = (
    "CONFIGS = [\n"
    '    ("single", 6), ("single", 12), ("single", 20), ("single", 28),\n'
    '    ("paired", 6), ("paired", 12), ("paired", 20), ("paired", 28),\n'
    "]"
)
assert old in s, "CONFIGS block not found verbatim"
s = s.replace(old, new)
open(F, "w").write(s)
print("edited OK; new CONFIGS:")
import subprocess

print(subprocess.run(["sed", "-n", "46,54p", F], capture_output=True, text=True).stdout)
