import pyarrow.parquet as pq, pyarrow as pa
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

t = pq.read_table(f"{BASE}/data/oas_scaleup_annotated.parquet")
n = t.num_rows
rng = np.random.default_rng(42)
idx = np.sort(rng.choice(n, 96477, replace=False))
sub = t.take(pa.array(idx))
pq.write_table(
    sub, f"{BASE}/data/oas_ab96k.parquet", compression="zstd"
)
from collections import Counter

print("ab96k rows", sub.num_rows, "splits", dict(Counter(sub.column("split").to_pylist())))
