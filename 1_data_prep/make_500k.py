import pyarrow.parquet as pq, pyarrow as pa, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

t = pq.read_table(f"{BASE}/data/oas_scaleup_annotated.parquet")
sub = t.slice(0, 500000)
pq.write_table(
    sub,
    f"{BASE}/data/oas_scaleup_500k.parquet",
    compression="zstd",
)
import pyarrow.compute as pc

sp = sub.column("split").to_pylist()
from collections import Counter

print("rows", sub.num_rows, "splits", dict(Counter(sp)))
