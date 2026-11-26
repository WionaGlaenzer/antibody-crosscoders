"""Aggregate scattered per-run result JSONs (per-model SAE stats, mixed-vs-antibody comparisons)
into a single summary dict for reporting. Read-only over results/; writes the collated summary.
"""

import pandas as pd, json, os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

R = f"{BASE}/results"
out = {}
# 1) mixed SAE stats vs antibody-only
sae = {}
for name in ["prot", "unp", "pair"]:
    m = json.load(open(f"{R}/sae_{name}_L20_mixed_stats.json"))
    sae[name] = {
        "VE": round(m["val_variance_explained"], 4),
        "dead_frac": round(m["dead_fraction"], 4),
        "n_active": m["n_active"],
        "train_min": round(m.get("train_minutes", 0), 1),
    }
out["mixed_sae"] = sae
# 2) mixed crosscoder classification summary
for f in ["summary_mixed_L20.json"]:
    p = f"{R}/{f}"
    if os.path.exists(p):
        out["mixed_cc_summary"] = json.load(open(p))
# 3) 8-config labeling
g = pd.read_csv(f"{R}/concept_f1_germline.csv")
out["germline_configs"] = sorted(g["config"].unique())
out["germline_rows"] = len(g)
percfg = {}
for cfg in sorted(g["config"].unique()):
    d = g[g["config"] == cfg]
    percfg[cfg] = {
        "n": len(d),
        "germ_det_F1gt0.7": int((d["best_germline_F1"] > 0.7).sum()),
        "gained": int(d["is_gained"].sum()),
    }
out["per_config"] = percfg
json.dump(out, open(f"{R}/mixed_final_collect.json", "w"), indent=2)
print(json.dumps(out, indent=2)[:2500])
