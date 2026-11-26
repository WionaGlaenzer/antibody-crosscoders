# REPRODUCE — figure-by-figure command log

Every figure in the paper (main **Fig 1–3** and supplementary **Fig S1–S7**,
**Tables S1–S4**) traced end-to-end: **figure ← plotter ← analysis/​training script
← checkpoint ← activation store (`.h5`) ← dataset (`.parquet`)**.

Paths are relative to the repo root; scripts resolve
`BASE = /cluster/project/reddy/wglaenzer/crosscoders`, with
`ACTS=$BASE/activations`, `CKPT=$BASE/checkpoints`, `RES=$BASE/results`,
`DATA=$BASE/data`.

## Environment (once)

```bash
source <lmod-init>                                              # see compute_details
module load eth_proxy stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/12.1.1
source $BASE/../environments/crosscoder/bin/activate           # frozen: requirements.txt
export HF_HOME=$BASE/caches/hf
# mmseqs for stage-1 clustering: module load openmpi/4.1.6 mmseqs2/14-7e284
```

GPU jobs on Euler: `sbatch -A es_reddy -p gpu.24h --gpus=1 --gres=gpumem:24g
--cpus-per-task=4 --mem-per-cpu=16g` (training needs `--mem-per-cpu=90g`; put
`#SBATCH` directives at the TOP of the job script, not via a scheduler= arg).

---

## Dataset layer (stage 1) — inputs shared by everything

| dataset (`data/`) | built by | notes |
|---|---|---|
| `oas_full_annotated.parquet` (40,985 seq) | *pilot-scale predecessor of* `1_data_prep/build_oas_scaleup.py` | legacy input, see README provenance note; used by all `full_*` / behavioral / SHM / steering analyses |
| `oas_scaleup_annotated.parquet` (2.48M seq) | `build_oas_scaleup.py <out> <summary>` | full OAS-paired, parse→dedup→region-label→mmseqs CDR3 cluster (0.8/0.8)→70/15/15 by cluster |
| `oas_pool_95nr.parquet` | `build_oas_pool.py` | whole-seq linclust 95%-NR pool |
| `general_pool_95nr.parquet` | `build_general_pool.py` | Swiss-Prot 95%-NR (non-Ig/TCR) |
| `general_protein_96k.parquet` | `rebuild_general_96k.sh` | 96,477 declustered reps |
| `exp_{paired,unpaired,mixed}_small.parquet` | `draw_experiment_subsets.py` | size-matched nested subsets (main-figure data) |
| `exp_{paired,unpaired,mixed}_scale.parquet` | `draw_experiment_subsets.py` / `regen_scale_subsets.py` | 10× scale-up subsets (Table S1) |

## Activation layer (stage 2) — `activations/*.h5`, each `(N,3,1024)` fp16

| store | extractor | from parquet |
|---|---|---|
| `full_{paired,single}_L{6,12,20,28}.h5` | `extract_activations.py` | `oas_full_annotated.parquet` |
| `exp_{paired,unpaired,mixed}_small_L20.h5` | `extract_L20.py` / `extract_mixed_L20.py` | `exp_*_small.parquet` |
| `exp_paired_small_L{1..30}.h5` | `extract_all_layers_paired.py` | `exp_paired_small.parquet` |
| `exp_{paired,unpaired,mixed}_scale_L20.h5` | `extract_scaleup.py` | `exp_*_scale.parquet` |
| `general_single_L20.h5`, `mixed_single_L20.h5` | `general_extract.py`, `build_mixed.py` | general / mixed parquet |

The extractor names above are the entry points; `extract_scaleup.py` / `extract_single_96k.py` /
`extract_L20.py` are now thin wrappers over the consolidated `extract_activations.py` — see the
**Extractor consolidation (arg mapping)** table below for the exact `--layers`/`--regime` each uses.

Extraction (example, single (regime,layer) covers L6/12/20/28 in one run):
```bash
python 2_extraction/extract_activations.py \
  --data $DATA/oas_full_annotated.parquet --out-dir $ACTS \
  --manifest $RES/extract_manifest.json --npz $RES/extract_norm.npz \
  --log $BASE/logs/extract.log --tag full
```

### Extractor consolidation (arg mapping)

The former near-identical extractor copies now share `extractor_core.py` (model +
tokenizer load with the vocab-identity assertion, `RunningStats` train mean/std,
per-residue stacking, sanity check). The three 354-line clones collapsed into one
CLI-driven `extract_activations.py --layers <L,..> --regime {single,paired,both}`;
the old names are kept as thin wrappers so each store's provenance stays explicit:

| store built | command (equivalent) |
|---|---|
| `full_{single,paired}_L{6,12,20,28}` | `extract_activations.py --layers 6,12,20,28 --regime both` (defaults) |
| `*_scale_L20` (paired) | `extract_scaleup.py` == `extract_activations.py --layers 20 --regime paired` |
| 96k single L20 | `extract_single_96k.py` == `extract_activations.py --layers 20 --regime single` |
| `exp_*_small_L20` | `extract_L20.py` == `extract_activations.py --layers 20 [--regime ..]` |

`extract_mixed_L20.py`, `general_extract.py` (+ `general_extract_96k.py` wrapper) and
`extract_all_layers_paired.py` keep their distinct data-loading/schema logic but import
the shared core. Refactor verified: on a GPU smoke test the refactored extractors produce
**bit-identical** h5 output (every dataset + attrs) to the pre-refactor versions
(`extract_activations` all 8 single/paired x L6/12/20/28; `extract_mixed_L20`; `general_extract`).


## Training layer (stage 3) — `checkpoints/*.pt`

Main size-matched crosscoders (seeds 0,1,2 for replication):
```bash
for regime in paired unpaired mixed; do for seed in 0 1 2; do
  python 3_training/train_crosscoder.py \
    --h5 $ACTS/exp_${regime}_small_L20.h5 --regime ${regime} --layer 20 \
    --ckpt_dir $CKPT --results_dir $RES/per_run \
    --tag ${regime}_small_s${seed} --seed ${seed} \
    --dict_size 32768 --k 32 --batch_size 4096 --steps 32000
done; done
# → checkpoints/cc_{regime}_small_s{seed}_L20.pt  (main-figure checkpoints)
```
Locked config (README): batch 4096, k 32, dict 32768 (exp 32), num_layers 3,
~32000 steps, lr auto 1.414e-4, warmup 1000, auxk_alpha 0.03125,
code_normalization CROSSCODER, target_rms 1.0.

---

# MAIN FIGURES

## Figure 1 — crosscoder overview
- **1a, 1b**: schematics (no data script).
- **1c** VE across layers {4,12,20,28}, paired: `4_analysis/velayer_recover.py`
  ← `cc_paired_velayer_L{4,12,20,28}.pt` ← `exp_paired_small_L{4,12,20,28}.h5`.
- **1d** VE at L20 by regime + **1e** active-latent %:
  `4_analysis/velayer_recover.py` / `mixed_firing_3seed_topk.py`
  ← `cc_{unpaired,paired,mixed}_small_s0_L20.pt` ← `exp_*_small_L20.h5`.
```bash
python 4_analysis/velayer_recover.py           # writes $RES/decnorm_*_velayer_L*.npz, VE table
python 5_figures/plot_ve_ternary_rebuild95.py # → Fig 1c/1d panels (reads velayer_decnorm_all.npz)
```

## Figure 2 — gained latents
- **2a** relative decoder-norm simplex (3 regimes, L20):
  `5_figures/plot_fig2a_simplex.py` / `plot_regime_simplex.py`
  ← `decnorm_{regime}_small_s0_L20.npz` (from `classify_only.py` on
  `cc_{regime}_small_s0_L20.pt`).
- **2b/2c** domain specificity + %-specific bars:
  `4_analysis/mixed_firing_3seed_topk.py` (3 seeds)
  ← `cc_mixed_small_s{0,1,2}_L20.pt` ← `exp_mixed_small_L20.h5`
  → `mixed_firing_3seed_topk.npz` → `plot_fig2bcd_fig3abc_panels.py`.
- **2d** gained-at-paired across three PAIRED-regime crosscoders (3-model blue /
  2-model orange / delta-crosscoder red), L20:
  ← `cc_paired_small_s{0,1,2}_L20.pt` (3-model) + `cc2_paired_rebuild95_L20.pt`
  (2-model) + `delta_paired_rebuild95_L20.pt` (delta); `fig2d_rpair_perlatent.csv`.
```bash
# per-regime classification (writes decnorm_{tag}.npz used by 2a):
for regime in unpaired paired mixed; do
  python 4_analysis/classify_only.py --tag ${regime}_small_s0_L20 \
    --h5 $ACTS/exp_${regime}_small_L20.h5 \
    --ckpt $CKPT/cc_${regime}_small_s0_L20.pt \
    --jkminder_repo $BASE/code/dictionary_learning_jkminder --out_dir $RES/per_run
done
python 5_figures/plot_fig2a_simplex.py
python 4_analysis/mixed_firing_3seed_topk.py       # → mixed_firing_3seed_topk.npz
python 5_figures/plot_fig2bcd_fig3abc_panels.py   # → Fig 2b/2c/2d + Fig 3a/b/c panels
```

## Figure 3 — interpretable features
- **3a/3b** region detectors + top-100 F1 (3 seeds, paired):
  `4_analysis/fig3_region_reuse.py` ← `cc_paired_small_s{0,1,2}_L20.pt`
  ← `exp_paired_small_L20.h5` → `fig3_perlatent_paired_s{seed}.npz`.
- **3c** region-detector firing freq (mixed): `4_analysis/fig3c_region_firing.py`
  ← `cc_mixed_small_s0_L20.pt` + `exp_mixed_small_L20.h5` +
  `exp_mixed_small.parquet` + `oas_pool_95nr.parquet` → `fig3c_region_firing.npz`.
- **3d** example-chain trace: `4_analysis/fig3d_mixed_trace.py` / `fig3d_toplat_profile.py`.
- **3e/3f/3g** decoder UMAP + concept clouds: `4_analysis/umap_mixed_analysis.py`
  ← `cc_mixed_small_s0_L20.pt` + `concept_labels_general95.npz`
  (from `concept_f1_folds_rebuild95v2.py`) → `umap_mixed_rebuild95.npz`.
```bash
python 4_analysis/fig3_region_reuse.py     # seeds 0,1,2 → fig3_perlatent_paired_s*.npz
python 4_analysis/fig3c_region_firing.py   # → fig3c_region_firing.npz
python 4_analysis/concept_f1_folds_rebuild95v2.py   # → concept_labels_general95.npz + concept F1 (Fig 3g, Table)
python 4_analysis/umap_mixed_analysis.py   # → umap_mixed_rebuild95.npz
python 4_analysis/fig3d_mixed_trace.py
python 5_figures/plot_fig2bcd_fig3abc_panels.py     # tiles Fig 3a/b/c (see Fig 2)
```

---

# SUPPLEMENTARY FIGURES & TABLES

## Fig S1 — VE across layers per regime (a unpaired, b paired, c mixed)
`4_analysis/velayer_recover.py` ← `cc_{regime}_velayer_L{4,12,20,28}.pt`
← `exp_{regime}_small_L{4,12,20,28}.h5` → `$RES/decnorm_{regime}_velayer_L*.npz`
+ VE table (`velayer_decnorm_all.npz`); plotted by `plot_ve_ternary_rebuild95.py`.
(`plot_alllayer_overview.py` plots the 30-layer VE for Fig S4 from
`figdata/alllayer_overview.csv`.)

## Fig S2 — hyperparameter sweep (expansion factor, BatchTopK K)
Checkpoints `cc_single{4k,8k,16k}.pt` (expansion) and
`cc_singlek{8,16,64,128}.pt` (K), all L20 unpaired, trained via
`train_crosscoder.py --dict_size … / --k …`. VE + n_active read from each run's
`summary_{tag}.json`. (Note: this figure used the earlier CDR3-only clustering,
per its caption.)

## Fig S3 — relative decoder norms across layers {4,12,20,28} × 3 regimes
`4_analysis/classify_only.py` on `cc_{regime}_velayer_L{4,12,20,28}.pt`
→ `decnorm_*.npz`; plotted by `plot_regime_simplex.py`.

## Fig S4 — 2-model crosscoders, all 30 layers (+ Delta-Crosscoder)
- **a VE / b gained-at-paired across L1–30**: `4_analysis/alllayer_decnorm_dist.py`
  ← `cc2_paired_alllayer_L{1..30}_last.pt` ← `exp_paired_small_L{1..30}.h5`
  → `alllayer_decnorm_dist.npz` → `plot_alllayer_rpair.py` / `plot_alllayer_overview.py`.
  Checkpoints from `train_crosscoder_2model.py` per layer.
- **Delta-Crosscoder** (text/failure analysis): `3_training/train_delta_crosscoder.py`
  → `checkpoints/delta_paired_rebuild95.pt` + `delta_summary_*.json`.
```bash
for L in $(seq 1 30); do
  python 3_training/train_crosscoder_2model.py \
    --h5 $ACTS/exp_paired_small_L${L}.h5 --regime paired --layer ${L} \
    --ckpt_dir $CKPT --tag paired_alllayer_L${L}   # note: writes cc2_paired_alllayer_L{L}
done
python 4_analysis/alllayer_decnorm_dist.py   # → alllayer_decnorm_dist.npz
python 5_figures/plot_alllayer_rpair.py     # → Fig S4
```

## Fig S5 — V-gene detector latents
`4_analysis/vgene_level_detection.py` ← `cc_paired_L20.pt` +
`full_paired_L20.h5` + `oas_full_annotated.parquet` (`v_call_heavy`);
one-vs-rest max-F1 per gene, Haque floor `2π/(1+π)+0.2`, 5-fold CV.

## Fig S6 — J-gene detector latents
Same gene-detection pipeline (`vgene_level_detection.py` on `j_call_heavy`, and
`concept_f1_folds_rebuild95v2.py` for the J-gene concept F1); `steering_ighj6.py` for the
IGHJ6 steering panel. Checkpoint `cc_paired_L20.pt`.

## Fig S7 — conserved-residue detector latents
`4_analysis/conserved_residues.py` (+ `conserved_profile.py`)
← `cc_paired_L20.pt` + `full_paired_L20.h5` + `oas_full_annotated.parquet`.
Concepts = (amino-acid identity + region): framework Cys (disulfide), FR2 Trp,
FR4 Gly (WGxG); InterPLM max-F1 sweep with prevalence floor.

## Table S1 — small vs scaled-up (10×) crosscoders, L20
`train_crosscoder.py` on `exp_{regime}_scale_L20.h5` (`cc_{regime}_scale.pt`) vs
small; metrics from each `summary_{tag}.json`. Uses `train_crosscoder_scale2shard.py`
for the RAM-fitting two-shard single-epoch scale-up.

## Table S2 / S3 — latent categorization across 3 seeds
`classify_only.py` on `cc_{regime}_small_s{0,1,2}_L20.pt`; category counts
mean±sd over seeds aggregated by `4_analysis/collect.py`.

## Latent scaling (A.1.6, feeds Fig S4 "only red survive")
Implemented inline in `train_crosscoder.py` / `classify_only.py`
(Minder 2504.02922 Eq.4/33–36); ν = β_base/|β_target|, artifact if ν ≥ 0.5.

---

## Notes / caveats for a re-runner
1. `oas_full_annotated.parquet` is a frozen legacy input (see README provenance
   note) — not bit-reproducible; `build_oas_scaleup.py … 400` regenerates an
   equivalent file.
2. `*.npz`/CSV intermediates are written to the **job cwd**; the committed
   `5_figures/figdata/*.csv` are the plain-text copies of the exact panel inputs
   (round-trip verified) — plots regenerate from those without recompute.
3. Encode convention differs by panel: `use_threshold=False` (BatchTopK) for
   domain-firing density (Fig 2b/c), `use_threshold=True` for region/concept
   detectors (Fig 3, S5–S7). Documented in `figdata/README.md`.
4. mmseqs clustering + split RNG make dataset/label rebuilds equivalent, not
   bit-identical.
