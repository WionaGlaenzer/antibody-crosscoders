# Sparse crosscoders identify antibody features learned during fine-tuning of protein language models

Code for the paper [*Sparse crosscoders identify antibody features learned during fine-tuning of protein language models*](https://openreview.net/forum?id=BwUX3MoiAg).
In this paper, we study fine-tuning of protein language models for the antibody domain using
crosscoders. We compare the ProtBERT, IgBERT-unpaired and IgBERT-paired models.

## Environment

Python 3.11 with the packages in `requirements.txt`, plus MMseqs2 (data preparation) and
HMMER (for ANARCI). Extraction and training need a GPU with ≥24 GB memory.

**On Euler**
```bash
module load stack/2024-06 python/3.11.6 eth_proxy
python -m venv ~/venvs/crosscoder && source ~/venvs/crosscoder/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
module load gcc/12.2.0 openmpi/4.1.6 mmseqs2/14-7e284
```
HMMER has no Euler module; build it from source and add it to your `PATH`.

**Elsewhere**
```bash
conda create -n crosscoder -c conda-forge -c bioconda python=3.11 mmseqs2 hmmer
conda activate crosscoder
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

**Paths.** Copy `config.example.yaml` to `config.yaml` and set your directories. Without a
`config.yaml`, the paper's paths in `config.example.yaml` are used.

## Pipeline

### 1_data_prep — build datasets
Datasets used in the paper (95% non-redundant, size-matched by residue count):

| script | output |
|---|---|
| `build_oas_scaleup.py` | `oas_scaleup_annotated.parquet`: paired OAS sequences, parsed, deduplicated, with region labels (its CDR3-based split is not used) |
| `step1_filter.py` | `filtered.fasta`: Swiss-Prot without Ig/TCR, length 20–512, deduplicated |
| `build_oas_pool.py` / `build_general_pool.py` | 95%-non-redundant pools (MMseqs2 linclust, 95% identity, 80% coverage; OAS clustered on concatenated heavy+light) |
| `pool_converge.py` | re-clusters both pools until no two representatives are ≥95% identical |
| `draw_experiment_subsets.py` | `exp_{unpaired,paired,mixed}_{small,scale}.parquet`: uniform samples (seed 42) with the same residue budget per regime, split 70/15/15 train/val/test |
| `regen_scale_subsets.py` | regenerates the scale-up subsets at 10× the residue budget |
| `metadata_labeling.py` | isotype/BType/disease/vaccine latent labels |
| `verify_split.py` | checks no cluster spans train/val/test |

`make_500k.py`, `make_ab96k.py`, `make_ab316k.py`, `step2_build.py`, `rebuild_general_96k.sh` and
`build_mixed*.py` build earlier, non-size-matched datasets.

### 2_extraction — token-aligned per-residue activations (N,3,1024) fp16
All extractors share `extractor_core.py` (model + tokenizer load with vocab-identity
assertion, `RunningStats` train mean/std, per-residue 3-model stacking, sanity check).
The former near-identical clones are now one CLI extractor plus thin wrappers.

| script | regime | notes |
|---|---|---|
| `extract_activations.py` | `--layers <L,..> --regime {single,paired,both}` | main 3-model extractor; heavy`[SEP]`light for paired. Defaults `6,12,20,28`/`both` reproduce the original run |
| `extract_scaleup.py` | paired, L20 | wrapper: `extract_activations.py --layers 20 --regime paired` |
| `extract_single_96k.py` | single, L20 | wrapper: `--layers 20 --regime single` |
| `extract_L20.py` | single+paired, L20 | wrapper: `--layers 20 [--regime ..]` |
| `extract_mixed_L20.py` | mixed | imports core; own mixed-parquet schema (antibody+general, `kind` column) |
| `extract_all_layers_paired.py` | paired, L1–30 | imports core; all 30 layers |
| `general_extract.py` / `_96k.py` | single (general proteins have no pairing) | imports core; writes `general_*_single_L20.h5` |
| `build_mixed.py` | single | 50/50 antibody+general mixed store |
| `verify_full.py` | — | post-extraction QC of an activation `.h5` (row count, split sizes, model order, norm stats) |

### 3_training — BatchTopK crosscoders / SAEs (jkminder fork)
| script | what |
|---|---|
| `train_crosscoder.py` | 3-model joint crosscoder (main); argparse: --h5 --regime --layer --dict_size --k --steps --seed |
| `train_crosscoder_2model.py` | 2-model unpaired↔paired (sensitivity) |
| `train_delta_crosscoder.py` | Delta-Crosscoder (arXiv:2603.04426 mech 1+2) |
| `train_sae.py` | per-model single-SAE baseline (expansion 16) |

### 4_analysis — latent classification, concept mapping & figure data
Decoder-norm categories (shared-all / base-only / gained-at-unpaired / gained-at-paired),
latent-scaling artifact filter, concept-F1 (V/D/J/isotype/region), SHM, steering, pairing.
Key: `classify_only.py`, `concept_labeling.py`(+`_dvj`), `steering_layers.py`,
`pairing_perturbation.py`, `pairing_characterize.py`, `enrichment.py`, `mixed_vj.py`.

Concept-F1 detectors share `concept_f1_core.py` (the InterPLM sorted-threshold max-F1
sweep, encode-and-collect pass, Haque floor). The reported per-concept analysis is
`concept_f1_folds_rebuild95v2.py` (5-fold CV on the size-matched mixed crosscoder, fresh
UniProt labels → `concept_f1_per_concept_rebuild95v2.csv`); antibody-region/V-gene and
unbiased nested-CV detectors are `ab_f1_folds_rebuild95.py` / `ab_detectors_nestedcv.py`.

This stage also holds the encode-and-measure scripts that produce figure data (they load a
checkpoint + activation `.h5`, encode, and write `npz`/`csv` into `5_figures/figdata/`):
`fig3_region_reuse.py`, `fig3c_region_firing.py`, `fig3d_mixed_trace.py`,
`fig3d_toplat_profile.py`, `fig3_mixed_abc_3seed.py`, `mixed_firing_3seed_topk.py`,
`alllayer_decnorm_dist.py`, `umap_mixed_analysis.py`, and `velayer_recover.py` (recovers
VE + decoder-norm summaries from walltime-killed velayer checkpoints).

### 5_figures — publication figures (plot-only renderers)
Two-step architecture: the analysis scripts in `4_analysis/` write frozen figure data
into `5_figures/figdata/` (`*.npz` / `*.csv`); the scripts here **only read those files
and render** — no checkpoints, no activation stores, no encoding. Re-plotting never
re-runs analysis. `paper_style.py` holds the shared matplotlib style imported by the
plotters.

| script | figure(s) | reads |
|---|---|---|
| `plot_fig2a_simplex.py` / `plot_regime_simplex.py` | Fig 2a relative-decoder-norm simplex | `figdata/*simplex*` |
| `plot_fig2bcd_fig3abc_panels.py` | Fig 2b–d, Fig 3a–c panels | `figdata/fig2*`, `fig3ab_*` |
| `plot_ve_ternary_rebuild95.py` | VE ternary (size-matched) | `figdata/*ve*rebuild95*` |
| `plot_regime_comparison_rebuild95.py` | regime comparison | `figdata/regime_comparison_rebuild95.csv` |
| `plot_rebuild95_restyled.py` | restyled rebuild95 panels | `figdata/*rebuild95*` |
| `plot_alllayer_overview.py` / `plot_alllayer_rpair.py` | Fig 1c / S1 VE-across-layers | `figdata/alllayer_*` |
| `make_category_figures.py` | decoder-norm category line-plots, grouped-stacked bars, threshold-free distributions | classification CSVs |

Figure data is documented per-file in `5_figures/figdata/README.md`, and the full
figure→script→checkpoint→store→parquet chain is in `REPRODUCE.md`.

## Classification thresholds (decoder-norm, 3 models sum-normalized to r = 1)

Relative decoder norms `r_prot`, `r_unp`, `r_pair` (sum to 1). Thresholds `HI = 0.5`,
`LO = 0.2`. `HI` gates a single model (only one of three sum-to-1 shares can exceed 0.5);
`LO` gates categories where multiple models must all qualify.

| category | rule | interpretation |
|---|---|---|
| shared-all | all three `r > LO` | every model contributes (germline backbone) |
| base-only | `r_prot > HI` & both IgBERT `< LO` | present in ProtBERT only |
| gained-at-unpaired | `r_prot < LO` & both IgBERT `> LO` | appears at IgBERT fine-tuning |
| gained-at-paired | `r_prot < LO` & `r_unp < LO` & `r_pair > HI` | paired-specific |

Latent-scaling artifact filter: `ν = β_unpaired / |β_paired|`; `ν ≥ 0.5` flags a
sparsity artifact.

## Main parameters

| parameter | value | | parameter | value |
|---|---|---|---|---|
| batch | 4096 | | warmup steps | 1000 |
| k (active latents) | 32 | | auxk_alpha | 0.03125 |
| dict size | 32768 (expansion 32) | | code_normalization | CROSSCODER |
| num_layers (models) | 3 | | target_rms | 1.0 |
| steps | 35000 | | seeds | 0, 1, 2 |
| learning rate | 1.414e-4 | | | |
