# rebuild95 paper-draft figure data (Fig 2 + Fig 3)

Per-panel plotting data for the rebuild95 regeneration of the interpretability
paper draft figures. Each CSV is the exact input to its panel; the plotters in
`../` read the source `.npz`/`.json` but these CSVs carry the same values in
plain text for reproducibility. Round-trip verified (Fig 2c fractions reproduce
to the decimal).

## Files
- `fig2bc_domain_firing_perlatent.csv`  — mixed crosscoder, 3 seeds, per active latent:
  category + ab/general firing freq + log2 ratio. Encode: use_threshold=FALSE
  (BatchTopK, mixed_label.py convention). Feeds Fig 2b (density) and Fig 2c
  (%-specific bars: freq>=0.05%, ratio>=32x).
- `fig2d_rpair_perlatent.csv`  — 3 seeds, per active latent: 3-way relative decoder
  norms (r_prot,r_unp,r_pair) summing to 1. Fig 2d plots r_pair (centered ~1/3;
  paired-specific region >=0.6). fig_normdist convention (make_category_figures.py).
- `fig3ab_region_f1_perlatent.csv` + `fig3ab_region_floors.csv`  — paired crosscoder,
  3 seeds, per-latent x per-region F1max + per-region floor + category + is_detector.
  Encode: use_threshold=TRUE (concept_labeling.py). Fig 3a counts detectors by
  region x category; Fig 3b top-100 F1 scatter vs floor.
- `fig3c_region_firing.csv`  — mixed model, per active latent: is_region_detector +
  ab/gen firing + log2 ratio. Fig 3c: region detectors vs all-active background.
- `fig3d_example_trace.csv`  — one example heavy chain (seq 25369, len 123): per residue
  true region + each detected region's top-detector activation. Fig 3d trace.
- `fig3efg_umap.csv`  — mixed decoder UMAP: per active latent x/y + category + firing +
  per-concept F1 (12 concepts). Fig 3e (category), 3f (firing ratio), 3g (concept clouds).
- `fig3g_concept_floors.csv`  — per-concept detection floor + n_detectors.

## Encode conventions (match original code)
- Fig 2b/2c domain density: use_threshold=FALSE (mixed_label.py)
- Fig 3 region/concept detectors: use_threshold=TRUE (concept_labeling.py L143)

## Regenerate
`python ../plot_fig2bcd_fig3abc_panels.py` (reads the source npz; same values as these CSVs).
