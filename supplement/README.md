# Supplement analyses

These are the scripts behind the supplement analyses that `reproduce.sh` does not run. They cover the further analyses of the duplication study (Supplementary Section S7) and the further analyses of the hidden-subtype study with Figure S5 (Supplementary Section S17). Each file is an unchanged copy of a script we ran. Most produced printed numbers. The rest are the helpers, queues and checks they need. For the S7 paragraph on HLCA, `duplication/m27_hlclf.py` (`run`, then `report`) scored the generated cells. The scripts that generated them are not here.

The scripts do not run from this repository alone. Paths inside them default to folders under `~/work` and `~/data` on our machines and the external volume `/Volumes/Files`; set `HR_WORK` or `HR_DATA` to move the first two. Most read inputs that we did not release: the AIDA, HLCA and lupus files, the generated cells and the subsamples of the 10-atlas set. Read them as a record of what we ran. File and directory names carry our working codes, which are read from disk and so kept: `sr3`, `sr3m`, `sr3h` (candidate groups, marker and halo features), `learn14` and `m14p` (the learned scorer), `m1_leiden_sub` (Leiden subclustering), `frozen4` (the final blend, "our method" in the paper), `build_atlas5` and `build_r5` (the pool builders for the test atlases), `r5` and `test5` (the test-atlas work folder and split name).

## Where each analysis lives

Headings follow the supplement. Paths are relative to this folder.

| Supplement analysis | Scripts | Inputs |
|---|---|---|
| One-versus-rest classifiers | `duplication/metrics/run_e1.py` (run on HLCA and lupus by `run_hlca_then_lupus.sh`), `duplication/one_vs_rest/j21_dup_vs_gen.py` | generated cells of AIDA, HLCA and lupus. The scripts that generated the HLCA and lupus cells are not here |
| Genes chosen on training cells only | `duplication/replay16.py` (with `HVG_TRAIN=1`), then `stat16.py` (with `REP=rep16t`) for the four real-cell conditions on AIDA. The generator runs with these genes have no script here | AIDA files |
| Prior-shift control | `duplication/prior_shift/priorshift.py`, on splits from `duplication/replay16.py` (`HVG_TRAIN=1`, plus `P31=1` for the 31-type set) | AIDA splits with genes chosen on training cells |
| Resampling evaluation donors | `duplication/donor_resampling/evdon.py`, `donor_boot.py` (helper `splits.py`) | lupus and HLCA files and their saved splits |
| Marker recovery | `duplication/gen16m.py` (with `SAVE_GENE=mk16`), `duplication/mk16.py`, `duplication/m27_de.py`, `duplication/marker_genes/run.py` (`mk16.py` and `m27_de.py` rerun by `duplication/m27_clip_chain.sh`) | AIDA files |
| Rare types that sit close to another type | `duplication/m27_gen_hard.py`, `m27_sdiff_regen.py`, `m27_hardclf.py`, `m27_gen_base.py` (`m27_sdiff_regen.py` and `m27_hardclf.py` rerun by `m27_clip_chain.sh`) | AIDA files |
| Unmodified scDesign3 against scVI on donor-disjoint splits | `duplication/gen16.py`, `gen16m.py`, `sd3_arm.R`, `figures/p1_panel_stats.py` (imports `duplication/panel_analysis.py`) | AIDA files and the released `part1_results.tgz` |
| Where conditional scVI loses recall | `duplication/gen12.py`, `figures/p1_panel_stats.py` | AIDA files and `part1_results.tgz` |
| A cell type with no training cells from one ancestry | `duplication/gen14.py` | AIDA files |
| Other latent codes | `duplication/m27_gen_base.py`, then `m27_latent.py` with `M27_LIB=anchor M27_TAG=_anchor` | saved AIDA cells and raw counts |
| Duplication on foundation-model embeddings | `duplication/foundation_models/cells_needed.py`, `embed_v2.py`, `run_v2.py`, `fm/run_fm_dup.py`, `fm2/analyze_gf6l.py`, `scgpt_cap2x2.py` (`run_v2.py` runs with `POOL=mean` or `cls`, and with `EMB=emb_scgpt*` for scGPT) | AIDA Singaporean subset (355,046 cells) and saved Geneformer 6-layer and scGPT embeddings, whose scripts we did not include |
| Choosing the condition by training-set size | `duplication/condition_by_size/run.py` | `probe31_runs.csv` in `part1_results.tgz` |
| Two further interpolation methods (ISMOTE and Just Mix Once) | `duplication/replay16.py`, `stat16.py`, `aug_li2025.py`, `aug_jmo.py`, `m27_clipj_chain.sh`, `m27_clipj_merge.py` | AIDA splits and `gen16.csv` |
| Cost to major types | `duplication/major_types/run.py` | `part1_results.tgz` and AIDA result tables |
| False positive rates | `duplication/false_positive_rates/run.py`, `duplication/metrics/run_e1.py`, and `dataset_units.py`, a later check with datasets as units that gives the printed $q$ values | `part1_results.tgz`, the per-row metrics above and the AIDA, HLCA and lupus result tables |
| The generator papers' distance metrics on AIDA | `duplication/gen16m.py`, `sd3_arm.R`, lane A of `m27_clip_chain.sh`, `m27_repl_fix.py` | saved AIDA cells of each method. The step that rewrote the scDiffusion cells has no script here |
| Time, memory and disk | `further/scale_and_regressors/scale_pipeline.py`, `storage.py` (with `--write`), `fit_worker.py`, `run_fits.sh`, `aggregate.py`, `further/queues/w1_scale.sh`. The embedding times come from `further/embeddings/w2_analyze.py`. The average precision at each size and the whole-atlas runs have no script here | subsamples of one atlas at six sizes, the 57 test subsamples and the training matrix |
| Other regressors and training size | `further/scale_and_regressors/fit_worker.py`, `run_fits.sh`, `eval_models.py`, `run_eval.sh`, `aggregate.py`, `parity.py` | the training subsamples and both test sets |
| Other embeddings in place of PCA | `further/embeddings/w2_embed.py`, `w2_cands.py`, `w2_score.py`, `w2_qa.py`, `w2_analyze.py`, `run_scoring.sh`, and the queues `further/queues/w2_fm_a.sh`, `w2_fm_b.sh`, `w2_fm_c.sh`, `w2_fm_gf.sh`. `w2_fm_plumb.sh` sets up, and `w2_gate_pca.py` and `gf_equality_check.py` are checks | subsamples of the 10-atlas set and the pretrained scGPT and Geneformer models |
| Unreliable broad labels | `further/labels_and_atac/make_variants.py`, `run_unit.py`, `list_units.py`, `m1_full.py`, `analyze_w6.py`, `further/queues/w6_p1.sh`, `memgate.sh` | subsamples of the 10-atlas set |
| Improving the annotation | `further/queues/r3c_matched.sh`, `further/annotations/r3c_matched.py`, `further/labels_and_atac/run_unit.py`, `analyze_w6.py`, `p2_refine.py` (`apply`), `m1_full.py` | subsamples of the 10-atlas set |
| Chromatin accessibility | `further/queues/atac_all.sh`, which runs `further/labels_and_atac/atac/select_ds.py`, `build_ga.py`, `pools.py`, `qhelp.py`, `aggregate.py` with `run_unit.py` and `m1_full.py`. `p3_gene_anno.R` and `p3_build.py` made the gene annotation file that `build_ga.py` reads | six CELLxGENE multiome or ATAC datasets |
| Figure S5 | `figures/figS_further.py`, `figures/paperstyle.py` | result tables of the time, embedding and unreliable-label analyses, which we did not release |


## Shared code

- `duplication/` holds the helpers that the scripts import: `stat16.py` (classifiers and matched recall), `polh.py` (the sign-flip test), `sdiff_mod.py` (our scDiffusion wrapper), `m27_hlclf.py` (classifiers for AIDA, HLCA and lupus), `metrics/common.py` (data loading for `run_e1.py`) and `mem_patch.R` (a memory patch for scDesign3). `m27_gen_base.py` writes the saved cells of the eight usual types that later scripts read.
- `further/method_at_submission/` is our method as it stood when we ran the further analyses, which import it. `hiddenrare/` at the repository root is the released version.
- `further/embeddings/`, `further/labels_and_atac/` and `further/scale_and_regressors/` also hold shared helpers: `harness.py`, `w2lib.py`, `w6common.py`, `gfcfg.py` and `gfload.py` (Geneformer loading for `w2_embed.py`), `w2_common.sh` (sourced by the w2 queues), `laptop_lock_run.sh` (a lock wrapper that `w2_fm_gf.sh` and `w2_fm_plumb.sh` run) and the pool lists.
- `further/queues/` holds the shell scripts that ran the jobs in order.

We did not copy third-party code. CFGen, scDiffusion and Geneformer load from clones of the authors' repositories. scGPT and scvi-tools come from pip, and scDesign3 and snapATAC2 are installed packages.
