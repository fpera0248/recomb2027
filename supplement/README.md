# Supplement analyses

These are the scripts behind the supplement analyses that `reproduce.sh` does not run. They cover the further analyses of the duplication study (Supplementary Section S7) and the further analyses of the hidden-subtype study with Figure S5 (Supplementary Section S17). Each file is a copy of a script we ran. On the `plain-names` branch we renamed 54 script files to descriptive names (the table at the end lists the earlier names), changed the import lines, file names inside run lines, and queue and chain script names to match, and edited comments, docstrings and path defaults (paths now read `HR_WORK`, `SUPP_DATA`, `GF_ROOT`), with defaults that resolve to the original paths. `further/embeddings/gfcfg.py` reads the cell list from `GF_CELLS_CSV`, which has no default because the original was a temporary file that was not released. One non-ASCII string literal in `figures/figS_further.py` was also written as an escape. We checked only that every changed script parses. We did not rerun any script after the renames, so we state that the logic of each script is unchanged and we make no claim that its numbers are identical to the saved ones. Most produced printed numbers. The rest are the helpers, queues and checks they need. For the S7 paragraph on HLCA, `duplication/hlca_classifier_readout.py` (`run`, then `report`) scored the generated cells. The scripts that generated them are not here.

The scripts do not run from this repository alone. Paths inside them default to folders under `~/work` and `~/data` on our machines and the external volume `/Volumes/Files`; set `HR_WORK` to move `~/work`, `SUPP_DATA` to move `~/data` and `GF_ROOT` to move `~/geneformer/Geneformer`. `SUPP_DATA` is unrelated to the `HR_DATA` of the main pipeline. Some scripts also call `$HOME/venv` interpreters. Most read inputs that we did not release: the AIDA, HLCA and lupus files, the generated cells and the subsamples of the 10-atlas set. Read them as a record of what we ran. Script files carry plain descriptive names (the table at the end lists the earlier names). Saved outputs, step keys, output folders, stored tables and the locked input tree still use our working codes, for example `m1_leiden_sub` (Leiden subclustering), `m14p` and `learn14_predict` (the learned scorer), `frozen4` and `preds_frozen4` (our method), `m27*`, `mk16`, `stat16*`, `rep16*` and `gen16*` output names (the donor-disjoint duplication runs), `w2_fm_*` (embedding-analysis output folders), `w6*` (label-corruption outputs), `r5` and `test5` (the test-atlas work folder and split name), and `sr3` (candidate-group outputs). Codes that still appear inside the scripts: `sr2` (the dev variants of the seprank baseline), `m1` to `m8` (the early detectors), `m10` (split-island detector), `m11` (the seprank baseline), `m15x` (rank-combination variants), `sd3clf16` (the four-classifier scDesign3 readout on AIDA that the clip chains call, not included), `ref_j21` (saved reference metrics of the one-versus-rest runs), `gen10` and `gen15` (earlier generator experiments), `fm3` (output folder of the foundation-model reruns), `e4_markers.csv` (marker-recovery output of `duplication/marker_genes/run.py`), and `D1`, `D2` and `D1null` (the development pilot designs and their null versions, used in `further/method_at_submission/`). The scripts load shared modules from the original working tree (`$HR_WORK/r5/code`, `jiaqi1011/code`, `cfgen_natrare`, `STAT16_DIR`), where those modules keep their earlier file names, so the import lines that reach into that tree still use `learn14`, `combo_r2`, `sr2`, `sr3`, `sr3m`, `sr3h`, `frozen4`, `build_r5`, `w6common`, `p2_refine`, `stat16` and `m27_hlclf`. The table at the end of this file maps every renamed file to its earlier name. Path defaults and comments also name folders of the original working tree: `jiaqi1011`, `j1004`, `rev1015`, `monday_q`, `recomb27`, `cfgen_natrare`, `w1`, `jiaqi1016`, `E2_major_types`, `opus_check` (under `tmp/`), `figaudit`, `hvg1015`, `hvg_c3`, `donorfix`, `src_mini`, `from_mini`, `p1data`, `si_v3`, `vm3_m12`, `r4_out`, `v4_integ`, `w6_annot_atac`, `scale_data`, `aida_sg`, `ablation_donor`, `gf6l_emb`, `recomb2027-r8i3wt`, `recomb2027-assets`, `preds_r2` and `preds_r2c` (method predictions and their combinations), `pcaspace_bak` (backups, with `preclip`, `preclip_stat16` and `<D>_sdiff_unclipped` holding outputs from before clipping), and the files `selection_amend2.txt` and `cand_api.json`; the defaults still point to them under `HR_WORK`. `mkh` is the marker and halo feature set of `learned_scorer`, `P4_s1` is the `learned_scorer` scoring rule the shipped scorer uses, and E1 or `E1_union` is the matched-k precision on the union of hidden cells (defined in `score_atl.py`). `scgpt_if` is the scGPT-embedding isolation-forest detector of the published-tool comparison. Comments and the run chains also name working scripts that are not included: `cfgen_aug.py` and `cfgen_knn.py` (our first CFGen augmentation and its kNN readout), `m27_gen_hlca.py`, `m27_gen_hlca_t.py` and `m27_gen_lupus_t.py` (the HLCA and lupus generator scripts), `m27_step1.py`, `m27_sd3clf.py`, `m27_mk.py`, `m27_hardmk.py` and `m27_repl.py` (earlier m27 classifier, marker and replicate readouts), `hl_counts.py` (the HLCA count rebuild), `sd3_arm_nofloor.R` (a copy of `sd3_arm.R` without the floor guard), `run_fm_dup_gf316m.py`, `01_shard_tokenize.py` and `02_embed.py` (our earlier Geneformer embedding scripts), `build_hlca.py` and `p3_pools.py` (earlier pool builders), `evaluate.py` (the evaluation step of `hmod`), `p1_planted_check.py` (a planted-label check), `make_paper_figs.py` and `plot_w1.py` (earlier figure scripts), and `mini_queue.sh`, `worker_sr3*.sh` and `timed.py` (the run queue, its workers and its timing wrapper). `BC1` to `BC4` and `check N` in `build_second_atlas_set.py` are its build-check status keys, and `leak5` and `build5` are its leak-check and build-status output folders. Other working labels in comments: set A = the 8-atlas test set (69 pools, `ablation_donor/data`); set B = the 10-atlas test set (57 pools, `r5/data`, split `test5`); phase-1 pools = the 29 set-B pools with pool index < 3; P1 to P3 are the parts of the label-corruption analysis; ASI, step0b and `step0b_ht` are our earlier scDesign3 augmentation pipeline (a separate project); "the mini" and "the laptop" are the two local machines we ran on, and a rented cloud GPU machine embedded the scGPT cells; `combo_r2` and test4/`r4` are earlier round names.

## Where each analysis lives

Headings follow the supplement. Paths are relative to this folder.

| Supplement analysis | Scripts | Inputs |
|---|---|---|
| One-versus-rest classifiers | `duplication/metrics/one_vs_rest_metrics.py` (run on HLCA and lupus by `run_hlca_then_lupus.sh`), `duplication/one_vs_rest/dup_vs_gen_one_vs_rest.py` | generated cells of AIDA, HLCA and lupus. The scripts that generated the HLCA and lupus cells are not here |
| Genes chosen on training cells only | `duplication/replay_split.py` (with `HVG_TRAIN=1`), then `statistical_augmentations.py` (with `REP=rep16t`) for the four real-cell conditions on AIDA. The generator runs with these genes have no script here | AIDA files |
| Prior-shift control | `duplication/prior_shift/priorshift.py`, on splits from `duplication/replay_split.py` (`HVG_TRAIN=1`, plus `P31=1` for the 31-type set) | AIDA splits with genes chosen on training cells |
| Resampling evaluation donors | `duplication/donor_resampling/evdon.py`, `donor_boot.py` (helper `splits.py`) | lupus and HLCA files and their saved splits |
| Marker recovery | `duplication/dup_vs_generators_savegene.py` (with `SAVE_GENE=mk16`), `duplication/marker_recovery.py`, `duplication/de_marker_recovery.py`, `duplication/marker_genes/run.py` (`marker_recovery.py` and `de_marker_recovery.py` rerun by `duplication/clip_chain.sh`) | AIDA files |
| Rare types that sit close to another type | `duplication/dup_vs_generators_hard_types.py`, `scdiffusion_clipped_regen.py`, `hard_types_classifier.py`, `dup_vs_generators_rawcounts.py` (`scdiffusion_clipped_regen.py` and `hard_types_classifier.py` rerun by `clip_chain.sh`) | AIDA files |
| Unmodified scDesign3 against scVI on donor-disjoint splits | `duplication/dup_vs_generators.py`, `dup_vs_generators_savegene.py`, `sd3_arm.R`, `figures/part1_panel_stats.py` (imports `duplication/panel_analysis.py`) | AIDA files and the released `part1_results.tgz` |
| Where conditional scVI loses recall | `duplication/scvi_with_labels.py`, `figures/part1_panel_stats.py` | AIDA files and `part1_results.tgz` |
| A cell type with no training cells from one ancestry | `duplication/zero_support_task.py` | AIDA files |
| Other latent codes | `duplication/dup_vs_generators_rawcounts.py`, then `latent_unification.py` with `M27_LIB=anchor M27_TAG=_anchor` | saved AIDA cells and raw counts |
| Duplication on foundation-model embeddings | `duplication/foundation_models/cells_needed.py`, `embed_v2.py`, `run_v2.py`, `fm/run_fm_dup.py`, `fm2/analyze_gf6l.py`, `scgpt_cap2x2.py` (`run_v2.py` runs with `POOL=mean` or `cls`, and with `EMB=emb_scgpt*` for scGPT) | AIDA Singaporean subset (355,046 cells) and saved Geneformer 6-layer and scGPT embeddings, whose scripts we did not include |
| Choosing the condition by training-set size | `duplication/condition_by_size/run.py` | `probe31_runs.csv` in `part1_results.tgz` |
| Two further interpolation methods (ISMOTE and Just Mix Once) | `duplication/replay_split.py`, `statistical_augmentations.py`, `aug_li2025.py`, `aug_jmo.py`, `clip_jitter_chain.sh`, `clip_jitter_merge.py` | AIDA splits and `gen16.csv` |
| Cost to major types | `duplication/major_types/run.py` | `part1_results.tgz` and AIDA result tables |
| False positive rates | `duplication/false_positive_rates/run.py`, `duplication/metrics/one_vs_rest_metrics.py`, and `dataset_units.py`, a later check with datasets as units that gives the printed $q$ values | `part1_results.tgz`, the per-row metrics above and the AIDA, HLCA and lupus result tables |
| The generator papers' distance metrics on AIDA | `duplication/dup_vs_generators_savegene.py`, `sd3_arm.R`, lane A of `clip_chain.sh`, `replicate_metrics_fix.py` | saved AIDA cells of each method. The step that rewrote the scDiffusion cells has no script here |
| Time, memory and disk | `further/scale_and_regressors/scale_pipeline.py`, `storage.py` (with `--write`), `fit_worker.py`, `run_fits.sh`, `aggregate.py`, `further/queues/scale_queue.sh`. The embedding times come from `further/embeddings/analyze_embeddings.py`. The average precision at each size and the whole-atlas runs have no script here | subsamples of one atlas at six sizes, the 57 test subsamples and the training matrix |
| Other regressors and training size | `further/scale_and_regressors/fit_worker.py`, `run_fits.sh`, `eval_models.py`, `run_eval.sh`, `aggregate.py`, `parity.py` | the training subsamples and both test sets |
| Other embeddings in place of PCA | `further/embeddings/embed.py`, `embedding_candidates.py`, `score_embeddings.py`, `embedding_qa.py`, `analyze_embeddings.py`, `run_scoring.sh`, and the queues `further/queues/fm_queue_a.sh`, `fm_queue_b.sh`, `fm_queue_c.sh`, `fm_queue_gf.sh`. `fm_plumbing_check.sh` sets up, and `gate_pca.py` and `gf_equality_check.py` are checks | subsamples of the 10-atlas set and the pretrained scGPT and Geneformer models |
| Unreliable broad labels | `further/labels_and_atac/make_variants.py`, `run_unit.py`, `list_units.py`, `leiden_sub_full.py`, `analyze_label_corruption.py`, `further/queues/label_corruption_queue.sh`, `memgate.sh` | subsamples of the 10-atlas set |
| Improving the annotation | `further/queues/matched_classes.sh`, `further/annotations/matched_classes.py`, `further/labels_and_atac/run_unit.py`, `analyze_label_corruption.py`, `annotation_refine.py` (`apply`), `leiden_sub_full.py` | subsamples of the 10-atlas set |
| Chromatin accessibility | `further/queues/atac_all.sh`, which runs `further/labels_and_atac/atac/select_ds.py`, `build_ga.py`, `pools.py`, `qhelp.py`, `aggregate.py` with `run_unit.py` and `leiden_sub_full.py`. `ensembl_gene_annotation.R` and `atac_gene_activity.py` made the gene annotation file that `build_ga.py` reads | six CELLxGENE multiome or ATAC datasets |
| Figure S5 | `figures/figS_further.py`, `figures/paperstyle.py` | result tables of the time, embedding and unreliable-label analyses, which we did not release |


## Shared code

- `duplication/` holds the helpers that the scripts import: `statistical_augmentations.py` (classifiers and matched recall), `polh.py` (the sign-flip test), `sdiff_mod.py` (our scDiffusion wrapper), `hlca_classifier_readout.py` (classifiers for AIDA, HLCA and lupus), `metrics/common.py` (data loading for `one_vs_rest_metrics.py`) and `mem_patch.R` (a memory patch for scDesign3). `dup_vs_generators_rawcounts.py` writes the saved cells of the eight usual types that later scripts read.
- `further/method_at_submission/` is our method as it stood when we ran the further analyses, which import it. `hiddenrare/` at the repository root is the released version.
- `further/embeddings/`, `further/labels_and_atac/` and `further/scale_and_regressors/` also hold shared helpers: `harness.py`, `embedding_lib.py`, `label_corruption_common.py`, `gfcfg.py` and `gfload.py` (Geneformer loading for `embed.py`), `common.sh` (sourced by the embedding queues `fm_queue_*.sh` and `fm_plumbing_check.sh`), `laptop_lock_run.sh` (a lock wrapper that `fm_queue_gf.sh` and `fm_plumbing_check.sh` run) and the pool lists.
- `further/queues/` holds the shell scripts that ran the jobs in order.

We did not copy third-party code. CFGen, scDiffusion and Geneformer load from clones of the authors' repositories. scGPT and scvi-tools come from pip, and scDesign3 and snapATAC2 are installed packages.

## Modules and scripts loaded from locked external trees

The scripts below load code by its original file name from working trees that are not part of this repository. They fail without those trees. Where a released copy exists here, it has a descriptive name and the earlier name is in the table at the end.

| Script | Tree it reads | Names it loads |
|---|---|---|
| `further/embeddings/harness.py`, `further/scale_and_regressors/harness.py` | `HR_WORK/r5/code` | `learn14` (released as `learned_scorer.py`), `hlib` (in `r5/code/hmod`), and `score_atl` (released with its name) |
| `further/embeddings/embedding_lib.py` | `R5CODE`, default `HR_WORK/r5/code` | `loader`, `sr2` (released as `seprank_variants.py`), `sr3`, `sr3m`, `sr3h` |
| `further/scale_and_regressors/scale_pipeline.py` | `HR_WORK/r5/code` | imports `build_r5` (released as `build_second_atlas_set.py`) and runs `sr3.py`, `sr3m.py`, `sr3h.py`, `run_m1_only.py` (released as `run_leiden_sub_only.py`), `learn14.py` and `frozen4.py` (released as `our_method.py`) as subprocesses |
| `further/queues/scale_queue.sh` | `HR_WORK/jiaqi1011/code/w1` | runs the external copy `scale_pipeline.py` of that tree, not the file released in `further/scale_and_regressors/`. The two share a purpose and the released file is the copy of the script we ran, but the queue as released does not call it. We did not change this |
| `further/queues/atac_all.sh` | `HR_WORK/r5/code`, `HR_WORK/jiaqi1011/code/w1` | checks that `sr3`, `sr3m`, `sr3h`, `sr2`, `split_island_v2`, `loader`, `learn14`, `rescore`, `score_atl`, `score_hlca`, `combo_r2`, `hmod/*` and `w1/harness.py` exist before it starts |
| `further/labels_and_atac/run_unit.py`, `annotation_refine.py`, `make_variants.py` | `HR_WORK/jiaqi1011/code/w1`, `HR_WORK/r5/code` | `harness` (the `w1` copy, not the released `further/embeddings/harness.py`), `hlib`, and `sr3.py`, `sr3m.py`, `sr3h.py` run as subprocesses |
| `further/annotations/matched_classes.py` | `HR_WORK/jiaqi1011/code/w6` and `w1` | `w6common`, `p2_refine`, `harness`, `score_hlca` |
| `further/scale_and_regressors/run_fits.sh` | `HR_WORK/jiaqi1011/code/w1` | runs from that folder |
| `further/queues/fm_queue_gf.sh`, `fm_plumbing_check.sh` | `HR_WORK/jiaqi1011/code/w2_fm` | re-run themselves through `laptop_lock_run.sh` of that tree when `W2_LOCKED` is unset |
| `duplication/metrics/common.py` | `HR_WORK/cfgen_natrare` | `m27_hlclf` (released as `hlca_classifier_readout.py`) and `stat16` (released as `statistical_augmentations.py`) |
| `duplication/foundation_models/fm/run_fm_dup.py` | `HR_WORK/cfgen_natrare` | `stat16` |
| `duplication/donor_resampling/donor_boot.py` | `STAT16_DIR` and `M27_SRC` | `stat16`, `polh` and `m27_hlclf` |
| `duplication/clip_chain.sh`, `duplication/clip_jitter_chain.sh`, `duplication/metrics/run_hlca_then_lupus.sh` | `HR_WORK/cfgen_natrare` | change into that tree and call scripts there by the file names written in the chain, so the released scripts must be copied into that folder first. They also call scripts that are not released (`m27_sd3clf.py`, `sd3clf16.py`) |

`duplication/prior_shift/priorshift.py` is not in this list. It imports the released `statistical_augmentations.py` and `polh.py` from the `duplication` folder, and `STAT16_DIR` can point to another folder that holds both files.

Names that stay because saved outputs or the trees above use them:

- Output folders and files `preds_m14p/m14p_frozen/`, `preds_frozen4/frozen4/` and `scores_preds_frozen4.csv`, which `further/method_at_submission/our_method.py`, `further/scale_and_regressors/parity.py` and `further/labels_and_atac/analyze_label_corruption.py` read or write. Saved results and the external `frozen4.py` use these paths.
- Step ids `sr3`, `sr3m`, `sr3h`, `m1_leiden_sub`, `learn14_predict` and `frozen4` in `scale.csv`, which `figures/figS_further.py` reads.
- Environment variables `STAT16_DIR`, `M27_SRC`, `M27_HL`, `M27_CSV`, `M27_LIB`, `M27_TAG`, `W2_LOCKED`, `W2_OUT`, `W6_ROOT`, `W6_CLEAN`, `W6_OUT` and `W6_OOD` (and the other `M27_*`, `W2_*` and `W6_*` names), which the run lines in this README and the external modules set or read.

## Earlier file names

| Earlier name | Current name |
|---|---|
| `duplication/m27_clip_chain.sh` | `duplication/clip_chain.sh` |
| `duplication/m27_clipj_merge.py` | `duplication/clip_jitter_merge.py` |
| `duplication/m27_de.py` | `duplication/de_marker_recovery.py` |
| `duplication/gen16.py` | `duplication/dup_vs_generators.py` |
| `duplication/m27_gen_hard.py` | `duplication/dup_vs_generators_hard_types.py` |
| `duplication/m27_gen_base.py` | `duplication/dup_vs_generators_rawcounts.py` |
| `duplication/gen16m.py` | `duplication/dup_vs_generators_savegene.py` |
| `duplication/m27_hardclf.py` | `duplication/hard_types_classifier.py` |
| `duplication/m27_hlclf.py` | `duplication/hlca_classifier_readout.py` |
| `duplication/m27_latent.py` | `duplication/latent_unification.py` |
| `duplication/mk16.py` | `duplication/marker_recovery.py` |
| `duplication/one_vs_rest/j21_dup_vs_gen.py` | `duplication/one_vs_rest/dup_vs_gen_one_vs_rest.py` |
| `duplication/replay16.py` | `duplication/replay_split.py` |
| `duplication/m27_repl_fix.py` | `duplication/replicate_metrics_fix.py` |
| `duplication/m27_sdiff_regen.py` | `duplication/scdiffusion_clipped_regen.py` |
| `duplication/gen12.py` | `duplication/scvi_with_labels.py` |
| `duplication/stat16.py` | `duplication/statistical_augmentations.py` |
| `duplication/gen14.py` | `duplication/zero_support_task.py` |
| `figures/p1_panel_stats.py` | `figures/part1_panel_stats.py` |
| `further/annotations/r3c_matched.py` | `further/annotations/matched_classes.py` |
| `further/embeddings/w2_analyze.py` | `further/embeddings/analyze_embeddings.py` |
| `further/embeddings/w2_common.sh` | `further/embeddings/common.sh` |
| `further/embeddings/w2_embed.py` | `further/embeddings/embed.py` |
| `further/embeddings/w2_cands.py` | `further/embeddings/embedding_candidates.py` |
| `further/embeddings/w2lib.py` | `further/embeddings/embedding_lib.py` |
| `further/embeddings/w2_qa.py` | `further/embeddings/embedding_qa.py` |
| `further/embeddings/w2_gate_pca.py` | `further/embeddings/gate_pca.py` |
| `further/embeddings/w2_score.py` | `further/embeddings/score_embeddings.py` |
| `further/labels_and_atac/analyze_w6.py` | `further/labels_and_atac/analyze_label_corruption.py` |
| `further/labels_and_atac/p2_refine.py` | `further/labels_and_atac/annotation_refine.py` |
| `further/labels_and_atac/p3_build.py` | `further/labels_and_atac/atac_gene_activity.py` |
| `further/labels_and_atac/w6common.py` | `further/labels_and_atac/label_corruption_common.py` |
| `further/labels_and_atac/m1_full.py` | `further/labels_and_atac/leiden_sub_full.py` |
| `further/method_at_submission/build_atlas5.py` | `further/method_at_submission/build_atlas_pools.py` |
| `further/method_at_submission/build_r5.py` | `further/method_at_submission/build_second_atlas_set.py` |
| `further/method_at_submission/sr3.py` | `further/method_at_submission/candidate_dump.py` |
| `further/method_at_submission/sr3h.py` | `further/method_at_submission/candidate_halo_features.py` |
| `further/method_at_submission/sr3m.py` | `further/method_at_submission/candidate_marker_features.py` |
| `further/method_at_submission/learn14.py` | `further/method_at_submission/learned_scorer.py` |
| `further/method_at_submission/frozen4.py` | `further/method_at_submission/our_method.py` |
| `further/method_at_submission/combo_r2.py` | `further/method_at_submission/rank_combination_sweep.py` |
| `further/method_at_submission/run_m1_only.py` | `further/method_at_submission/run_leiden_sub_only.py` |
| `further/method_at_submission/sr2.py` | `further/method_at_submission/seprank_variants.py` |
| `further/method_at_submission/split_island_v2.py` | `further/method_at_submission/split_island_detector.py` |
| `further/queues/w2_fm_plumb.sh` | `further/queues/fm_plumbing_check.sh` |
| `further/queues/w2_fm_a.sh` | `further/queues/fm_queue_a.sh` |
| `further/queues/w2_fm_b.sh` | `further/queues/fm_queue_b.sh` |
| `further/queues/w2_fm_c.sh` | `further/queues/fm_queue_c.sh` |
| `further/queues/w2_fm_gf.sh` | `further/queues/fm_queue_gf.sh` |
| `further/queues/w6_p1.sh` | `further/queues/label_corruption_queue.sh` |
| `further/queues/r3c_matched.sh` | `further/queues/matched_classes.sh` |
| `further/queues/w1_scale.sh` | `further/queues/scale_queue.sh` |
| `duplication/m27_clipj_chain.sh` | `duplication/clip_jitter_chain.sh` |
| `duplication/metrics/run_e1.py` | `duplication/metrics/one_vs_rest_metrics.py` |
| `further/labels_and_atac/p3_gene_anno.R` | `further/labels_and_atac/ensembl_gene_annotation.R` |
